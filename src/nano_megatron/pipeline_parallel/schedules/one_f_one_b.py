"""Non-interleaved one-forward/one-backward pipeline schedule."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from ..p2p import P2PCommunicator, _pipeline_group
from ..stage import PipelineStage, StepOutput
from .base import (
    ComputeContext,
    accumulate_metrics,
    average_metrics,
    backward,
    data_parallel_context,
    extract_loss,
    finalize_pipeline_stage_gradients,
    forward_data_parallel_context,
    match_output_gradient,
    no_compute_context,
    prepare_pipeline_stage,
)


@dataclass(frozen=True)
class ScheduleEvent:
    phase: str
    kind: str
    microbatch: int


def build_1f1b_plan(
    pipeline_size: int, pipeline_rank: int, num_microbatches: int
) -> tuple[ScheduleEvent, ...]:
    """Return an inspectable logical event plan used by CPU unit tests and tracing."""

    if not 0 <= pipeline_rank < pipeline_size:
        raise ValueError("invalid pipeline rank")
    if num_microbatches < 1:
        raise ValueError("num_microbatches must be positive")
    warmup = min(pipeline_size - pipeline_rank - 1, num_microbatches)
    events: list[ScheduleEvent] = []
    for microbatch in range(warmup):
        events.append(ScheduleEvent("warmup", "forward", microbatch))
    for microbatch in range(warmup, num_microbatches):
        events.append(ScheduleEvent("steady", "forward", microbatch))
        events.append(ScheduleEvent("steady", "backward", microbatch - warmup))
    for microbatch in range(num_microbatches - warmup, num_microbatches):
        events.append(ScheduleEvent("cooldown", "backward", microbatch))
    return tuple(events)


class OneForwardOneBackwardSchedule:
    """Readable non-interleaved 1F1B implementation for static activation shapes."""

    def __init__(
        self,
        parallel: Any,
        communicator: P2PCommunicator | None = None,
        *,
        compute_context: ComputeContext | None = None,
    ) -> None:
        self.parallel = parallel
        self.group = _pipeline_group(parallel)
        self.communicator = communicator
        self.compute_context = compute_context or no_compute_context

    def _rank(self) -> int:
        return int(self.group.rank)

    def _size(self) -> int:
        return int(self.group.size)

    def _first(self) -> bool:
        return self.parallel.is_pipeline_first_stage()

    def _last(self) -> bool:
        return self.parallel.is_pipeline_last_stage()

    def _forward(
        self,
        stage: PipelineStage,
        hidden: Tensor | None,
        batch: Any,
        divisor: int,
        data_parallel: Any,
        *,
        synchronize_gradients: bool,
    ) -> tuple[Tensor, Tensor | None, dict[str, float]]:
        with forward_data_parallel_context(
            data_parallel,
            synchronize_gradients=synchronize_gradients,
        ), self.compute_context():
            output = stage(hidden, batch)
        if self._last():
            loss, metrics = extract_loss(output, divisor)
            return loss, loss, metrics
        if not isinstance(output, Tensor):
            raise TypeError("non-last pipeline stages must return a Tensor")
        return output, None, {}

    def forward_backward(
        self,
        *,
        stage: PipelineStage,
        microbatches: Sequence[Any],
        data_parallel: Any = None,
        forward_only: bool = False,
    ) -> StepOutput:
        if forward_only:
            # GPipe is the simpler and memory-safe forward-only schedule.
            from .gpipe import GPipeSchedule

            return GPipeSchedule(
                self.parallel,
                self.communicator,
                compute_context=self.compute_context,
            ).forward_backward(
                stage=stage,
                microbatches=microbatches,
                data_parallel=data_parallel,
                forward_only=True,
            )
        if not microbatches:
            raise ValueError("at least one microbatch is required")
        if self._size() == 1:
            from .gpipe import GPipeSchedule

            return GPipeSchedule(
                self.parallel,
                self.communicator,
                compute_context=self.compute_context,
            ).forward_backward(
                stage=stage,
                microbatches=microbatches,
                data_parallel=data_parallel,
            )
        if self.communicator is None:
            raise ValueError("a P2PCommunicator is required when pipeline size is greater than one")

        num_microbatches = len(microbatches)
        warmup = min(self._size() - self._rank() - 1, num_microbatches)
        remaining = num_microbatches - warmup
        queue: list[tuple[Tensor | None, Tensor]] = []
        losses: list[Tensor] = []
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        prepare_pipeline_stage(stage)

        # Warmup forwards fill the pipeline.
        for index in range(warmup):
            hidden = None if self._first() else self.communicator.recv_forward()
            output, loss, local_metrics = self._forward(
                stage,
                hidden,
                microbatches[index],
                num_microbatches,
                data_parallel,
                synchronize_gradients=index == num_microbatches - 1,
            )
            if not self._last():
                self.communicator.send_forward(output)
            if loss is not None:
                losses.append(loss)
            accumulate_metrics(metric_sums, metric_counts, local_metrics)
            queue.append((hidden, output))

        current_hidden = (
            None
            if remaining == 0 or self._first()
            else self.communicator.recv_forward()
        )
        for steady_index in range(remaining):
            microbatch_index = warmup + steady_index
            output, loss, local_metrics = self._forward(
                stage,
                current_hidden,
                microbatches[microbatch_index],
                num_microbatches,
                data_parallel,
                synchronize_gradients=microbatch_index == num_microbatches - 1,
            )
            queue.append((current_hidden, output))
            if loss is not None:
                losses.append(loss)
            accumulate_metrics(metric_sums, metric_counts, local_metrics)

            output_gradient = (
                None
                if self._last()
                else self.communicator.send_forward_recv_backward(output)
            )
            backward_hidden, backward_output = queue.pop(0)
            output_gradient = match_output_gradient(backward_output, output_gradient)
            is_final_backward = steady_index == remaining - 1 and warmup == 0
            with data_parallel_context(
                data_parallel, is_last_microbatch=is_final_backward
            ):
                backward(data_parallel, backward_output, output_gradient)

            input_gradient = backward_hidden.grad if backward_hidden is not None else None
            receive_next = steady_index < remaining - 1
            current_hidden = self.communicator.send_backward_recv_forward(
                input_gradient, receive_forward=receive_next
            )

        # Drain warmup activations.
        for cooldown_index, (hidden, output) in enumerate(queue):
            output_gradient = None if self._last() else self.communicator.recv_backward()
            output_gradient = match_output_gradient(output, output_gradient)
            is_final_backward = cooldown_index == len(queue) - 1
            with data_parallel_context(
                data_parallel, is_last_microbatch=is_final_backward
            ):
                backward(data_parallel, output, output_gradient)
            if hidden is not None:
                if hidden.grad is None:
                    raise RuntimeError("pipeline input gradient was not produced")
                self.communicator.send_backward(hidden.grad)

        finalize_pipeline_stage_gradients(stage)

        return StepOutput(tuple(losses), average_metrics(metric_sums, metric_counts))
