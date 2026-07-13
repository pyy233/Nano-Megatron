"""Simple all-forward/all-backward GPipe reference schedule."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor

from ..p2p import P2PCommunicator
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


class GPipeSchedule:
    def __init__(
        self,
        parallel: Any,
        communicator: P2PCommunicator | None = None,
        *,
        compute_context: ComputeContext | None = None,
    ) -> None:
        self.parallel = parallel
        self.communicator = communicator
        self.compute_context = compute_context or no_compute_context

    def _first(self) -> bool:
        return self.parallel.is_pipeline_first_stage()

    def _last(self) -> bool:
        return self.parallel.is_pipeline_last_stage()

    def forward_backward(
        self,
        *,
        stage: PipelineStage,
        microbatches: Sequence[Any],
        data_parallel: Any = None,
        forward_only: bool = False,
    ) -> StepOutput:
        if not microbatches:
            raise ValueError("at least one microbatch is required")

        records: list[tuple[Tensor | None, Tensor, Tensor | None]] = []
        losses: list[Tensor] = []
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}

        prepare_pipeline_stage(stage)
        for index, batch in enumerate(microbatches):
            hidden = None if self._first() else self.communicator.recv_forward()
            # GPipe drains backward in reverse order, so the first forward is
            # the graph whose backward performs the final DDP reduction.
            with forward_data_parallel_context(
                data_parallel,
                synchronize_gradients=forward_only or index == 0,
            ), self.compute_context():
                output = stage(hidden, batch)
            if self._last():
                loss, local_metrics = extract_loss(output, len(microbatches))
                losses.append(loss)
                accumulate_metrics(metric_sums, metric_counts, local_metrics)
                records.append((hidden, loss, None))
            else:
                assert isinstance(output, Tensor)
                self.communicator.send_forward(output)
                records.append((hidden, output, None))

        if not forward_only:
            for reverse_index, (hidden, output, _) in enumerate(reversed(records)):
                is_last_backward = reverse_index == len(records) - 1
                with data_parallel_context(
                    data_parallel, is_last_microbatch=is_last_backward
                ):
                    output_gradient = None if self._last() else self.communicator.recv_backward()
                    output_gradient = match_output_gradient(output, output_gradient)
                    backward(data_parallel, output, output_gradient)
                if hidden is not None:
                    if hidden.grad is None:
                        raise RuntimeError("pipeline input gradient was not produced")
                    self.communicator.send_backward(hidden.grad)

            finalize_pipeline_stage_gradients(stage)

        return StepOutput(tuple(losses), average_metrics(metric_sums, metric_counts))
