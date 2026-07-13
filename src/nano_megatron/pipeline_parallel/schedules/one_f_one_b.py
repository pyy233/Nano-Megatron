"""Non-interleaved one-forward/one-backward plan on the common executor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..p2p import P2PCommunicator, _pipeline_group
from ..stage import PipelineStage, StepOutput
from .base import ComputeContext
from .executor import (
    LinearPipelineRoute,
    PipelineEvent,
    PipelineScheduleExecutor,
    PipelineWork,
)


def build_1f1b_plan(
    pipeline_size: int,
    pipeline_rank: int,
    num_microbatches: int,
    *,
    forward_only: bool = False,
) -> tuple[PipelineEvent, ...]:
    """Return the rank-local atomic compute order for non-interleaved 1F1B."""

    if pipeline_size < 1:
        raise ValueError("pipeline_size must be positive")
    if not 0 <= pipeline_rank < pipeline_size:
        raise ValueError("invalid pipeline rank")
    if num_microbatches < 1:
        raise ValueError("num_microbatches must be positive")
    if forward_only:
        return tuple(
            PipelineEvent(
                "forward",
                forward=PipelineWork(microbatch),
            )
            for microbatch in range(num_microbatches)
        )

    warmup = min(pipeline_size - pipeline_rank - 1, num_microbatches)
    events: list[PipelineEvent] = []
    for microbatch in range(warmup):
        events.append(
            PipelineEvent(
                "warmup",
                forward=PipelineWork(microbatch),
            )
        )
    for microbatch in range(warmup, num_microbatches):
        events.append(
            PipelineEvent(
                "steady",
                forward=PipelineWork(microbatch),
            )
        )
        events.append(
            PipelineEvent(
                "steady",
                backward=PipelineWork(microbatch - warmup),
            )
        )
    for microbatch in range(num_microbatches - warmup, num_microbatches):
        events.append(
            PipelineEvent(
                "cooldown",
                backward=PipelineWork(microbatch),
            )
        )
    return tuple(events)


class OneForwardOneBackwardSchedule:
    """Non-interleaved 1F1B differs only by its rank-specific event plan."""

    def __init__(
        self,
        parallel: Any,
        communicator: P2PCommunicator | None = None,
        *,
        compute_context: ComputeContext | None = None,
        overlap_p2p: bool = False,
    ) -> None:
        self.parallel = parallel
        self.group = _pipeline_group(parallel)
        self.communicator = communicator
        self.executor = PipelineScheduleExecutor(
            route=LinearPipelineRoute(parallel),
            communicator=communicator,
            compute_context=compute_context,
            overlap_p2p=overlap_p2p,
        )

    def forward_backward(
        self,
        *,
        stage: PipelineStage,
        microbatches: Sequence[Any],
        data_parallel: Any = None,
        forward_only: bool = False,
    ) -> StepOutput:
        return self.executor.run(
            stage=stage,
            microbatches=microbatches,
            events=build_1f1b_plan(
                int(self.group.size),
                int(self.group.rank),
                len(microbatches),
                forward_only=forward_only,
            ),
            data_parallel=data_parallel,
            forward_only=forward_only,
        )
