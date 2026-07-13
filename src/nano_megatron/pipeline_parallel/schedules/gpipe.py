"""All-forward/all-backward GPipe plan on the common executor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..p2p import P2PCommunicator
from ..stage import PipelineStage, StepOutput
from .base import ComputeContext
from .executor import (
    LinearPipelineRoute,
    PipelineEvent,
    PipelineScheduleExecutor,
    PipelineWork,
)


def build_gpipe_plan(
    num_microbatches: int,
    *,
    forward_only: bool = False,
) -> tuple[PipelineEvent, ...]:
    if num_microbatches < 1:
        raise ValueError("num_microbatches must be positive")
    events = [
        PipelineEvent("forward", forward=PipelineWork(index)) for index in range(num_microbatches)
    ]
    if not forward_only:
        events.extend(
            PipelineEvent("backward", backward=PipelineWork(index))
            for index in reversed(range(num_microbatches))
        )
    return tuple(events)


class GPipeSchedule:
    """GPipe differs only by its plan; execution is shared by all schedules."""

    def __init__(
        self,
        parallel: Any,
        communicator: P2PCommunicator | None = None,
        *,
        compute_context: ComputeContext | None = None,
        overlap_p2p: bool = False,
    ) -> None:
        self.parallel = parallel
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
            events=build_gpipe_plan(
                len(microbatches),
                forward_only=forward_only,
            ),
            data_parallel=data_parallel,
            forward_only=forward_only,
        )
