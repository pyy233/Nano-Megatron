"""Interleaved one-forward/one-backward plan on the common executor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..p2p import P2PCommunicator
from ..partition import VirtualPipelineLayout
from ..stage import PipelineStage, StepOutput
from .base import ComputeContext
from .executor import (
    PipelineEvent,
    PipelineScheduleExecutor,
    PipelineWork,
    VirtualPipelineRoute,
)


def build_interleaved_schedule_table(
    num_microbatches: int,
    num_chunks: int,
    microbatch_group_size: int,
) -> tuple[PipelineWork, ...]:
    if num_microbatches < 1:
        raise ValueError("num_microbatches must be positive")
    if num_chunks < 1:
        raise ValueError("num_chunks must be positive")
    if not 1 <= microbatch_group_size <= num_microbatches:
        raise ValueError("microbatch_group_size must be between 1 and num_microbatches")
    table: list[PipelineWork] = []
    for group_start in range(0, num_microbatches, microbatch_group_size):
        group_end = min(group_start + microbatch_group_size, num_microbatches)
        for chunk in range(num_chunks):
            table.extend(
                PipelineWork(microbatch, chunk) for microbatch in range(group_start, group_end)
            )
    return tuple(table)


def build_interleaved_plan(
    *,
    pipeline_size: int,
    pipeline_rank: int,
    num_microbatches: int,
    num_chunks: int,
    forward_only: bool = False,
) -> tuple[PipelineEvent, ...]:
    if pipeline_size < 2:
        raise ValueError("interleaved pipeline requires pipeline_size >= 2")
    if not 0 <= pipeline_rank < pipeline_size:
        raise ValueError("pipeline_rank is outside the pipeline")
    if num_chunks < 2:
        raise ValueError("interleaved pipeline requires at least two model chunks")
    if num_microbatches < 1:
        raise ValueError("num_microbatches must be positive")
    if forward_only:
        forward_table = build_interleaved_schedule_table(
            num_microbatches,
            num_chunks,
            min(pipeline_size, num_microbatches),
        )
        return tuple(PipelineEvent("forward", forward=work) for work in forward_table)
    if num_microbatches < pipeline_size or num_microbatches % pipeline_size:
        raise ValueError(
            "interleaved pipeline requires num_microbatches to be a positive "
            "multiple of pipeline_size"
        )

    forward_table = build_interleaved_schedule_table(
        num_microbatches,
        num_chunks,
        pipeline_size,
    )
    backward_table = tuple(
        PipelineWork(work.microbatch, num_chunks - work.chunk - 1) for work in forward_table
    )
    total = len(forward_table)
    warmup = min(
        total,
        (pipeline_size - pipeline_rank - 1) * 2 + (num_chunks - 1) * pipeline_size,
    )
    events: list[PipelineEvent] = []
    events.extend(PipelineEvent("warmup", forward=forward_table[index]) for index in range(warmup))
    remaining = total - warmup
    for index in range(remaining):
        events.append(
            PipelineEvent(
                "steady",
                forward=forward_table[warmup + index],
                backward=backward_table[index],
            )
        )
    events.extend(
        PipelineEvent("cooldown", backward=backward_table[index])
        for index in range(remaining, total)
    )
    return tuple(events)


class InterleavedOneForwardOneBackwardSchedule:
    """Virtual 1F1B differs only by its plan and virtual route."""

    def __init__(
        self,
        parallel: Any,
        layout: VirtualPipelineLayout,
        communicator: P2PCommunicator,
        *,
        compute_context: ComputeContext | None = None,
        overlap_p2p: bool = False,
    ) -> None:
        if layout.virtual_stages_per_rank < 2:
            raise ValueError("interleaved schedule requires at least two local chunks")
        self.parallel = parallel
        self.layout = layout
        self.communicator = communicator
        self.executor = PipelineScheduleExecutor(
            route=VirtualPipelineRoute(parallel, layout),
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
            events=build_interleaved_plan(
                pipeline_size=self.layout.pipeline_size,
                pipeline_rank=int(self.parallel.pp.rank),
                num_microbatches=len(microbatches),
                num_chunks=self.layout.virtual_stages_per_rank,
                forward_only=forward_only,
            ),
            data_parallel=data_parallel,
            forward_only=forward_only,
        )
