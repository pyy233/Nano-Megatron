"""Deterministic Transformer-layer partitioning across pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerPartition:
    stage: int
    start_layer: int
    end_layer: int
    owns_embedding: bool
    owns_final_norm: bool
    owns_lm_head: bool

    @property
    def num_layers(self) -> int:
        return self.end_layer - self.start_layer


@dataclass(frozen=True)
class PipelineStageAddress:
    """One logical stage hosted by a physical PP rank."""

    pp_rank: int
    chunk: int
    logical_stage: int


@dataclass(frozen=True)
class VirtualPipelineLayout:
    """Pure mapping between logical stages and local model chunks.

    Virtual pipeline stages share a process and therefore are deliberately
    not another :class:`ParallelContext` axis.  The mapping is injected into
    builders and schedules instead of mutating a module-level virtual rank.
    """

    num_layers: int
    pipeline_size: int
    virtual_stages_per_rank: int = 1

    def __post_init__(self) -> None:
        if self.pipeline_size < 1:
            raise ValueError("pipeline_size must be positive")
        if self.virtual_stages_per_rank < 1:
            raise ValueError("virtual_stages_per_rank must be positive")
        if self.num_layers < self.num_logical_stages:
            raise ValueError(
                "num_layers must be at least pipeline_size * "
                "virtual_stages_per_rank; empty chunks are unsupported"
            )

    @property
    def num_logical_stages(self) -> int:
        return self.pipeline_size * self.virtual_stages_per_rank

    @property
    def partitions(self) -> tuple[LayerPartition, ...]:
        return partition_layers(self.num_layers, self.num_logical_stages)

    def address(self, pp_rank: int, chunk: int) -> PipelineStageAddress:
        if not 0 <= pp_rank < self.pipeline_size:
            raise ValueError("pp_rank is outside the pipeline")
        if not 0 <= chunk < self.virtual_stages_per_rank:
            raise ValueError("chunk is outside the virtual pipeline")
        logical_stage = chunk * self.pipeline_size + pp_rank
        return PipelineStageAddress(pp_rank, chunk, logical_stage)

    def partition(self, address: PipelineStageAddress) -> LayerPartition:
        expected = self.address(address.pp_rank, address.chunk)
        if expected != address:
            raise ValueError(f"inconsistent pipeline stage address: {address}")
        return self.partitions[address.logical_stage]

    def local_partitions(self, pp_rank: int) -> tuple[LayerPartition, ...]:
        return tuple(
            self.partition(self.address(pp_rank, chunk))
            for chunk in range(self.virtual_stages_per_rank)
        )

    def previous(self, address: PipelineStageAddress) -> PipelineStageAddress | None:
        current = self.address(address.pp_rank, address.chunk)
        if current.logical_stage == 0:
            return None
        logical_stage = current.logical_stage - 1
        chunk, pp_rank = divmod(logical_stage, self.pipeline_size)
        return self.address(pp_rank, chunk)

    def next(self, address: PipelineStageAddress) -> PipelineStageAddress | None:
        current = self.address(address.pp_rank, address.chunk)
        if current.logical_stage == self.num_logical_stages - 1:
            return None
        logical_stage = current.logical_stage + 1
        chunk, pp_rank = divmod(logical_stage, self.pipeline_size)
        return self.address(pp_rank, chunk)

    def is_first(self, address: PipelineStageAddress) -> bool:
        return self.address(address.pp_rank, address.chunk).logical_stage == 0

    def is_last(self, address: PipelineStageAddress) -> bool:
        return (
            self.address(address.pp_rank, address.chunk).logical_stage
            == self.num_logical_stages - 1
        )


def partition_layers(num_layers: int, pipeline_size: int) -> tuple[LayerPartition, ...]:
    """Split layers as evenly as possible, assigning remainders to early stages."""

    if pipeline_size < 1:
        raise ValueError("pipeline_size must be positive")
    if num_layers < pipeline_size:
        raise ValueError("num_layers must be at least pipeline_size; empty stages are unsupported")

    base, remainder = divmod(num_layers, pipeline_size)
    partitions: list[LayerPartition] = []
    start = 0
    for stage in range(pipeline_size):
        count = base + (1 if stage < remainder else 0)
        end = start + count
        partitions.append(
            LayerPartition(
                stage=stage,
                start_layer=start,
                end_layer=end,
                owns_embedding=stage == 0,
                owns_final_norm=stage == pipeline_size - 1,
                owns_lm_head=stage == pipeline_size - 1,
            )
        )
        start = end
    return tuple(partitions)


def partition_for_rank(num_layers: int, pipeline_size: int, pipeline_rank: int) -> LayerPartition:
    if not 0 <= pipeline_rank < pipeline_size:
        raise ValueError("pipeline_rank is outside the pipeline")
    return partition_layers(num_layers, pipeline_size)[pipeline_rank]
