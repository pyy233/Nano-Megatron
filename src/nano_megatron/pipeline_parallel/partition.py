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
