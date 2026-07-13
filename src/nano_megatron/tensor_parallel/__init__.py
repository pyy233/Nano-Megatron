"""Tensor- and sequence-parallel layers and collectives."""

from nano_megatron.tensor_parallel.cross_entropy import VocabParallelCrossEntropy
from nano_megatron.tensor_parallel.embedding import VocabParallelEmbedding
from nano_megatron.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelLinear,
)
from nano_megatron.tensor_parallel.sequence_parallel import (
    copy_to_tensor_parallel_region,
    gather_from_sequence_parallel_region,
    gather_from_tensor_parallel_region,
    reduce_from_tensor_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    register_sequence_parallel_gradient_hooks,
    scatter_to_sequence_parallel_region,
    scatter_to_tensor_parallel_region,
)

__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelCrossEntropy",
    "VocabParallelEmbedding",
    "VocabParallelLinear",
    "copy_to_tensor_parallel_region",
    "gather_from_sequence_parallel_region",
    "gather_from_tensor_parallel_region",
    "reduce_from_tensor_parallel_region",
    "reduce_scatter_to_sequence_parallel_region",
    "register_sequence_parallel_gradient_hooks",
    "scatter_to_sequence_parallel_region",
    "scatter_to_tensor_parallel_region",
]
