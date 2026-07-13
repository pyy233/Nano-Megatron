"""Vocabulary-parallel token embedding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nano_megatron.tensor_parallel._utils import tensor_parallel_group
from nano_megatron.tensor_parallel.sequence_parallel import (
    reduce_from_tensor_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)

if TYPE_CHECKING:
    from nano_megatron.nn.kernels.interface import KernelBackend
    from nano_megatron.parallel import ParallelContext, ParallelGroup


class VocabParallelEmbedding(nn.Module):
    """Shard embedding rows across the tensor-parallel group."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        *,
        parallel: ParallelContext | ParallelGroup,
        kernels: KernelBackend | None = None,
        sequence_parallel: bool = False,
        init_std: float = 0.02,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        del kernels  # Embedding has no backend-specific primitive in phase one.
        self.group = tensor_parallel_group(parallel)
        if vocab_size % self.group.size:
            raise ValueError(
                f"vocab_size ({vocab_size}) must be divisible by TP size ({self.group.size})"
            )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_embeddings_per_partition = vocab_size // self.group.size
        self.vocab_start_index = self.group.rank * self.num_embeddings_per_partition
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition
        self.sequence_parallel = sequence_parallel
        self.weight = nn.Parameter(
            torch.empty(
                self.num_embeddings_per_partition,
                hidden_size,
                device=device,
                dtype=dtype,
            )
        )
        nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"input_ids must be an integer tensor, got {input_ids.dtype}")
        outside_partition = (input_ids < self.vocab_start_index) | (
            input_ids >= self.vocab_end_index
        )
        local_ids = (input_ids - self.vocab_start_index).masked_fill(outside_partition, 0)
        output = F.embedding(local_ids, self.weight)
        output = output.masked_fill(outside_partition.unsqueeze(-1), 0.0)
        if self.sequence_parallel:
            if output.ndim < 3:
                raise ValueError(
                    "sequence parallel embedding expects input_ids shaped [batch, sequence]"
                )
            return reduce_scatter_to_sequence_parallel_region(output, self.group)
        return reduce_from_tensor_parallel_region(output, self.group)
