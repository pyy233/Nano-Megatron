"""Tensor-parallel linear layers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nano_megatron.nn.kernels.interface import KernelBackend
from nano_megatron.tensor_parallel._utils import tensor_parallel_group
from nano_megatron.tensor_parallel.sequence_parallel import (
    copy_to_tensor_parallel_region,
    gather_from_sequence_parallel_region,
    gather_from_tensor_parallel_region,
    reduce_from_tensor_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    register_sequence_parallel_gradient_hooks,
    scatter_to_tensor_parallel_region,
)

if TYPE_CHECKING:
    from nano_megatron.parallel import ParallelContext, ParallelGroup


class ColumnParallelLinear(nn.Module):
    """Shard a linear layer along its output-feature dimension."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        parallel: ParallelContext | ParallelGroup,
        kernels: KernelBackend,
        bias: bool = True,
        gather_output: bool = False,
        sequence_parallel: bool = False,
        disable_input_gradient_reduce: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.group = tensor_parallel_group(parallel)
        if out_features % self.group.size:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by TP size ({self.group.size})"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.output_size_per_partition = out_features // self.group.size
        self.gather_output = gather_output
        self.sequence_parallel = sequence_parallel
        self.disable_input_gradient_reduce = disable_input_gradient_reduce
        self.linear = kernels.linear(
            in_features,
            self.output_size_per_partition,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    @property
    def weight(self) -> nn.Parameter:
        return self.linear.weight  # type: ignore[no-any-return]

    @property
    def bias(self) -> nn.Parameter | None:
        return self.linear.bias  # type: ignore[no-any-return]

    def forward(self, x: Tensor) -> tuple[Tensor, None]:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input hidden size {self.in_features}, got {x.shape[-1]}")
        if self.sequence_parallel:
            x = gather_from_sequence_parallel_region(x, self.group)
        elif not self.disable_input_gradient_reduce:
            x = copy_to_tensor_parallel_region(x, self.group)
        output = self.linear(x)
        if self.gather_output:
            output = gather_from_tensor_parallel_region(output, self.group)
        return output, None


class RowParallelLinear(nn.Module):
    """Shard a linear layer along its input-feature dimension."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        parallel: ParallelContext | ParallelGroup,
        kernels: KernelBackend,
        bias: bool = True,
        input_is_parallel: bool = True,
        sequence_parallel: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.group = tensor_parallel_group(parallel)
        if in_features % self.group.size:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by TP size ({self.group.size})"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.input_size_per_partition = in_features // self.group.size
        self.input_is_parallel = input_is_parallel
        self.sequence_parallel = sequence_parallel
        self.linear = kernels.linear(
            self.input_size_per_partition,
            out_features,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        )
        if self.bias is not None:
            # This bias is replicated, rather than sharded, over TP ranks.
            # A constant initialization keeps replicas identical even when
            # each TP shard owns a distinct parameter-initialization stream.
            nn.init.zeros_(self.bias)
            if self.sequence_parallel:
                register_sequence_parallel_gradient_hooks(self, self.group, recurse=False)

    @property
    def weight(self) -> nn.Parameter:
        return self.linear.weight  # type: ignore[no-any-return]

    def forward(self, x: Tensor) -> tuple[Tensor, None]:
        if self.input_is_parallel:
            if x.shape[-1] != self.input_size_per_partition:
                raise ValueError(
                    "parallel input must have local hidden size "
                    f"{self.input_size_per_partition}, got {x.shape[-1]}"
                )
        else:
            if x.shape[-1] != self.in_features:
                raise ValueError(
                    f"expected input hidden size {self.in_features}, got {x.shape[-1]}"
                )
            x = scatter_to_tensor_parallel_region(x, self.group)

        output = self.linear(x)
        if self.sequence_parallel:
            output = reduce_scatter_to_sequence_parallel_region(output, self.group)
        else:
            output = reduce_from_tensor_parallel_region(output, self.group)
        if self.bias is not None:
            output = output + self.bias
        return output, None


class VocabParallelLinear(nn.Module):
    """LM head sharded along the vocabulary dimension."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        parallel: ParallelContext | ParallelGroup,
        bias: bool = False,
        sequence_parallel: bool = False,
        weight: nn.Parameter | None = None,
        init_std: float = 0.02,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.group = tensor_parallel_group(parallel)
        if vocab_size % self.group.size:
            raise ValueError(
                f"vocab_size ({vocab_size}) must be divisible by TP size ({self.group.size})"
            )
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.vocab_size_per_partition = vocab_size // self.group.size
        self.sequence_parallel = sequence_parallel
        if weight is None:
            self.weight = nn.Parameter(
                torch.empty(
                    self.vocab_size_per_partition,
                    hidden_size,
                    device=device,
                    dtype=dtype,
                )
            )
            nn.init.normal_(self.weight, mean=0.0, std=init_std)
        else:
            expected = (self.vocab_size_per_partition, hidden_size)
            if tuple(weight.shape) != expected:
                raise ValueError(
                    f"tied weight must have shape {expected}, got {tuple(weight.shape)}"
                )
            self.weight = weight
        self.bias = (
            nn.Parameter(torch.zeros(self.vocab_size_per_partition, device=device, dtype=dtype))
            if bias
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.hidden_size:
            raise ValueError(f"expected input hidden size {self.hidden_size}, got {x.shape[-1]}")
        if self.sequence_parallel:
            x = gather_from_sequence_parallel_region(x, self.group)
        else:
            x = copy_to_tensor_parallel_region(x, self.group)
        return F.linear(x, self.weight, self.bias)
