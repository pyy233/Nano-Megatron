"""Tensor-parallel SwiGLU feed-forward network."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as F
from torch import Tensor, nn

from nano_megatron.models.gpt._config import config_value, sequence_parallel_enabled
from nano_megatron.nn.kernels.interface import KernelBackend
from nano_megatron.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from nano_megatron.tensor_parallel.sequence_parallel import (
    gather_from_sequence_parallel_region,
)

if TYPE_CHECKING:
    from nano_megatron.config import GPTConfig
    from nano_megatron.parallel import ParallelContext


class GPTMLP(nn.Module):
    """SwiGLU MLP with column-parallel gate/up and row-parallel down."""

    def __init__(
        self,
        config: GPTConfig,
        layer_index: int,
        *,
        parallel: ParallelContext,
        kernels: KernelBackend,
        sequence_parallel: bool | None = None,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.parallel = parallel
        self.sequence_parallel = sequence_parallel_enabled(parallel, sequence_parallel)
        hidden_size = int(config_value(config, "hidden_size"))
        ffn_hidden_size = int(config_value(config, "ffn_hidden_size"))
        bias = bool(config_value(config, "linear_bias", "bias", default=False))

        # Separate projections make the gate/up partition unambiguous for any
        # TP size. The sequence all-gather is still performed only once.
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            ffn_hidden_size,
            parallel=parallel,
            kernels=kernels,
            bias=bias,
            sequence_parallel=False,
            disable_input_gradient_reduce=self.sequence_parallel,
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            ffn_hidden_size,
            parallel=parallel,
            kernels=kernels,
            bias=bias,
            sequence_parallel=False,
            disable_input_gradient_reduce=self.sequence_parallel,
        )
        self.down_proj = RowParallelLinear(
            ffn_hidden_size,
            hidden_size,
            parallel=parallel,
            kernels=kernels,
            bias=bias,
            input_is_parallel=True,
            sequence_parallel=self.sequence_parallel,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(hidden_states, self.parallel.tp)
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        output, _ = self.down_proj(F.silu(gate) * up)
        return output
