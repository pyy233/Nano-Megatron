"""Tensor-parallel causal self-attention."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from torch import Tensor, nn

from nano_megatron.models.gpt._config import config_value, sequence_parallel_enabled
from nano_megatron.nn.dropout import activation_rng_context
from nano_megatron.nn.kernels.interface import KernelBackend
from nano_megatron.nn.rotary import RotaryEmbedding, apply_rotary_pos_emb
from nano_megatron.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from nano_megatron.tensor_parallel.sequence_parallel import (
    gather_from_sequence_parallel_region,
)

if TYPE_CHECKING:
    from nano_megatron.config import GPTConfig
    from nano_megatron.parallel import ParallelContext


class ContextParallelAttention(Protocol):
    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        causal: bool = True,
        sequence_offset: int = 0,
    ) -> Tensor: ...


class GPTAttention(nn.Module):
    """MHA/GQA with column-parallel Q/K/V and row-parallel output."""

    def __init__(
        self,
        config: GPTConfig,
        layer_index: int,
        *,
        parallel: ParallelContext,
        kernels: KernelBackend,
        rng=None,
        cp_attention: ContextParallelAttention | nn.Module | None = None,
        sequence_parallel: bool | None = None,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.parallel = parallel
        self.kernels = kernels
        self.rng = rng
        self.cp_attention = cp_attention
        self.sequence_parallel = sequence_parallel_enabled(parallel, sequence_parallel)

        self.hidden_size = int(config_value(config, "hidden_size"))
        self.num_heads = int(config_value(config, "heads", "num_attention_heads"))
        self.num_kv_heads = int(
            config_value(config, "num_kv_heads", "kv_heads", default=self.num_heads)
        )
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by the number of attention heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError(
                "query heads must be divisible by KV heads for grouped-query attention"
            )
        if self.num_heads % parallel.tp.size or self.num_kv_heads % parallel.tp.size:
            raise ValueError("query and KV head counts must both be divisible by TP size")

        self.head_dim = self.hidden_size // self.num_heads
        self.local_num_heads = self.num_heads // parallel.tp.size
        self.local_num_kv_heads = self.num_kv_heads // parallel.tp.size
        self.kv_hidden_size = self.num_kv_heads * self.head_dim
        self.local_hidden_size = self.hidden_size // parallel.tp.size
        self.local_kv_hidden_size = self.kv_hidden_size // parallel.tp.size
        self.dropout = float(config_value(config, "attention_dropout", "dropout", default=0.0))
        bias = bool(config_value(config, "linear_bias", "bias", default=False))

        # Gathering SP once here avoids doing the same all-gather independently
        # for Q, K and V.
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            parallel=parallel,
            kernels=kernels,
            bias=bias,
            sequence_parallel=False,
            disable_input_gradient_reduce=self.sequence_parallel,
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.kv_hidden_size,
            parallel=parallel,
            kernels=kernels,
            bias=bias,
            sequence_parallel=False,
            disable_input_gradient_reduce=self.sequence_parallel,
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.kv_hidden_size,
            parallel=parallel,
            kernels=kernels,
            bias=bias,
            sequence_parallel=False,
            disable_input_gradient_reduce=self.sequence_parallel,
        )
        self.out_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            parallel=parallel,
            kernels=kernels,
            bias=bias,
            input_is_parallel=True,
            sequence_parallel=self.sequence_parallel,
        )
        self.rope = RotaryEmbedding(
            self.head_dim,
            theta=float(config_value(config, "rope_theta", default=10_000.0)),
        )

    def _prepare_position_ids(
        self,
        position_ids: Tensor | None,
        *,
        original_sequence_length: int,
        gathered_sequence_length: int,
        device,
        sequence_offset: int,
    ) -> Tensor:
        if position_ids is None:
            import torch

            return torch.arange(
                sequence_offset,
                sequence_offset + gathered_sequence_length,
                device=device,
            )
        if position_ids.shape[-1] == gathered_sequence_length:
            return position_ids
        if self.sequence_parallel and position_ids.shape[-1] == original_sequence_length:
            sequence_dim = 0 if position_ids.ndim == 1 else 1
            return gather_from_sequence_parallel_region(
                position_ids,
                self.parallel.tp,
                sequence_dim=sequence_dim,
            )
        raise ValueError(
            "position_ids sequence length must match either the SP-local or gathered sequence"
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        position_ids: Tensor | None = None,
        sequence_offset: int = 0,
        microbatch_index: int = 0,
    ) -> Tensor:
        original_sequence_length = hidden_states.shape[1]
        if self.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(hidden_states, self.parallel.tp)
        position_ids = self._prepare_position_ids(
            position_ids,
            original_sequence_length=original_sequence_length,
            gathered_sequence_length=hidden_states.shape[1],
            device=hidden_states.device,
            sequence_offset=sequence_offset,
        )

        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        batch, sequence, _ = q.shape
        q = q.view(batch, sequence, self.local_num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, sequence, self.local_num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, sequence, self.local_num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(position_ids, device=q.device, dtype=q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if self.cp_attention is not None:
            context = self.cp_attention(
                q,
                k,
                v,
                causal=True,
                sequence_offset=sequence_offset,
            )
        else:
            with activation_rng_context(
                self.rng if self.training and self.dropout > 0.0 else None,
                q,
                layer=self.layer_index,
                microbatch=microbatch_index,
                global_token_offset=sequence_offset,
                op=f"attention.tp={self.parallel.tp.rank}",
            ):
                context = self.kernels.local_attention(
                    q,
                    k,
                    v,
                    causal=True,
                    sequence_offset=sequence_offset,
                    dropout_p=self.dropout if self.training else 0.0,
                )
        context = context.transpose(1, 2).contiguous().view(batch, sequence, self.local_hidden_size)
        output, _ = self.out_proj(context)
        return output
