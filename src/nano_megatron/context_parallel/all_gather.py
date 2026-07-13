"""Readable all-gather context-parallel attention reference implementation."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from nano_megatron.parallel.group import require_parallel_group

from .utils import group_rank, group_size, unwrap_process_group


def _autograd_all_gather(tensor: Tensor, group: Any) -> tuple[Tensor, ...]:
    """All-gather with an autograd-aware implementation when distributed is active."""

    if group_size(group) == 1:
        return (tensor,)
    try:
        from torch.distributed.nn.functional import all_gather

        return tuple(all_gather(tensor, group=unwrap_process_group(group)))
    except (ImportError, RuntimeError) as error:
        if torch.is_grad_enabled() and tensor.requires_grad:
            raise RuntimeError(
                "context-parallel training requires an autograd-aware all_gather; "
                "this PyTorch/backend combination only supports the forward-only fallback"
            ) from error
        # This fallback is useful for forward-only diagnostics on older PyTorch builds.
        outputs = [torch.empty_like(tensor) for _ in range(group_size(group))]
        dist.all_gather(outputs, tensor, group=unwrap_process_group(group))
        return tuple(outputs)


class AllGatherContextParallelAttention(nn.Module):
    """Gather K/V across CP ranks and attend with local Q.

    This implementation intentionally favors a compact, inspectable reference over memory
    efficiency. ``RingContextParallelAttention`` must be checked against it numerically.
    """

    def __init__(
        self,
        group: Any,
        *,
        dropout_p: float = 0.0,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        if dropout_p != 0.0:
            raise ValueError("context-parallel attention dropout is not supported in phase one")
        self.group = require_parallel_group(group, name="context-parallel group")
        self.dropout_p = dropout_p
        self.scale = scale

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        causal: bool = True,
        sequence_offset: int | None = None,
    ) -> Tensor:
        if (
            q.ndim != 4
            or k.shape != v.shape
            or q.shape[0] != k.shape[0]
            or q.shape[-2] != k.shape[-2]
            or q.shape[-1] != k.shape[-1]
        ):
            raise ValueError("q, k, v must have compatible [batch, heads, sequence, dim] shapes")
        if q.device != k.device or q.device != v.device:
            raise ValueError("q, k, and v must be on the same device")
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise ValueError("q, k, and v must have the same dtype")

        local_sequence = q.size(-2)
        gathered_k = torch.cat(_autograd_all_gather(k, self.group), dim=-2)
        gathered_v = torch.cat(_autograd_all_gather(v, self.group), dim=-2)
        if q.size(1) != gathered_k.size(1):
            if q.size(1) % gathered_k.size(1):
                raise ValueError("query heads must be divisible by KV heads")
            repeat = q.size(1) // gathered_k.size(1)
            gathered_k = gathered_k.repeat_interleave(repeat, dim=1)
            gathered_v = gathered_v.repeat_interleave(repeat, dim=1)

        attention_mask: Tensor | None = None
        if causal:
            offset = (
                group_rank(self.group) * local_sequence
                if sequence_offset is None
                else sequence_offset
            )
            query_positions = torch.arange(
                offset, offset + local_sequence, device=q.device, dtype=torch.long
            )
            key_positions = torch.arange(gathered_k.size(-2), device=q.device, dtype=torch.long)
            attention_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            attention_mask = attention_mask.view(1, 1, local_sequence, gathered_k.size(-2))

        return F.scaled_dot_product_attention(
            q,
            gathered_k,
            gathered_v,
            attn_mask=attention_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )
