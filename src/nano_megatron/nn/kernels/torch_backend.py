"""Always-available PyTorch kernel backend."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nano_megatron.nn.norms import RMSNorm


class TorchKernelBackend:
    """Reference backend built only from public PyTorch operations."""

    name = "torch"

    def linear(self, in_features: int, out_features: int, **kwargs: Any) -> nn.Module:
        return nn.Linear(in_features, out_features, **kwargs)

    def rms_norm(self, hidden_size: int, eps: float, **kwargs: Any) -> nn.Module:
        return RMSNorm(hidden_size, eps, **kwargs)

    def local_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        causal: bool = True,
        sequence_offset: int = 0,
        dropout_p: float = 0.0,
        scale: float | None = None,
        attention_mask: Tensor | None = None,
        **_: Any,
    ) -> Tensor:
        """Run SDPA for tensors in ``[B,H,S,D]`` layout.

        GQA is expressed explicitly by repeating K/V heads. This is less
        memory-efficient than fused GQA, but keeps the CPU reference path
        portable across PyTorch SDPA backends.
        """

        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q, k and v must have [batch, heads, sequence, head_dim] shape")
        if k.shape != v.shape:
            raise ValueError("k and v must have identical shapes")
        if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
            raise ValueError("q and k/v must agree on batch size and head_dim")
        if q.shape[1] != k.shape[1]:
            if q.shape[1] % k.shape[1]:
                raise ValueError(
                    f"query heads ({q.shape[1]}) must be divisible by KV heads ({k.shape[1]})"
                )
            repeat = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        is_causal = causal and attention_mask is None and sequence_offset == 0
        mask = attention_mask
        if causal and not is_causal:
            query_positions = sequence_offset + torch.arange(q.shape[-2], device=q.device)
            key_positions = torch.arange(k.shape[-2], device=q.device)
            causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            mask = causal_mask if mask is None else (mask & causal_mask)

        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )
