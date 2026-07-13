"""Rotary position embeddings (RoPE)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _reshape_frequencies(frequencies: Tensor, x: Tensor) -> Tensor:
    """Make ``[S,D]`` or ``[B,S,D]`` frequencies broadcast over heads."""

    if frequencies.ndim == 2:
        return frequencies.unsqueeze(0).unsqueeze(0)
    if frequencies.ndim == 3:
        return frequencies.unsqueeze(1)
    raise ValueError(
        "RoPE cos/sin tensors must have shape [sequence, head_dim] or "
        f"[batch, sequence, head_dim], got {tuple(frequencies.shape)}"
    )


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply RoPE to query and key tensors in ``[B,H,S,D]`` layout."""

    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must use [batch, heads, sequence, head_dim] layout")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head_dim")
    if q.shape[-1] % 2:
        raise ValueError("RoPE head_dim must be even")
    cos = _reshape_frequencies(cos, q).to(device=q.device, dtype=q.dtype)
    sin = _reshape_frequencies(sin, q).to(device=q.device, dtype=q.dtype)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class RotaryEmbedding(nn.Module):
    """Generate RoPE cosine and sine values for explicit position ids."""

    def __init__(self, dim: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError(f"RoPE dim must be a positive even integer, got {dim}")
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        position_ids: Tensor,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[Tensor, Tensor]:
        if position_ids.ndim not in (1, 2):
            raise ValueError("position_ids must have shape [sequence] or [batch, sequence]")
        target_device = device if device is not None else position_ids.device
        positions = position_ids.to(device=target_device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(device=target_device)
        frequencies = positions.unsqueeze(-1) * inv_freq
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        output_dtype = dtype if dtype is not None else torch.get_default_dtype()
        return embeddings.cos().to(output_dtype), embeddings.sin().to(output_dtype)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, theta={self.theta}"
