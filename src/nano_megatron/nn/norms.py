"""Normalization layers.

The implementation intentionally stays in plain PyTorch so it can serve as a
numerical oracle for optional fused backends.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned per-channel scale."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1.0e-6,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected the last dimension to be {self.hidden_size}, got {x.shape[-1]}"
            )
        input_dtype = x.dtype
        variance = x.float().square().mean(dim=-1, keepdim=True)
        normalized = x.float() * torch.rsqrt(variance + self.eps)
        return normalized.to(input_dtype) * self.weight.to(input_dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"
