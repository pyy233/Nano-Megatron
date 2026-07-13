"""Interfaces shared by context-parallel attention implementations."""

from __future__ import annotations

from typing import Protocol

from torch import Tensor


class ContextParallelAttention(Protocol):
    """Compute attention for queries local to one context-parallel rank."""

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        causal: bool = True,
        sequence_offset: int | None = None,
    ) -> Tensor:
        """Return local-query attention output for ``[B, H, S, D]`` tensors."""
