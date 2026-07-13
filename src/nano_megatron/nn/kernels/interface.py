"""Typed interface for replaceable model kernels."""

from __future__ import annotations

from typing import Any, Protocol

from torch import Tensor, nn


class KernelBackend(Protocol):
    """The deliberately small kernel surface consumed by the GPT model."""

    name: str

    def linear(self, in_features: int, out_features: int, **kwargs: Any) -> nn.Module: ...

    def rms_norm(self, hidden_size: int, eps: float, **kwargs: Any) -> nn.Module: ...

    def local_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        **kwargs: Any,
    ) -> Tensor: ...
