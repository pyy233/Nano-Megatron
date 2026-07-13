"""Optional Transformer Engine adapter with an explicit, narrow surface."""

from __future__ import annotations

import importlib
import inspect
from typing import Any

from torch import Tensor, nn

from .torch_backend import TorchKernelBackend


def _accepts_keyword(factory: Any, name: str) -> bool:
    """Best-effort feature detection across Transformer Engine versions."""

    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return True
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


class TransformerEngineBackend:
    """Use TE for local Linear/RMSNorm and keep communication in this project.

    Nano-Megatron's tensor-parallel wrappers already own sharding and
    collectives, so TE modules are deliberately constructed as local modules.
    Attention falls back to the readable PyTorch SDPA implementation in phase
    one; no FP8/quantization context is enabled here.
    """

    name = "transformer_engine"

    def __init__(self, parallel: Any = None, *, te_module: Any | None = None) -> None:
        self.parallel = parallel
        self._te = te_module or importlib.import_module("transformer_engine.pytorch")
        self._torch = TorchKernelBackend()

    def linear(self, in_features: int, out_features: int, **kwargs: Any) -> nn.Module:
        bias = bool(kwargs.pop("bias", True))
        device = kwargs.pop("device", None)
        dtype = kwargs.pop("dtype", None)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported Transformer Engine Linear options: {names}")

        options: dict[str, Any] = {"bias": bias}
        moved_device = device
        moved_dtype = dtype
        if device is not None and _accepts_keyword(self._te.Linear, "device"):
            options["device"] = device
            moved_device = None
        if dtype is not None:
            if _accepts_keyword(self._te.Linear, "params_dtype"):
                options["params_dtype"] = dtype
                moved_dtype = None
            elif _accepts_keyword(self._te.Linear, "dtype"):
                options["dtype"] = dtype
                moved_dtype = None

        module = self._te.Linear(in_features, out_features, **options)
        if moved_device is not None or moved_dtype is not None:
            module = module.to(device=moved_device, dtype=moved_dtype)
        return module

    def rms_norm(self, hidden_size: int, eps: float, **kwargs: Any) -> nn.Module:
        device = kwargs.pop("device", None)
        dtype = kwargs.pop("dtype", None)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported Transformer Engine RMSNorm options: {names}")

        options: dict[str, Any] = {"eps": eps}
        moved_device = device
        moved_dtype = dtype
        if device is not None and _accepts_keyword(self._te.RMSNorm, "device"):
            options["device"] = device
            moved_device = None
        if dtype is not None:
            if _accepts_keyword(self._te.RMSNorm, "params_dtype"):
                options["params_dtype"] = dtype
                moved_dtype = None
            elif _accepts_keyword(self._te.RMSNorm, "dtype"):
                options["dtype"] = dtype
                moved_dtype = None

        if _accepts_keyword(self._te.RMSNorm, "normalized_shape"):
            module = self._te.RMSNorm(normalized_shape=hidden_size, **options)
        else:
            module = self._te.RMSNorm(hidden_size, **options)
        if moved_device is not None or moved_dtype is not None:
            module = module.to(device=moved_device, dtype=moved_dtype)
        return module

    def local_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        return self._torch.local_attention(q, k, v, **kwargs)


__all__ = ["TransformerEngineBackend"]
