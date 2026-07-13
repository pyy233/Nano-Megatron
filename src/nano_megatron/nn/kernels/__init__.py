"""Kernel backend interfaces and implementations."""

from __future__ import annotations

from typing import Any

from nano_megatron.nn.kernels.interface import KernelBackend
from nano_megatron.nn.kernels.torch_backend import TorchKernelBackend
from nano_megatron.nn.kernels.transformer_engine import TransformerEngineBackend


def build_kernel_backend(config: Any, parallel: Any) -> KernelBackend:
    """Build the explicitly configured kernel backend.

    ``parallel`` is part of the factory contract because accelerated backends
    need its process groups. The plain PyTorch backend deliberately does not.
    """

    backend = getattr(config, "backend", config)
    value = getattr(backend, "value", backend)
    if value == "torch":
        return TorchKernelBackend()
    if value == "transformer_engine":
        try:
            return TransformerEngineBackend(parallel)
        except ImportError as error:
            raise ImportError(
                "kernels.backend='transformer_engine' requires the optional "
                "transformer-engine package"
            ) from error
    raise ValueError(f"unknown kernel backend: {value!r}")


__all__ = [
    "KernelBackend",
    "TorchKernelBackend",
    "TransformerEngineBackend",
    "build_kernel_backend",
]
