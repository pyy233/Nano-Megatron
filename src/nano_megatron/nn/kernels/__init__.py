"""Kernel backend interfaces and implementations."""

from __future__ import annotations

from typing import Any

from nano_megatron.nn.kernels.interface import KernelBackend
from nano_megatron.nn.kernels.torch_backend import TorchKernelBackend


def build_kernel_backend(config: Any, parallel: Any) -> KernelBackend:
    """Build the explicitly configured kernel backend.

    ``parallel`` remains part of the factory contract for future backends. The
    plain PyTorch backend deliberately does not need it.
    """

    backend = getattr(config, "backend", config)
    value = getattr(backend, "value", backend)
    if value == "torch":
        return TorchKernelBackend()
    raise ValueError(f"unknown kernel backend: {value!r}")


__all__ = [
    "KernelBackend",
    "TorchKernelBackend",
    "build_kernel_backend",
]
