from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class OffloadPolicy:
    optimizer_state: bool = False
    zero3_params_and_grads: bool = False
    activations: bool = False
    pin_memory: bool = True
    non_blocking: bool = True

    @classmethod
    def from_config(cls, config: object | None) -> OffloadPolicy:
        if config is None:
            return cls()
        return cls(
            optimizer_state=bool(getattr(config, "optimizer_state", False)),
            zero3_params_and_grads=bool(
                getattr(config, "zero3_params_and_grads", False)
            ),
            activations=bool(getattr(config, "activations", False)),
            pin_memory=bool(getattr(config, "pin_memory", True)),
            non_blocking=bool(getattr(config, "non_blocking", True)),
        )

    def validate(self, mode: str) -> None:
        mode = mode.lower()
        if self.optimizer_state and mode not in {"zero1", "zero2"}:
            raise ValueError("optimizer_state offload is supported only by ZeRO-1/2")
        if self.zero3_params_and_grads and mode != "zero3":
            raise ValueError("zero3_params_and_grads requires data_parallel.mode='zero3'")


@dataclass
class _OffloadedTensor:
    tensor: Tensor
    original_device: torch.device


class ActivationOffloader:
    """Saved-tensor CPU offload with a deliberately small, readable policy surface."""

    def __init__(self, policy: OffloadPolicy) -> None:
        self.policy = policy

    def context(self) -> AbstractContextManager[Any]:
        if not self.policy.activations:
            return nullcontext()
        hooks = getattr(torch.autograd.graph, "saved_tensors_hooks", None)
        if hooks is None:
            raise RuntimeError(
                "activation offload requires torch.autograd.graph.saved_tensors_hooks"
            )
        return hooks(self._pack, self._unpack)

    def _pack(self, tensor: Tensor) -> Tensor | _OffloadedTensor:
        if tensor.device.type == "cpu":
            return tensor
        cpu = torch.empty_like(
            tensor,
            device="cpu",
            pin_memory=self.policy.pin_memory and torch.cuda.is_available(),
        )
        cpu.copy_(tensor, non_blocking=self.policy.non_blocking)
        return _OffloadedTensor(cpu, tensor.device)

    def _unpack(self, value: Tensor | _OffloadedTensor) -> Tensor:
        if isinstance(value, Tensor):
            return value
        return value.tensor.to(
            value.original_device,
            non_blocking=self.policy.non_blocking,
        )


class OptimizerStateStorage:
    """Places ZeRO optimizer shards on their configured long-lived device."""

    def __init__(self, policy: OffloadPolicy, compute_device: torch.device) -> None:
        self.policy = policy
        self.compute_device = compute_device
        self.device = torch.device("cpu") if policy.optimizer_state else compute_device

    def allocate(self, shape: torch.Size | tuple[int, ...]) -> Tensor:
        pin_memory = (
            self.device.type == "cpu"
            and self.compute_device.type == "cuda"
            and self.policy.pin_memory
            and torch.cuda.is_available()
        )
        return torch.zeros(shape, dtype=torch.float32, device=self.device, pin_memory=pin_memory)

    def move_gradient(self, gradient: Tensor) -> Tensor:
        return gradient.to(
            self.device,
            dtype=torch.float32,
            non_blocking=self.policy.non_blocking,
        )

    def move_parameter_for_collective(self, parameter: Tensor, dtype: torch.dtype) -> Tensor:
        return parameter.to(
            self.compute_device,
            dtype=dtype,
            non_blocking=self.policy.non_blocking,
        )
