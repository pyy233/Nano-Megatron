"""Non-reentrant activation checkpointing and optional saved-tensor offload."""

from __future__ import annotations

import copy
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, TypeVar

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

T = TypeVar("T")


@dataclass(frozen=True)
class _OffloadedTensor:
    tensor: Tensor
    original_device: torch.device


def _offload_context(*, pin_memory: bool, non_blocking: bool):
    if not hasattr(torch.autograd.graph, "saved_tensors_hooks"):
        return nullcontext()

    def pack(tensor: Tensor) -> Tensor | _OffloadedTensor:
        if tensor.device.type == "cpu":
            return tensor
        cpu = tensor.detach().to("cpu", non_blocking=non_blocking)
        if pin_memory and torch.cuda.is_available() and not cpu.is_pinned():
            cpu = cpu.pin_memory()
        return _OffloadedTensor(cpu, tensor.device)

    def unpack(value: Tensor | _OffloadedTensor) -> Tensor:
        if isinstance(value, Tensor):
            return value
        return value.tensor.to(value.original_device, non_blocking=non_blocking)

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


def _stateful_function(function: Callable[..., T], rng: Any | None) -> Callable[..., T]:
    """Preserve an explicitly injected RNG tracker across recomputation.

    A tracker is intentionally duck-typed: it only needs ``state_dict`` and
    ``load_state_dict``. The regular PyTorch CPU/CUDA RNG is preserved by
    ``torch.utils.checkpoint`` itself.
    """

    if rng is None or not hasattr(rng, "state_dict") or not hasattr(rng, "load_state_dict"):
        return function

    initial_state = copy.deepcopy(rng.state_dict())
    state_after_forward: dict[str, Any] | None = None

    def wrapped(*args: Any, **kwargs: Any) -> T:
        nonlocal state_after_forward
        is_first_forward = state_after_forward is None
        rng.load_state_dict(copy.deepcopy(initial_state))
        try:
            return function(*args, **kwargs)
        finally:
            if is_first_forward:
                state_after_forward = copy.deepcopy(rng.state_dict())
            else:
                rng.load_state_dict(copy.deepcopy(state_after_forward))

    return wrapped


def activation_checkpoint(
    function: Callable[..., T],
    *args: Any,
    rng: Any | None = None,
    offload_saved_tensors: bool = False,
    pin_memory: bool = True,
    non_blocking: bool = True,
    **kwargs: Any,
) -> T:
    """Checkpoint ``function`` with PyTorch's non-reentrant implementation."""

    checkpointed = _stateful_function(function, rng)

    def call(*inputs: Any) -> T:
        return checkpointed(*inputs, **kwargs)

    context = (
        _offload_context(pin_memory=pin_memory, non_blocking=non_blocking)
        if offload_saved_tensors
        else nullcontext()
    )
    with context:
        return torch_checkpoint(
            call,
            *args,
            use_reentrant=False,
            preserve_rng_state=True,
        )


class ActivationCheckpoint(nn.Module):
    """Module wrapper around :func:`activation_checkpoint`."""

    def __init__(
        self,
        module: nn.Module,
        *,
        rng: Any | None = None,
        offload_saved_tensors: bool = False,
        pin_memory: bool = True,
        non_blocking: bool = True,
    ) -> None:
        super().__init__()
        self.module = module
        self.rng = rng
        self.offload_saved_tensors = offload_saved_tensors
        self.pin_memory = pin_memory
        self.non_blocking = non_blocking

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return activation_checkpoint(
            self.module,
            *args,
            rng=self.rng,
            offload_saved_tensors=self.offload_saved_tensors,
            pin_memory=self.pin_memory,
            non_blocking=self.non_blocking,
            **kwargs,
        )
