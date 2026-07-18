"""Shared helpers for pipeline schedules."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
from torch import Tensor


def data_parallel_context(
    strategy: Any,
    *,
    is_last_microbatch: bool,
    unit: Any | None = None,
):
    if strategy is None or not hasattr(strategy, "microbatch_context"):
        return nullcontext()
    context = strategy.microbatch_context
    if unit is not None and "unit" in inspect.signature(context).parameters:
        return context(is_last_microbatch=is_last_microbatch, unit=unit)
    return context(is_last_microbatch=is_last_microbatch)


def activation_context(strategy: Any):
    if strategy is None or not hasattr(strategy, "activation_context"):
        return nullcontext()
    return strategy.activation_context()


def forward_data_parallel_context(
    strategy: Any,
    *,
    synchronize_gradients: bool,
    unit: Any | None = None,
):
    if strategy is None:
        return nullcontext()
    context = getattr(strategy, "forward_microbatch_context", None)
    if callable(context):
        if unit is not None and "unit" in inspect.signature(context).parameters:
            return context(
                synchronize_gradients=synchronize_gradients,
                unit=unit,
            )
        return context(synchronize_gradients=synchronize_gradients)
    return activation_context(strategy)


ComputeContext = Callable[[], AbstractContextManager[Any]]


def no_compute_context() -> AbstractContextManager[None]:
    return nullcontext()


def match_output_gradient(output: Tensor, gradient: Tensor | None) -> Tensor | None:
    if gradient is None:
        return None
    if gradient.shape != output.shape:
        raise ValueError(
            f"pipeline gradient shape {tuple(gradient.shape)} does not match output "
            f"shape {tuple(output.shape)}"
        )
    return gradient.to(device=output.device, dtype=output.dtype)


def _stage_hook(stage: Any, name: str) -> None:
    """Call an optional pipeline lifecycle hook through common wrappers."""

    current = stage
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        hook = getattr(current, name, None)
        if callable(hook):
            hook()
            return
        current = getattr(current, "module", None)


def prepare_pipeline_stage(stage: Any) -> None:
    _stage_hook(stage, "synchronize_tied_embedding_weights")


def finalize_pipeline_stage_gradients(stage: Any) -> None:
    _stage_hook(stage, "synchronize_tied_embedding_gradients")


def backward(strategy: Any, loss_or_output: Tensor, gradient: Tensor | None = None) -> None:
    if gradient is not None:
        torch.autograd.backward(loss_or_output, gradient)
    elif strategy is not None and hasattr(strategy, "backward"):
        strategy.backward(loss_or_output)
    else:
        loss_or_output.backward()


def _loss_token_count(batch: Any) -> int:
    if not isinstance(batch, Mapping):
        return 1
    labels = batch.get("labels")
    if not isinstance(labels, Tensor):
        return 1
    return int((labels != -100).sum().detach().cpu())


def extract_loss(
    value: Tensor,
    divisor: int,
    batch: Any | None = None,
) -> tuple[Tensor, dict[str, float]]:
    if not isinstance(value, Tensor):
        raise TypeError("pipeline stages must return a Tensor")
    if value.ndim != 0:
        raise TypeError("the last pipeline stage must return a scalar loss Tensor")
    loss = float(value.detach().cpu())
    token_count = _loss_token_count(batch)
    return value / divisor, {
        "loss": loss,
        "loss_sum": loss * token_count,
        "token_count": float(token_count),
    }


def accumulate_metrics(
    sums: dict[str, float],
    counts: dict[str, int],
    metrics: dict[str, float],
) -> None:
    for name, value in metrics.items():
        sums[name] = sums.get(name, 0.0) + value
        counts[name] = counts.get(name, 0) + 1


def average_metrics(sums: dict[str, float], counts: dict[str, int]) -> dict[str, float]:
    additive = {"loss_sum", "token_count"}
    return {
        name: value if name in additive else value / counts[name]
        for name, value in sums.items()
    }
