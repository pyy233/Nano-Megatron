"""Microbatch splitting and global-batch arithmetic."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass, replace
from typing import Any

from torch import Tensor


def global_batch_size(
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    data_parallel_size: int,
    expert_parallel_size: int,
) -> int:
    values = (
        micro_batch_size,
        gradient_accumulation_steps,
        data_parallel_size,
        expert_parallel_size,
    )
    if any(value < 1 for value in values):
        raise ValueError("batch and parallel sizes must be positive")
    return (
        micro_batch_size
        * gradient_accumulation_steps
        * data_parallel_size
        * expert_parallel_size
    )


def _slice_batch(value: Any, start: int, end: int, batch_size: int) -> Any:
    if isinstance(value, Tensor):
        if value.ndim > 0 and value.size(0) == batch_size:
            return value[start:end]
        return value
    if isinstance(value, Mapping):
        return type(value)(
            (key, _slice_batch(item, start, end, batch_size)) for key, item in value.items()
        )
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_slice_batch(item, start, end, batch_size) for item in value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return type(value)(_slice_batch(item, start, end, batch_size) for item in value)
    if is_dataclass(value):
        updates = {
            field_name: _slice_batch(getattr(value, field_name), start, end, batch_size)
            for field_name in value.__dataclass_fields__
        }
        return replace(value, **updates)
    return value


def split_microbatches(batch: Any, micro_batch_size: int) -> tuple[Any, ...]:
    """Split every leading-batch tensor in a nested batch object."""

    if micro_batch_size < 1:
        raise ValueError("micro_batch_size must be positive")

    tensors: list[Tensor] = []

    def collect(value: Any) -> None:
        if isinstance(value, Tensor) and value.ndim > 0:
            tensors.append(value)
        elif isinstance(value, Mapping):
            for item in value.values():
                collect(item)
        elif is_dataclass(value):
            for field_name in value.__dataclass_fields__:
                collect(getattr(value, field_name))

    collect(batch)
    if not tensors:
        raise ValueError("batch contains no tensor with a leading batch dimension")
    batch_size = tensors[0].size(0)
    if any(tensor.size(0) != batch_size for tensor in tensors):
        raise ValueError("all batched tensors must share the same leading dimension")
    if batch_size % micro_batch_size != 0:
        raise ValueError("batch size must be divisible by micro_batch_size")

    return tuple(
        _slice_batch(batch, start, start + micro_batch_size, batch_size)
        for start in range(0, batch_size, micro_batch_size)
    )
