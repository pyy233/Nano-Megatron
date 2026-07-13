"""Metric accumulation with explicit distributed groups."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor


class MetricStore:
    def __init__(self) -> None:
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    def update(self, metrics: Mapping[str, float | Tensor]) -> None:
        for name, value in metrics.items():
            self._sums[name] += float(value.detach().cpu()) if isinstance(value, Tensor) else value
            self._counts[name] += 1

    def compute(self) -> dict[str, float]:
        return {
            name: self._sums[name] / self._counts[name]
            for name in self._sums
            if self._counts[name]
        }

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


def reduce_scalar(value: float | Tensor, group: Any | None = None) -> float:
    tensor = value.detach().float() if isinstance(value, Tensor) else torch.tensor(float(value))
    if dist.is_available() and dist.is_initialized() and group is not None:
        raw_group = getattr(group, "process_group", group)
        tensor = tensor.to(torch.device("cuda") if dist.get_backend(raw_group) == "nccl" else "cpu")
        dist.all_reduce(tensor, group=raw_group)
        tensor /= dist.get_world_size(raw_group)
    return float(tensor.cpu())
