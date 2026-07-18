"""Token-weighted metrics with explicit parallel ownership."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor


class MetricStore:
    """Accumulate optimizer-step or validation-batch metrics without bias."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    def update(self, metrics: Mapping[str, float | Tensor]) -> None:
        for name, value in metrics.items():
            scalar = float(value.detach().cpu()) if isinstance(value, Tensor) else float(value)
            self._sums[name] += scalar
            self._counts[name] += 1

    def compute(self) -> dict[str, float]:
        result = {
            name: self._sums[name] / self._counts[name]
            for name in self._sums
            if self._counts[name] and name not in {"loss", "loss_sum", "token_count"}
        }
        token_count = self._sums.get("token_count", 0.0)
        if token_count > 0.0:
            loss = self._sums["loss_sum"] / token_count
            result["loss"] = loss
            result["perplexity"] = math.exp(loss) if loss < 709.0 else math.inf
            result["token_count"] = token_count
        elif self._counts.get("loss", 0):
            result["loss"] = self._sums["loss"] / self._counts["loss"]
        return result

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


def _collective_device(group: Any, fallback: torch.device) -> torch.device:
    backend = str(getattr(group, "backend", "")).lower()
    if not backend and dist.is_available() and dist.is_initialized():
        raw_group = getattr(group, "process_group", group)
        backend = str(dist.get_backend(raw_group)).lower()
    return fallback if backend == "nccl" else torch.device("cpu")


def _require_collective(group: Any, operation: str) -> Any:
    if int(group.size) == 1:
        return None
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError(f"{operation} requires initialized torch.distributed")
    process_group = getattr(group, "process_group", None)
    if process_group is None:
        raise ValueError(f"{operation} requires a materialized process group")
    return process_group


def aggregate_loss_metrics(
    metrics: Mapping[str, float],
    parallel: Any,
    *,
    device: torch.device,
) -> dict[str, float]:
    """Return one semantic global loss on every pipeline stage.

    Loss is present only on the last PP stage. There, `(loss_sum,
    token_count)` is summed over DENSE_REPLICA = DP×EP×CP. TP is deliberately
    excluded because vocabulary-parallel CE already produces identical loss on
    each TP rank. The totals are then broadcast from the last PP stage so rank
    zero and every stage observe the same value.
    """

    result = {name: float(value) for name, value in metrics.items()}
    is_last = bool(parallel.is_pipeline_last_stage())
    dense_group = parallel.dense_replica
    pp_group = parallel.pp
    collective_device = _collective_device(pp_group, device)
    totals = torch.zeros(2, dtype=torch.float64, device=collective_device)

    if is_last:
        if "loss_sum" in result and "token_count" in result:
            totals[0] = result["loss_sum"]
            totals[1] = result["token_count"]
        elif "loss" in result:
            totals[0] = result["loss"]
            totals[1] = 1.0
        else:
            raise RuntimeError("the last pipeline stage did not produce loss metrics")
        dense_process_group = _require_collective(dense_group, "loss aggregation")
        if dense_process_group is not None:
            dense_device = _collective_device(dense_group, device)
            if totals.device != dense_device:
                totals = totals.to(dense_device)
            dist.all_reduce(totals, group=dense_process_group)
            if totals.device != collective_device:
                totals = totals.to(collective_device)

    pp_process_group = _require_collective(pp_group, "pipeline metric routing")
    if pp_process_group is not None:
        dist.broadcast(
            totals,
            src=int(pp_group.ranks[-1]),
            group=pp_process_group,
        )
    token_count = float(totals[1].cpu())
    if token_count <= 0.0:
        raise RuntimeError("global loss token count must be positive")
    loss_sum = float(totals[0].cpu())
    result["loss_sum"] = loss_sum
    result["token_count"] = token_count
    result["loss"] = loss_sum / token_count
    return result


def reduce_scalar(value: float | Tensor, group: Any | None = None) -> float:
    """Backward-compatible unweighted scalar mean helper."""

    tensor = value.detach().float() if isinstance(value, Tensor) else torch.tensor(float(value))
    if dist.is_available() and dist.is_initialized() and group is not None:
        raw_group = getattr(group, "process_group", group)
        tensor = tensor.to(torch.device("cuda") if dist.get_backend(raw_group) == "nccl" else "cpu")
        dist.all_reduce(tensor, group=raw_group)
        tensor /= dist.get_world_size(raw_group)
    return float(tensor.cpu())


__all__ = ["MetricStore", "aggregate_loss_metrics", "reduce_scalar"]
