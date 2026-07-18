from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from nano_megatron.training import MetricStore, aggregate_loss_metrics


@dataclass(frozen=True)
class _Group:
    ranks: tuple[int, ...] = (0,)
    rank: int = 0
    size: int = 1
    process_group: object | None = None
    backend: str = "gloo"


class _Parallel:
    rank = 0
    dense_replica = _Group()
    pp = _Group()

    @staticmethod
    def is_pipeline_last_stage() -> bool:
        return True


def test_metric_store_uses_token_weighted_loss_and_perplexity() -> None:
    store = MetricStore()
    store.update({"loss_sum": 4.0, "token_count": 2.0, "learning_rate": 0.3})
    store.update({"loss_sum": 9.0, "token_count": 3.0, "learning_rate": 0.1})

    metrics = store.compute()

    assert metrics["loss"] == pytest.approx(13.0 / 5.0)
    assert metrics["perplexity"] == pytest.approx(torch.exp(torch.tensor(2.6)).item())
    assert metrics["token_count"] == 5.0
    assert metrics["learning_rate"] == pytest.approx(0.2)


def test_single_process_loss_aggregation_preserves_additive_totals() -> None:
    metrics = aggregate_loss_metrics(
        {"loss": 2.0, "loss_sum": 12.0, "token_count": 6.0},
        _Parallel(),
        device=torch.device("cpu"),
    )

    assert metrics["loss"] == 2.0
    assert metrics["loss_sum"] == 12.0
    assert metrics["token_count"] == 6.0
