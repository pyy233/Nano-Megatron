from __future__ import annotations

from types import SimpleNamespace

import pytest

from nano_megatron.training import LearningRateScheduler


class _Strategy:
    def __init__(self) -> None:
        self.learning_rate = -1.0

    def set_learning_rate(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate


def _scheduler(*, schedule: str = "cosine") -> tuple[LearningRateScheduler, _Strategy]:
    strategy = _Strategy()
    scheduler = LearningRateScheduler(
        config=SimpleNamespace(
            schedule=schedule,
            warmup_steps=2,
            decay_steps=6,
            min_lr=0.1,
        ),
        optimizer_config=SimpleNamespace(lr=1.0),
        data_parallel=strategy,
        max_steps=6,
    )
    return scheduler, strategy


def test_warmup_cosine_rate_is_keyed_by_completed_updates() -> None:
    scheduler, strategy = _scheduler()

    assert strategy.learning_rate == pytest.approx(0.5)
    assert scheduler.set_completed_steps(1) == pytest.approx(1.0)
    assert scheduler.set_completed_steps(2) == pytest.approx(1.0)
    assert scheduler.set_completed_steps(4) == pytest.approx(0.55)
    assert scheduler.set_completed_steps(6) == pytest.approx(0.1)
    assert scheduler.set_completed_steps(99) == pytest.approx(0.1)


def test_constant_schedule_keeps_warmup_but_skips_decay() -> None:
    scheduler, _ = _scheduler(schedule="constant")

    assert scheduler.rate_at(0) == pytest.approx(0.5)
    assert scheduler.rate_at(1) == pytest.approx(1.0)
    assert scheduler.rate_at(99) == pytest.approx(1.0)


def test_scheduler_state_round_trip_and_step_mismatch_detection() -> None:
    scheduler, _ = _scheduler()
    scheduler.set_completed_steps(3)
    state = scheduler.state_dict()
    restored, strategy = _scheduler()

    restored.load_state_dict(state, expected_completed_steps=3)

    assert restored.completed_steps == 3
    assert strategy.learning_rate == pytest.approx(scheduler.learning_rate)
    with pytest.raises(RuntimeError, match="TrainerState.step"):
        restored.load_state_dict(state, expected_completed_steps=4)
