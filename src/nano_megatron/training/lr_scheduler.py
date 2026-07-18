"""Small learning-rate schedules keyed by completed optimizer updates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def _schedule_name(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


class LearningRateScheduler:
    """Apply warmup and optional cosine decay through the DP strategy contract.

    ``completed_steps`` is the number of optimizer updates already committed.
    The LR installed for that state is the rate used by the *next* update. This
    makes checkpoint restoration independent from PyTorch scheduler call-order
    conventions and also works for the hand-written ZeRO-1/2 optimizer.
    """

    _STATE_VERSION = 1

    def __init__(
        self,
        *,
        config: Any,
        optimizer_config: Any,
        data_parallel: Any,
        max_steps: int,
    ) -> None:
        self.schedule = _schedule_name(config.schedule)
        if self.schedule not in {"constant", "cosine"}:
            raise ValueError(f"unsupported learning-rate schedule: {self.schedule!r}")
        self.base_lr = float(optimizer_config.lr)
        self.min_lr = float(config.min_lr)
        self.warmup_steps = int(config.warmup_steps)
        self.decay_steps = int(config.decay_steps or max_steps)
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.decay_steps < 1:
            raise ValueError("decay_steps must be positive")
        if self.warmup_steps > self.decay_steps:
            raise ValueError("warmup_steps cannot exceed decay_steps")
        if not 0.0 <= self.min_lr <= self.base_lr:
            raise ValueError("min_lr must be between zero and the optimizer base LR")
        self.data_parallel = data_parallel
        self.completed_steps = 0
        self._apply()

    @property
    def learning_rate(self) -> float:
        return self.rate_at(self.completed_steps)

    def rate_at(self, completed_steps: int) -> float:
        if isinstance(completed_steps, bool) or not isinstance(completed_steps, int):
            raise TypeError("completed_steps must be an integer")
        if completed_steps < 0:
            raise ValueError("completed_steps must be non-negative")
        if self.warmup_steps and completed_steps < self.warmup_steps:
            return self.base_lr * float(completed_steps + 1) / float(self.warmup_steps)
        if self.schedule == "constant":
            return self.base_lr
        decay_span = self.decay_steps - self.warmup_steps
        if decay_span <= 0:
            return self.min_lr if completed_steps >= self.decay_steps else self.base_lr
        progress = (completed_steps - self.warmup_steps) / decay_span
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def set_completed_steps(self, completed_steps: int) -> float:
        self.completed_steps = int(completed_steps)
        if self.completed_steps < 0:
            raise ValueError("completed_steps must be non-negative")
        return self._apply()

    def step(self) -> float:
        return self.set_completed_steps(self.completed_steps + 1)

    def state_dict(self) -> dict[str, int | float | str]:
        return {
            "version": self._STATE_VERSION,
            "schedule": self.schedule,
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "completed_steps": self.completed_steps,
            "learning_rate": self.learning_rate,
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        expected_completed_steps: int | None = None,
    ) -> None:
        version = int(state.get("version", self._STATE_VERSION))
        if version != self._STATE_VERSION:
            raise ValueError(f"unsupported LR scheduler state version: {version}")
        expected = {
            "schedule": self.schedule,
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
        }
        for name, current in expected.items():
            saved = state.get(name, current)
            if saved != current:
                raise RuntimeError(
                    f"LR scheduler configuration changed across resume for {name}: "
                    f"{saved!r} != {current!r}"
                )
        completed_steps = int(state.get("completed_steps", 0))
        if expected_completed_steps is not None and completed_steps != expected_completed_steps:
            raise RuntimeError(
                "LR scheduler step does not match TrainerState.step: "
                f"{completed_steps} != {expected_completed_steps}"
            )
        self.set_completed_steps(completed_steps)

    def _apply(self) -> float:
        learning_rate = self.learning_rate
        self.data_parallel.set_learning_rate(learning_rate)
        return learning_rate


__all__ = ["LearningRateScheduler"]
