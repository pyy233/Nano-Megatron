"""Pipeline-stage result types and narrow protocols."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from torch import Tensor


@dataclass
class LossOutput:
    loss: Tensor
    metrics: Mapping[str, Tensor | float] = field(default_factory=dict)


@dataclass
class StepOutput:
    losses: tuple[Tensor, ...]
    metrics: dict[str, float]

    @property
    def loss(self) -> Tensor | None:
        if not self.losses:
            return None
        # Schedules store each loss after dividing by the number of microbatches so the same
        # tensors can be used for backward. Summing reconstructs the step-average loss.
        return sum(loss.detach() for loss in self.losses)


class PipelineStage(Protocol):
    def __call__(self, hidden_states: Tensor | None, batch: Any) -> Tensor | LossOutput: ...


def as_loss_output(value: Tensor | LossOutput) -> LossOutput:
    if isinstance(value, LossOutput):
        return value
    if value.ndim == 0:
        return LossOutput(value)
    raise TypeError("the last pipeline stage must return LossOutput or a scalar Tensor")
