"""Pipeline-stage result types and narrow protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from torch import Tensor


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
    def __call__(self, hidden_states: Tensor | None, batch: Any) -> Tensor: ...
