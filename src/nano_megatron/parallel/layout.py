"""Debug-only activation layout metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TensorLayout:
    cp_sharded: bool
    sp_sharded: bool
    tp_hidden_sharded: bool
    sequence_dim: int = 1

    def __post_init__(self) -> None:
        for name in ("cp_sharded", "sp_sharded", "tp_hidden_sharded"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if not isinstance(self.sequence_dim, int):
            raise TypeError("sequence_dim must be an integer")
