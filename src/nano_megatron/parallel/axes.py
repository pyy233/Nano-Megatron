"""Names and coordinates for Nano-Megatron's five parallel axes.

This module deliberately has no dependency on :mod:`torch`.  Rank layout is
useful in configuration validation, checkpoint tooling, and unit tests long
before a process group exists.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar


class ParallelAxis(StrEnum):
    """A first-class dimension in the process topology."""

    TP = "tp"
    PP = "pp"
    CP = "cp"
    EP = "ep"
    DP = "dp"

    @classmethod
    def parse(cls, value: ParallelAxis | str) -> ParallelAxis:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as error:
            choices = ", ".join(axis.value for axis in cls)
            raise ValueError(
                f"unknown parallel axis {value!r}; expected one of: {choices}"
            ) from error


# Coordinate field order is semantic and intentionally independent from rank
# order.  Rank order is configurable and lives in ParallelTopology.
PARALLEL_AXES: tuple[ParallelAxis, ...] = (
    ParallelAxis.TP,
    ParallelAxis.CP,
    ParallelAxis.EP,
    ParallelAxis.DP,
    ParallelAxis.PP,
)

DEFAULT_RANK_ORDER: tuple[ParallelAxis, ...] = PARALLEL_AXES


@dataclass(frozen=True, slots=True)
class ParallelCoordinate:
    """Coordinates of one global rank in semantic axis order."""

    tp: int
    cp: int
    ep: int
    dp: int
    pp: int

    _FIELD_BY_AXIS: ClassVar[Mapping[ParallelAxis, str]] = {
        ParallelAxis.TP: "tp",
        ParallelAxis.CP: "cp",
        ParallelAxis.EP: "ep",
        ParallelAxis.DP: "dp",
        ParallelAxis.PP: "pp",
    }

    def get(self, axis: ParallelAxis | str) -> int:
        parsed = ParallelAxis.parse(axis)
        return getattr(self, self._FIELD_BY_AXIS[parsed])

    def as_dict(self) -> dict[ParallelAxis, int]:
        return {axis: self.get(axis) for axis in PARALLEL_AXES}

    def replace(self, **coordinates: int) -> ParallelCoordinate:
        values = {axis.value: self.get(axis) for axis in PARALLEL_AXES}
        unknown = set(coordinates).difference(values)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"unknown coordinate field(s): {names}")
        values.update(coordinates)
        return ParallelCoordinate(**values)

    def __getitem__(self, axis: ParallelAxis | str) -> int:
        return self.get(axis)

    def __iter__(self) -> Iterator[int]:
        for axis in PARALLEL_AXES:
            yield self.get(axis)


def normalize_rank_order(
    order: Iterable[ParallelAxis | str],
) -> tuple[ParallelAxis, ...]:
    """Parse and validate a rank order.

    The first axis changes fastest.  Every topology always contains all five
    axes exactly once, including size-one axes.
    """

    normalized = tuple(ParallelAxis.parse(axis) for axis in order)
    expected = set(PARALLEL_AXES)
    actual = set(normalized)
    if len(normalized) != len(PARALLEL_AXES) or actual != expected:
        missing = sorted(axis.value for axis in expected.difference(actual))
        duplicates = sorted(axis.value for axis in actual if normalized.count(axis) > 1)
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if duplicates:
            details.append(f"duplicates={duplicates}")
        if not details:
            details.append(f"received {len(normalized)} entries")
        raise ValueError(
            "rank order must contain tp, pp, cp, ep, and dp exactly once ("
            + "; ".join(details)
            + ")"
        )
    return normalized
