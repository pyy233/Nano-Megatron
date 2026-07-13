"""Pure mathematical mapping between global ranks and five-dimensional coordinates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from .axes import (
    DEFAULT_RANK_ORDER,
    PARALLEL_AXES,
    ParallelAxis,
    ParallelCoordinate,
    normalize_rank_order,
)

if TYPE_CHECKING:
    from nano_megatron.config.schema import ParallelConfig


def _positive_size(axis: ParallelAxis, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"size for axis {axis.value!r} must be an integer, got {value!r}")
    if value < 1:
        raise ValueError(f"size for axis {axis.value!r} must be >= 1, got {value}")
    return value


@dataclass(frozen=True, slots=True, init=False)
class ParallelTopology:
    """An immutable five-axis rank topology.

    ``order`` follows Megatron's convention: its left-most axis changes
    fastest.  Size-one dimensions remain in the order so a checkpoint has an
    unambiguous topology even when a dimension is currently degenerate.
    """

    _sizes: Mapping[ParallelAxis, int]
    order: tuple[ParallelAxis, ...]
    _strides: Mapping[ParallelAxis, int]
    world_size: int

    def __init__(
        self,
        sizes: Mapping[ParallelAxis | str, int],
        order: Sequence[ParallelAxis | str] = DEFAULT_RANK_ORDER,
    ) -> None:
        parsed_sizes: dict[ParallelAxis, int] = {}
        for raw_axis, value in sizes.items():
            axis = ParallelAxis.parse(raw_axis)
            if axis in parsed_sizes:
                raise ValueError(f"size for axis {axis.value!r} was specified more than once")
            parsed_sizes[axis] = _positive_size(axis, value)

        missing = set(PARALLEL_AXES).difference(parsed_sizes)
        extra_count = len(parsed_sizes) - len(PARALLEL_AXES)
        if missing or extra_count:
            names = ", ".join(sorted(axis.value for axis in missing))
            raise ValueError(
                f"sizes must define all five parallel axes; missing: {names or 'none'}"
            )

        normalized_order = normalize_rank_order(order)
        strides: dict[ParallelAxis, int] = {}
        stride = 1
        for axis in normalized_order:
            strides[axis] = stride
            stride *= parsed_sizes[axis]

        object.__setattr__(self, "_sizes", MappingProxyType(parsed_sizes))
        object.__setattr__(self, "order", normalized_order)
        object.__setattr__(self, "_strides", MappingProxyType(strides))
        object.__setattr__(self, "world_size", stride)

    @classmethod
    def from_sizes(
        cls,
        *,
        tensor: int = 1,
        pipeline: int = 1,
        context: int = 1,
        expert: int = 1,
        data: int = 1,
        order: Sequence[ParallelAxis | str] = DEFAULT_RANK_ORDER,
    ) -> ParallelTopology:
        return cls(
            {
                ParallelAxis.TP: tensor,
                ParallelAxis.PP: pipeline,
                ParallelAxis.CP: context,
                ParallelAxis.EP: expert,
                ParallelAxis.DP: data,
            },
            order,
        )

    @classmethod
    def from_config(
        cls,
        config: ParallelConfig,
        *,
        world_size: int | None = None,
    ) -> ParallelTopology:
        data_size = config.resolve_data_parallel_size(world_size)
        return cls.from_sizes(
            tensor=config.tensor,
            pipeline=config.pipeline,
            context=config.context,
            expert=config.expert,
            data=data_size,
            order=config.order,
        )

    @property
    def sizes(self) -> Mapping[ParallelAxis, int]:
        return self._sizes

    @property
    def tensor_parallel_size(self) -> int:
        return self.size(ParallelAxis.TP)

    @property
    def pipeline_parallel_size(self) -> int:
        return self.size(ParallelAxis.PP)

    @property
    def context_parallel_size(self) -> int:
        return self.size(ParallelAxis.CP)

    @property
    def expert_parallel_size(self) -> int:
        return self.size(ParallelAxis.EP)

    @property
    def data_parallel_size(self) -> int:
        return self.size(ParallelAxis.DP)

    @property
    def batch_replica_size(self) -> int:
        return self.data_parallel_size * self.expert_parallel_size

    @property
    def dense_replica_size(self) -> int:
        return self.data_parallel_size * self.expert_parallel_size * self.context_parallel_size

    @property
    def expert_replica_size(self) -> int:
        return self.data_parallel_size * self.context_parallel_size

    def size(self, axis: ParallelAxis | str) -> int:
        return self._sizes[ParallelAxis.parse(axis)]

    def stride(self, axis: ParallelAxis | str) -> int:
        return self._strides[ParallelAxis.parse(axis)]

    def rank_to_coordinate(self, rank: int) -> ParallelCoordinate:
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError(f"rank must be an integer, got {rank!r}")
        if not 0 <= rank < self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size}), got {rank}")

        values: dict[str, int] = {}
        for axis in PARALLEL_AXES:
            values[axis.value] = (rank // self._strides[axis]) % self._sizes[axis]
        return ParallelCoordinate(**values)

    def coordinate_to_rank(
        self,
        coordinate: ParallelCoordinate | Mapping[ParallelAxis | str, int],
    ) -> int:
        if isinstance(coordinate, ParallelCoordinate):
            values = coordinate.as_dict()
        else:
            values = self._normalize_coordinate_mapping(coordinate)

        rank = 0
        for axis in PARALLEL_AXES:
            value = values[axis]
            size = self._sizes[axis]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"coordinate for axis {axis.value!r} must be an integer, got {value!r}"
                )
            if not 0 <= value < size:
                raise ValueError(
                    f"coordinate for axis {axis.value!r} must be in [0, {size}), got {value}"
                )
            rank += value * self._strides[axis]
        return rank

    def iter_coordinates(self) -> Iterable[ParallelCoordinate]:
        for rank in range(self.world_size):
            yield self.rank_to_coordinate(rank)

    def ranks_varying(
        self,
        rank: int,
        varying_axes: Iterable[ParallelAxis | str],
    ) -> tuple[int, ...]:
        axes = frozenset(ParallelAxis.parse(axis) for axis in varying_axes)
        coordinate = self.rank_to_coordinate(rank)
        fixed = {axis: coordinate.get(axis) for axis in PARALLEL_AXES if axis not in axes}
        return tuple(
            candidate
            for candidate in range(self.world_size)
            if all(
                self.rank_to_coordinate(candidate).get(axis) == value
                for axis, value in fixed.items()
            )
        )

    def _normalize_coordinate_mapping(
        self,
        coordinate: Mapping[ParallelAxis | str, int],
    ) -> dict[ParallelAxis, int]:
        values: dict[ParallelAxis, int] = {}
        for raw_axis, value in coordinate.items():
            axis = ParallelAxis.parse(raw_axis)
            if axis in values:
                raise ValueError(f"coordinate for axis {axis.value!r} was specified twice")
            values[axis] = value
        missing = set(PARALLEL_AXES).difference(values)
        if missing:
            names = ", ".join(sorted(axis.value for axis in missing))
            raise ValueError(f"coordinate is missing axis value(s): {names}")
        return values

    def __len__(self) -> int:
        return self.world_size

    def __repr__(self) -> str:
        sizes = ", ".join(f"{axis.value}={self.size(axis)}" for axis in PARALLEL_AXES)
        order = "-".join(axis.value for axis in self.order)
        return f"ParallelTopology({sizes}, order={order!r})"
