"""Declarative process-group families derived from a parallel topology."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from .axes import PARALLEL_AXES, ParallelAxis
from .topology import ParallelTopology


class GroupKey(StrEnum):
    TP = "tp"
    PP = "pp"
    PP_TRANSPORT_1 = "pp_transport_1"
    PP_TRANSPORT_2 = "pp_transport_2"
    CP = "cp"
    EP = "ep"
    DP_AXIS = "dp_axis"
    TP_CP = "tp_cp"
    TP_EP = "tp_ep"
    DENSE_REPLICA = "dense_replica"
    EXPERT_REPLICA = "expert_replica"
    BATCH_REPLICA = "batch_replica"
    EMBEDDING = "embedding"


GroupKeyLike = GroupKey | str


def normalize_group_key(key: GroupKeyLike) -> GroupKeyLike:
    if isinstance(key, GroupKey):
        return key
    if not isinstance(key, str) or not key.strip():
        raise TypeError(f"group key must be a GroupKey or non-empty string, got {key!r}")
    value = key.strip()
    try:
        return GroupKey(value)
    except ValueError:
        return value


def _normalize_select(
    select: Mapping[ParallelAxis | str, tuple[int, ...] | list[int]],
) -> Mapping[ParallelAxis, tuple[int, ...]]:
    normalized: dict[ParallelAxis, tuple[int, ...]] = {}
    for raw_axis, raw_indices in select.items():
        axis = ParallelAxis.parse(raw_axis)
        if axis in normalized:
            raise ValueError(f"selection for axis {axis.value!r} was specified twice")
        if not raw_indices:
            raise ValueError(f"selection for axis {axis.value!r} cannot be empty")
        indices = tuple(raw_indices)
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
            raise TypeError(f"selection for axis {axis.value!r} must contain integers")
        normalized[axis] = indices
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class GroupSpec:
    """Describe a family of groups.

    ``varying_axes`` are group member dimensions.  Other dimensions identify
    separate groups.  ``select`` can restrict either role; negative values use
    normal Python indexing against the corresponding axis size.
    """

    key: GroupKeyLike
    varying_axes: frozenset[ParallelAxis]
    select: Mapping[ParallelAxis, tuple[int, ...]] = field(default_factory=dict)
    backend: str | None = None
    channel: str = "default"

    def __post_init__(self) -> None:
        key = normalize_group_key(self.key)
        axes = frozenset(ParallelAxis.parse(axis) for axis in self.varying_axes)
        if not axes:
            raise ValueError("a group spec must vary at least one parallel axis")
        unknown_axes = axes.difference(PARALLEL_AXES)
        if unknown_axes:
            raise ValueError(f"unknown varying axes: {unknown_axes}")
        if self.backend is not None and not self.backend.strip():
            raise ValueError("group backend cannot be an empty string")
        if not isinstance(self.channel, str) or not self.channel.strip():
            raise ValueError("group channel must be a non-empty string")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "varying_axes", axes)
        object.__setattr__(self, "select", _normalize_select(self.select))
        object.__setattr__(self, "backend", self.backend.strip() if self.backend else None)
        object.__setattr__(self, "channel", self.channel.strip())


@dataclass(frozen=True, slots=True)
class PlannedGroup:
    """One concrete rank tuple produced from a :class:`GroupSpec`."""

    spec: GroupSpec
    ranks: tuple[int, ...]
    fixed_coordinate: Mapping[ParallelAxis, int]

    @property
    def key(self) -> GroupKeyLike:
        return self.spec.key

    @property
    def backend(self) -> str | None:
        return self.spec.backend

    @property
    def channel(self) -> str:
        return self.spec.channel


@dataclass(frozen=True, slots=True, init=False)
class GroupPlan:
    """An immutable, ordered collection of uniquely keyed group specs."""

    specs: tuple[GroupSpec, ...]

    def __init__(self, *specs: GroupSpec) -> None:
        if len(specs) == 1 and not isinstance(specs[0], GroupSpec):
            # Convenient for callers holding a tuple/list while retaining the
            # documented variadic constructor.
            specs = tuple(specs[0])  # type: ignore[arg-type, assignment]
        normalized = tuple(specs)
        if any(not isinstance(spec, GroupSpec) for spec in normalized):
            raise TypeError("GroupPlan entries must be GroupSpec instances")
        seen: set[GroupKeyLike] = set()
        for spec in normalized:
            if spec.key in seen:
                raise ValueError(f"group key {spec.key!s} is declared more than once")
            seen.add(spec.key)
        object.__setattr__(self, "specs", normalized)

    def spec(self, key: GroupKeyLike) -> GroupSpec:
        normalized = normalize_group_key(key)
        for spec in self.specs:
            if spec.key == normalized:
                return spec
        raise KeyError(f"group key {normalized!s} is not present in this plan")

    def extend(self, *specs: GroupSpec) -> GroupPlan:
        return GroupPlan(*self.specs, *specs)

    def expand(self, topology: ParallelTopology) -> tuple[PlannedGroup, ...]:
        groups: list[PlannedGroup] = []
        for spec in self.specs:
            groups.extend(self._expand_spec(topology, spec))
        return tuple(groups)

    def groups(self, topology: ParallelTopology, key: GroupKeyLike) -> tuple[PlannedGroup, ...]:
        spec = self.spec(key)
        return self._expand_spec(topology, spec)

    @staticmethod
    def _expand_spec(
        topology: ParallelTopology,
        spec: GroupSpec,
    ) -> tuple[PlannedGroup, ...]:
        allowed = {
            axis: _selected_values(topology.size(axis), spec.select.get(axis))
            for axis in PARALLEL_AXES
        }
        selected_sets = {axis: frozenset(values) for axis, values in allowed.items()}

        grouped: dict[tuple[int, ...], list[int]] = {}
        fixed_axes = tuple(axis for axis in topology.order if axis not in spec.varying_axes)
        for rank in range(topology.world_size):
            coordinate = topology.rank_to_coordinate(rank)
            if any(coordinate.get(axis) not in selected_sets[axis] for axis in PARALLEL_AXES):
                continue
            fixed_key = tuple(coordinate.get(axis) for axis in fixed_axes)
            grouped.setdefault(fixed_key, []).append(rank)

        concrete: list[PlannedGroup] = []
        for fixed_key, ranks in grouped.items():
            fixed = MappingProxyType(dict(zip(fixed_axes, fixed_key, strict=True)))
            concrete.append(PlannedGroup(spec=spec, ranks=tuple(ranks), fixed_coordinate=fixed))
        return tuple(concrete)

    def __iter__(self) -> Iterator[GroupSpec]:
        return iter(self.specs)

    def __len__(self) -> int:
        return len(self.specs)


def _selected_values(size: int, selection: tuple[int, ...] | None) -> tuple[int, ...]:
    if selection is None:
        return tuple(range(size))
    values: list[int] = []
    for raw_index in selection:
        index = raw_index + size if raw_index < 0 else raw_index
        if not 0 <= index < size:
            raise ValueError(
                f"group selection index {raw_index} is out of range for axis size {size}"
            )
        if index not in values:
            values.append(index)
    return tuple(values)


DEFAULT_GROUP_PLAN = GroupPlan(
    GroupSpec(GroupKey.TP, frozenset({ParallelAxis.TP})),
    GroupSpec(GroupKey.PP, frozenset({ParallelAxis.PP})),
    # NCCL may coalesce every operation in one batch_isend_irecv call into a
    # single Work.  Separate pipeline transport channels let a receive be
    # waited without also draining the previous activation/gradient send.
    # PP itself is transport color 0; these communicators provide colors 1/2
    # for a proper coloring of both even and odd physical pipeline cycles.
    GroupSpec(
        GroupKey.PP_TRANSPORT_1,
        frozenset({ParallelAxis.PP}),
        channel="pp_transport_1",
    ),
    GroupSpec(
        GroupKey.PP_TRANSPORT_2,
        frozenset({ParallelAxis.PP}),
        channel="pp_transport_2",
    ),
    GroupSpec(GroupKey.CP, frozenset({ParallelAxis.CP})),
    GroupSpec(GroupKey.EP, frozenset({ParallelAxis.EP})),
    GroupSpec(GroupKey.DP_AXIS, frozenset({ParallelAxis.DP})),
    GroupSpec(GroupKey.TP_CP, frozenset({ParallelAxis.TP, ParallelAxis.CP})),
    GroupSpec(GroupKey.TP_EP, frozenset({ParallelAxis.TP, ParallelAxis.EP})),
    GroupSpec(
        GroupKey.DENSE_REPLICA,
        frozenset({ParallelAxis.DP, ParallelAxis.EP, ParallelAxis.CP}),
    ),
    GroupSpec(
        GroupKey.EXPERT_REPLICA,
        frozenset({ParallelAxis.DP, ParallelAxis.CP}),
    ),
    GroupSpec(
        GroupKey.BATCH_REPLICA,
        frozenset({ParallelAxis.TP, ParallelAxis.PP, ParallelAxis.CP}),
    ),
    GroupSpec(
        GroupKey.EMBEDDING,
        frozenset({ParallelAxis.PP}),
        select={ParallelAxis.PP: (0, -1)},
    ),
)
