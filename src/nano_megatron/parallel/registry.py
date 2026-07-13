"""Deterministic materialization and reuse of declared process groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .group import ParallelGroup
from .group_plan import GroupKeyLike, GroupPlan, PlannedGroup, normalize_group_key
from .topology import ParallelTopology


class GroupRuntime(Protocol):
    rank: int
    world_size: int
    backend: str

    def new_group(self, ranks: tuple[int, ...], *, backend: str | None = None) -> Any | None: ...

    def destroy_group(self, process_group: Any | None) -> None: ...

    def create_device_mesh(
        self,
        process_group: Any | None,
        *,
        ranks: tuple[int, ...],
        name: str,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class _ResourceKey:
    ranks: tuple[int, ...]
    backend: str
    channel: str


class ParallelGroupRegistry:
    """Materialize a :class:`GroupPlan` in one stable global order.

    Identical ``(ranks, backend, channel)`` declarations share one underlying
    communicator.  A different channel intentionally creates a separate
    communicator even for identical ranks, which is useful for overlap.
    """

    def __init__(self, runtime: GroupRuntime, topology: ParallelTopology) -> None:
        if runtime.world_size != topology.world_size:
            raise ValueError(
                f"runtime world size {runtime.world_size} != topology world size "
                f"{topology.world_size}"
            )
        if not 0 <= runtime.rank < runtime.world_size:
            raise ValueError(f"runtime rank {runtime.rank} is outside its world size")
        self._runtime = runtime
        self._topology = topology
        self._plan: GroupPlan | None = None
        self._local_groups: dict[GroupKeyLike, ParallelGroup] = {}
        self._families: dict[GroupKeyLike, tuple[tuple[int, ...], ...]] = {}
        self._resources: dict[_ResourceKey, Any | None] = {}
        self._meshes: dict[GroupKeyLike, Any] = {}
        self._closed = False

    @property
    def plan(self) -> GroupPlan:
        if self._plan is None:
            raise RuntimeError("group registry has not been materialized")
        return self._plan

    @property
    def topology(self) -> ParallelTopology:
        return self._topology

    @property
    def resource_count(self) -> int:
        return len(self._resources)

    def materialize(self, plan: GroupPlan) -> ParallelGroupRegistry:
        if self._closed:
            raise RuntimeError("cannot materialize a closed group registry")
        if self._plan is not None:
            if self._plan == plan:
                return self
            raise RuntimeError("a group registry can materialize only one immutable plan")

        expanded = plan.expand(self._topology)
        by_key: dict[GroupKeyLike, list[PlannedGroup]] = {spec.key: [] for spec in plan}
        for planned in expanded:
            by_key[planned.key].append(planned)

        # Every rank executes exactly this loop.  Calling new_group in the same
        # deterministic order is a c10d correctness requirement.
        for planned in expanded:
            backend = planned.backend or self._runtime.backend
            # A singleton communicator has no concurrent communication to
            # isolate.  Reuse it across declared channels so PP=1 does not
            # materialize two otherwise idle transport resources per family.
            resource_channel = "default" if len(planned.ranks) == 1 else planned.channel
            resource_key = _ResourceKey(planned.ranks, backend, resource_channel)
            if resource_key not in self._resources:
                self._resources[resource_key] = self._runtime.new_group(
                    planned.ranks,
                    backend=backend,
                )

            if self._runtime.rank not in planned.ranks:
                continue
            if planned.key in self._local_groups:
                raise RuntimeError(
                    f"rank {self._runtime.rank} belongs to more than one group "
                    f"for key {planned.key!s}"
                )
            local_rank = planned.ranks.index(self._runtime.rank)
            self._local_groups[planned.key] = ParallelGroup(
                key=planned.key,
                ranks=planned.ranks,
                process_group=self._resources[resource_key],
                rank=local_rank,
                size=len(planned.ranks),
                backend=backend,
                channel=planned.channel,
            )

        self._families = {
            key: tuple(planned.ranks for planned in concrete) for key, concrete in by_key.items()
        }
        self._plan = plan
        return self

    def has_group(self, key: GroupKeyLike) -> bool:
        self._require_open()
        return normalize_group_key(key) in self._local_groups

    def group(self, key: GroupKeyLike) -> ParallelGroup:
        self._require_open()
        normalized = normalize_group_key(key)
        try:
            return self._local_groups[normalized]
        except KeyError as error:
            if normalized not in self._families:
                raise KeyError(f"group key {normalized!s} is not declared in the plan") from error
            raise KeyError(
                f"global rank {self._runtime.rank} is not a member of group {normalized!s}"
            ) from error

    def family(self, key: GroupKeyLike) -> tuple[tuple[int, ...], ...]:
        self._require_open()
        normalized = normalize_group_key(key)
        try:
            return self._families[normalized]
        except KeyError as error:
            raise KeyError(f"group key {normalized!s} is not declared in the plan") from error

    def group_ranks_for_rank(self, key: GroupKeyLike, rank: int) -> tuple[int, ...] | None:
        if not 0 <= rank < self._topology.world_size:
            raise ValueError(f"rank must be in [0, {self._topology.world_size}), got {rank}")
        for ranks in self.family(key):
            if rank in ranks:
                return ranks
        return None

    def mesh(self, key: GroupKeyLike) -> Any:
        self._require_open()
        normalized = normalize_group_key(key)
        if normalized not in self._meshes:
            group = self.group(normalized)
            self._meshes[normalized] = self._runtime.create_device_mesh(
                group.process_group,
                ranks=group.ranks,
                name=str(normalized),
            )
        return self._meshes[normalized]

    def close(self) -> None:
        if self._closed:
            return
        # Meshes borrow the process groups.  Dropping references before group
        # destruction makes ownership explicit and avoids relying on GC order.
        self._meshes.clear()
        for process_group in reversed(tuple(self._resources.values())):
            self._runtime.destroy_group(process_group)
        self._resources.clear()
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("parallel group registry is closed")


# A shorter compatibility name is useful in type annotations while the
# concrete class name keeps its purpose clear.
GroupRegistry = ParallelGroupRegistry
