"""Immutable, explicitly injected view of all parallel state for one rank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nano_megatron.config.schema import ParallelConfig
from nano_megatron.distributed.runtime import DistributedRuntime

from .axes import ParallelAxis, ParallelCoordinate
from .group import ParallelGroup
from .group_plan import DEFAULT_GROUP_PLAN, GroupKey, GroupKeyLike, GroupPlan
from .registry import ParallelGroupRegistry
from .topology import ParallelTopology

_AXIS_GROUP_KEYS = {
    ParallelAxis.TP: GroupKey.TP,
    ParallelAxis.PP: GroupKey.PP,
    ParallelAxis.CP: GroupKey.CP,
    ParallelAxis.EP: GroupKey.EP,
    ParallelAxis.DP: GroupKey.DP_AXIS,
}


@dataclass(frozen=True, slots=True, init=False)
class ParallelContext:
    """All parallel state required by model/training objects.

    It has no global registration API and no ``None`` fallback.  The registry
    is internally mutable only for lazy DeviceMesh creation and idempotent
    cleanup; the topology, config, coordinate, and group plan never change.
    """

    runtime: DistributedRuntime
    config: ParallelConfig
    topology: ParallelTopology
    coordinate: ParallelCoordinate
    _group_plan: GroupPlan
    _registry: ParallelGroupRegistry

    @classmethod
    def create(
        cls,
        runtime: DistributedRuntime,
        config: ParallelConfig,
        group_plan: GroupPlan | None = None,
    ) -> ParallelContext:
        if not runtime.is_initialized:
            raise RuntimeError("ParallelContext.create requires an initialized DistributedRuntime")
        resolved_config = config.resolved(runtime.world_size)
        topology = ParallelTopology.from_config(resolved_config, world_size=runtime.world_size)
        plan = DEFAULT_GROUP_PLAN if group_plan is None else group_plan
        _validate_context_group_plan(plan)
        registry = ParallelGroupRegistry(runtime, topology).materialize(plan)

        instance = object.__new__(cls)
        object.__setattr__(instance, "runtime", runtime)
        object.__setattr__(instance, "config", resolved_config)
        object.__setattr__(instance, "topology", topology)
        object.__setattr__(instance, "coordinate", topology.rank_to_coordinate(runtime.rank))
        object.__setattr__(instance, "_group_plan", plan)
        object.__setattr__(instance, "_registry", registry)
        return instance

    @property
    def rank(self) -> int:
        return self.runtime.rank

    @property
    def world_size(self) -> int:
        return self.topology.world_size

    @property
    def sequence_parallel(self) -> bool:
        return self.config.sequence_parallel

    def group(self, key: GroupKeyLike) -> ParallelGroup:
        return self._registry.group(key)

    def has_group(self, key: GroupKeyLike) -> bool:
        return self._registry.has_group(key)

    def axis_group(self, axis: ParallelAxis | str) -> ParallelGroup:
        parsed = ParallelAxis.parse(axis)
        return self.group(_AXIS_GROUP_KEYS[parsed])

    def mesh(self, key: GroupKeyLike) -> Any:
        return self._registry.mesh(key)

    def group_plan(self) -> GroupPlan:
        return self._group_plan

    def group_family(self, key: GroupKeyLike) -> tuple[tuple[int, ...], ...]:
        return self._registry.family(key)

    @property
    def tp(self) -> ParallelGroup:
        return self.group(GroupKey.TP)

    @property
    def pp(self) -> ParallelGroup:
        return self.group(GroupKey.PP)

    @property
    def cp(self) -> ParallelGroup:
        return self.group(GroupKey.CP)

    @property
    def ep(self) -> ParallelGroup:
        return self.group(GroupKey.EP)

    @property
    def dp(self) -> ParallelGroup:
        return self.group(GroupKey.DP_AXIS)

    @property
    def dense_replica(self) -> ParallelGroup:
        return self.group(GroupKey.DENSE_REPLICA)

    @property
    def expert_replica(self) -> ParallelGroup:
        return self.group(GroupKey.EXPERT_REPLICA)

    @property
    def batch_replica(self) -> ParallelGroup:
        return self.group(GroupKey.BATCH_REPLICA)

    def is_pipeline_first_stage(self) -> bool:
        return self.coordinate.pp == 0

    def is_pipeline_last_stage(self) -> bool:
        return self.coordinate.pp == self.topology.pipeline_parallel_size - 1

    def pipeline_prev_rank(self) -> int | None:
        if self.is_pipeline_first_stage():
            return None
        return self.pp.global_rank_at(self.pp.rank - 1)

    def pipeline_next_rank(self) -> int | None:
        if self.is_pipeline_last_stage():
            return None
        return self.pp.global_rank_at(self.pp.rank + 1)

    def close(self) -> None:
        self._registry.close()

    def __enter__(self) -> ParallelContext:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"ParallelContext(rank={self.rank}, coordinate={self.coordinate!r}, "
            f"topology={self.topology!r})"
        )


def _validate_context_group_plan(plan: GroupPlan) -> None:
    """Keep built-in key semantics stable while allowing additional specs."""

    for expected in DEFAULT_GROUP_PLAN:
        try:
            actual = plan.spec(expected.key)
        except KeyError as error:
            raise ValueError(
                f"ParallelContext group plan is missing required key {expected.key!s}"
            ) from error
        if actual.varying_axes != expected.varying_axes or actual.select != expected.select:
            raise ValueError(
                f"group key {expected.key!s} has fixed Nano-Megatron semantics: "
                f"varying_axes={expected.varying_axes}, select={dict(expected.select)}"
            )
