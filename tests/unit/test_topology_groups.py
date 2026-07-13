from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from nano_megatron.parallel import (
    DEFAULT_GROUP_PLAN,
    GroupKey,
    GroupPlan,
    GroupSpec,
    ParallelAxis,
    ParallelGroup,
    ParallelGroupRegistry,
    ParallelTopology,
)


def test_multi_rank_parallel_group_requires_materialized_process_group() -> None:
    with pytest.raises(ValueError, match="materialized process_group"):
        ParallelGroup(
            key=GroupKey.TP,
            ranks=(0, 1),
            process_group=None,
            rank=0,
            size=2,
            backend="gloo",
        )


def test_cp_ep_and_replica_groups_expand_from_one_topology() -> None:
    topology = ParallelTopology.from_sizes(context=2, expert=2, data=2)

    cp = DEFAULT_GROUP_PLAN.groups(topology, GroupKey.CP)
    ep = DEFAULT_GROUP_PLAN.groups(topology, GroupKey.EP)
    dense = DEFAULT_GROUP_PLAN.groups(topology, GroupKey.DENSE_REPLICA)
    expert = DEFAULT_GROUP_PLAN.groups(topology, GroupKey.EXPERT_REPLICA)
    batch = DEFAULT_GROUP_PLAN.groups(topology, GroupKey.BATCH_REPLICA)

    assert len(cp) == 4 and all(len(group.ranks) == 2 for group in cp)
    assert len(ep) == 4 and all(len(group.ranks) == 2 for group in ep)
    assert len(dense) == 1 and dense[0].ranks == tuple(range(8))
    assert len(expert) == 2 and all(len(group.ranks) == 4 for group in expert)
    assert len(batch) == 4 and all(len(group.ranks) == 2 for group in batch)


def test_embedding_select_uses_first_and_last_pipeline_stages() -> None:
    topology = ParallelTopology.from_sizes(tensor=2, pipeline=4)
    groups = DEFAULT_GROUP_PLAN.groups(topology, GroupKey.EMBEDDING)

    assert {group.ranks for group in groups} == {(0, 6), (1, 7)}
    assert all(2 not in group.ranks and 4 not in group.ranks for group in groups)


def test_size_one_embedding_selection_deduplicates_zero_and_minus_one() -> None:
    topology = ParallelTopology.from_sizes()
    groups = DEFAULT_GROUP_PLAN.groups(topology, GroupKey.EMBEDDING)
    assert len(groups) == 1
    assert groups[0].ranks == (0,)


def test_custom_group_key_and_channel_are_extensible() -> None:
    plan = GroupPlan(
        GroupSpec("hierarchical_cp", frozenset({ParallelAxis.CP}), channel="p2p"),
    )
    topology = ParallelTopology.from_sizes(context=2, data=2)
    assert {group.ranks for group in plan.groups(topology, "hierarchical_cp")} == {
        (0, 1),
        (2, 3),
    }


def test_default_pipeline_transport_groups_reuse_ranks_on_distinct_channels() -> None:
    topology = ParallelTopology.from_sizes(pipeline=5, data=2)
    pp = DEFAULT_GROUP_PLAN.groups(topology, GroupKey.PP)
    transport_1 = DEFAULT_GROUP_PLAN.groups(topology, GroupKey.PP_TRANSPORT_1)
    transport_2 = DEFAULT_GROUP_PLAN.groups(topology, GroupKey.PP_TRANSPORT_2)

    assert [group.ranks for group in transport_1] == [group.ranks for group in pp]
    assert [group.ranks for group in transport_2] == [group.ranks for group in pp]
    assert DEFAULT_GROUP_PLAN.spec(GroupKey.PP).channel == "default"
    assert DEFAULT_GROUP_PLAN.spec(GroupKey.PP_TRANSPORT_1).channel == "pp_transport_1"
    assert DEFAULT_GROUP_PLAN.spec(GroupKey.PP_TRANSPORT_2).channel == "pp_transport_2"


def test_out_of_range_selection_is_rejected_during_expansion() -> None:
    plan = GroupPlan(
        GroupSpec(
            "bad_selection",
            frozenset({ParallelAxis.PP}),
            select={ParallelAxis.PP: (3,)},
        )
    )
    with pytest.raises(ValueError, match="out of range"):
        plan.expand(ParallelTopology.from_sizes(pipeline=2))


@dataclass
class FakeRuntime:
    rank: int
    world_size: int
    backend: str = "gloo"

    def __post_init__(self) -> None:
        self.created: list[tuple[tuple[int, ...], str]] = []
        self.destroyed: list[Any] = []

    def new_group(self, ranks: tuple[int, ...], *, backend: str | None = None) -> object:
        self.created.append((ranks, backend or self.backend))
        return object()

    def destroy_group(self, process_group: object | None) -> None:
        self.destroyed.append(process_group)

    def create_device_mesh(
        self,
        process_group: object | None,
        *,
        ranks: tuple[int, ...],
        name: str,
    ) -> tuple[object | None, tuple[int, ...], str]:
        return process_group, ranks, name


def test_registry_reuses_singleton_resources_across_channels() -> None:
    topology = ParallelTopology.from_sizes()
    runtime = FakeRuntime(rank=0, world_size=1)
    registry = ParallelGroupRegistry(runtime, topology).materialize(DEFAULT_GROUP_PLAN)

    # There is no communication to isolate for singleton groups, so all
    # default collectives and transport channels collapse to one resource.
    assert len(runtime.created) == 1
    assert registry.resource_count == 1
    assert (
        registry.group(GroupKey.TP).process_group
        is registry.group(GroupKey.DENSE_REPLICA).process_group
    )
    assert (
        registry.group(GroupKey.PP_TRANSPORT_1).process_group
        is registry.group(GroupKey.PP).process_group
    )
    assert (
        registry.group(GroupKey.PP_TRANSPORT_2).process_group
        is registry.group(GroupKey.PP_TRANSPORT_1).process_group
    )
    assert registry.mesh(GroupKey.TP)[1] == (0,)

    registry.close()
    registry.close()
    assert len(runtime.destroyed) == 1


def test_registry_keeps_same_ranks_on_distinct_channels_separate() -> None:
    plan = GroupPlan(
        GroupSpec("grad", frozenset({ParallelAxis.DP}), channel="grad"),
        GroupSpec("params", frozenset({ParallelAxis.DP}), channel="params"),
    )
    topology = ParallelTopology.from_sizes(data=2)
    runtime = FakeRuntime(rank=0, world_size=2)
    registry = ParallelGroupRegistry(runtime, topology).materialize(plan)

    assert len(runtime.created) == 2
    assert registry.group("grad").process_group is not registry.group("params").process_group


def test_registry_keeps_multi_rank_pipeline_transports_separate() -> None:
    topology = ParallelTopology.from_sizes(pipeline=2)
    runtime = FakeRuntime(rank=0, world_size=2)
    registry = ParallelGroupRegistry(runtime, topology).materialize(DEFAULT_GROUP_PLAN)

    pp = registry.group(GroupKey.PP).process_group
    transport_1 = registry.group(GroupKey.PP_TRANSPORT_1).process_group
    transport_2 = registry.group(GroupKey.PP_TRANSPORT_2).process_group
    assert len({id(pp), id(transport_1), id(transport_2)}) == 3
