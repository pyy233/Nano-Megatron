from __future__ import annotations

import itertools

import pytest

from nano_megatron.parallel import (
    PARALLEL_AXES,
    ParallelAxis,
    ParallelCoordinate,
    ParallelTopology,
)


def test_default_order_has_leftmost_axis_varying_fastest() -> None:
    topology = ParallelTopology.from_sizes(
        tensor=2,
        context=2,
        expert=2,
        data=2,
        pipeline=2,
    )

    assert topology.rank_to_coordinate(0) == ParallelCoordinate(0, 0, 0, 0, 0)
    assert topology.rank_to_coordinate(1) == ParallelCoordinate(1, 0, 0, 0, 0)
    assert topology.rank_to_coordinate(2) == ParallelCoordinate(0, 1, 0, 0, 0)
    assert topology.rank_to_coordinate(4) == ParallelCoordinate(0, 0, 1, 0, 0)
    assert topology.rank_to_coordinate(8) == ParallelCoordinate(0, 0, 0, 1, 0)
    assert topology.rank_to_coordinate(16) == ParallelCoordinate(0, 0, 0, 0, 1)


def test_rank_coordinate_round_trip_for_every_order() -> None:
    for order in itertools.permutations(PARALLEL_AXES):
        topology = ParallelTopology.from_sizes(
            tensor=2,
            context=3,
            expert=2,
            data=2,
            pipeline=2,
            order=order,
        )
        for rank in range(topology.world_size):
            coordinate = topology.rank_to_coordinate(rank)
            assert topology.coordinate_to_rank(coordinate) == rank
            assert topology.coordinate_to_rank(coordinate.as_dict()) == rank


def test_topology_derived_replica_sizes_keep_ep_independent() -> None:
    topology = ParallelTopology.from_sizes(context=2, expert=3, data=5)

    assert topology.batch_replica_size == 15
    assert topology.dense_replica_size == 30
    assert topology.expert_replica_size == 10


def test_ranks_varying_is_order_independent() -> None:
    topology = ParallelTopology.from_sizes(
        tensor=2,
        pipeline=2,
        context=2,
        expert=2,
        data=2,
        order=("pp", "dp", "ep", "cp", "tp"),
    )
    rank = topology.coordinate_to_rank(ParallelCoordinate(tp=1, cp=0, ep=1, dp=0, pp=1))
    ranks = topology.ranks_varying(rank, {ParallelAxis.CP, ParallelAxis.EP})

    coordinates = [topology.rank_to_coordinate(member) for member in ranks]
    assert len(coordinates) == 4
    assert {(coordinate.cp, coordinate.ep) for coordinate in coordinates} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    assert all(
        coordinate.tp == 1 and coordinate.dp == 0 and coordinate.pp == 1
        for coordinate in coordinates
    )


@pytest.mark.parametrize("rank", [-1, 4, 1.5, True])
def test_invalid_ranks_are_rejected(rank: object) -> None:
    topology = ParallelTopology.from_sizes(tensor=2, data=2)
    with pytest.raises((TypeError, ValueError)):
        topology.rank_to_coordinate(rank)  # type: ignore[arg-type]
