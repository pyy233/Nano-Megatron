from __future__ import annotations

from nano_megatron.parallel import ParallelCoordinate, ParallelRNG, RNGStream


def test_rng_replica_seed_semantics() -> None:
    base = ParallelRNG(123, ParallelCoordinate(tp=1, cp=0, ep=0, dp=0, pp=1))
    dense_replica = ParallelRNG(123, ParallelCoordinate(tp=1, cp=1, ep=1, dp=2, pp=1))
    different_expert = ParallelRNG(123, ParallelCoordinate(tp=1, cp=1, ep=1, dp=2, pp=1))

    assert base.seed(RNGStream.DENSE_INIT) == dense_replica.seed(RNGStream.DENSE_INIT)
    assert base.seed(RNGStream.EXPERT_INIT) != different_expert.seed(RNGStream.EXPERT_INIT)
    assert base.seed(RNGStream.DATA) != dense_replica.seed(RNGStream.DATA)


def test_rng_state_round_trip_restores_stream_counters() -> None:
    rng = ParallelRNG(7, ParallelCoordinate(0, 0, 0, 0, 0))
    state = rng.state_dict()
    expected = rng.next_seed("activation", "layer-0")
    rng.next_seed("activation", "layer-0")

    rng.load_state_dict(state)

    assert rng.next_seed("activation", "layer-0") == expected


def test_activation_seed_is_explicitly_derived_from_work_identity() -> None:
    rng = ParallelRNG(9, ParallelCoordinate(0, 0, 0, 1, 0))
    first = rng.activation_seed(layer=3, microbatch=2, global_token_offset=128)

    assert first == rng.activation_seed(layer=3, microbatch=2, global_token_offset=128)
    assert first != rng.activation_seed(layer=3, microbatch=3, global_token_offset=128)
