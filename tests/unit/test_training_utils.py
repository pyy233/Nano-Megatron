from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from nano_megatron.training import BatchRouter  # noqa: E402
from nano_megatron.training.microbatches import (  # noqa: E402
    global_batch_size,
    split_microbatches,
)


@dataclass(frozen=True)
class _Group:
    rank: int = 0
    size: int = 1
    process_group: object | None = None


@dataclass(frozen=True)
class _Parallel:
    cp: _Group
    batch_replica: _Group = _Group()


_FAKE_PROCESS_GROUP = object()


def test_global_batch_includes_independent_ep_axis() -> None:
    assert global_batch_size(2, 4, data_parallel_size=3, expert_parallel_size=2) == 48


def test_split_nested_microbatches() -> None:
    batch = {
        "tokens": torch.arange(24).view(6, 4),
        "labels": torch.arange(24).view(6, 4) + 1,
        "constant": torch.tensor(7),
    }
    microbatches = split_microbatches(batch, 2)
    assert len(microbatches) == 3
    assert microbatches[1]["tokens"].shape == (2, 4)
    assert microbatches[2]["constant"].item() == 7


def test_split_rejects_inconsistent_batch_dimensions() -> None:
    with pytest.raises(ValueError, match="leading dimension"):
        split_microbatches({"a": torch.zeros(4, 2), "b": torch.zeros(3, 2)}, 1)


def test_batch_router_slices_standard_sequence_fields_for_cp_rank() -> None:
    parallel = _Parallel(
        cp=_Group(rank=1, size=2, process_group=_FAKE_PROCESS_GROUP),
    )
    batch = {
        "input_ids": torch.arange(16).view(2, 8),
        "labels": torch.arange(16).view(2, 8) + 1,
        "position_ids": torch.arange(8),
        "metadata": torch.tensor(7),
    }

    routed = BatchRouter(parallel).route(batch)

    torch.testing.assert_close(routed["input_ids"], batch["input_ids"][:, 4:])
    torch.testing.assert_close(routed["labels"], batch["labels"][:, 4:])
    torch.testing.assert_close(routed["position_ids"], batch["position_ids"][4:])
    assert routed["metadata"].item() == 7


def test_batch_router_rejects_non_divisible_cp_sequence() -> None:
    parallel = _Parallel(
        cp=_Group(rank=0, size=2, process_group=_FAKE_PROCESS_GROUP),
    )
    with pytest.raises(ValueError, match="divisible"):
        BatchRouter(parallel).route({"input_ids": torch.arange(5).view(1, 5)})


def test_batch_router_requires_explicit_batch_replica_group() -> None:
    parallel = type("Parallel", (), {"cp": _Group()})()
    with pytest.raises(TypeError, match="batch-replica group"):
        BatchRouter(parallel)


def test_batch_router_requires_explicit_context_parallel_group() -> None:
    parallel = type("Parallel", (), {"batch_replica": _Group()})()
    router = BatchRouter(parallel)
    with pytest.raises(TypeError, match="context-parallel group"):
        router.route({"input_ids": torch.arange(4).view(1, 4)})


def test_batch_router_rejects_unmaterialized_multi_rank_group() -> None:
    parallel = _Parallel(
        cp=_Group(),
        batch_replica=_Group(rank=0, size=2, process_group=None),
    )
    with pytest.raises(ValueError, match="materialized process_group"):
        BatchRouter(parallel)
