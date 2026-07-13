from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nano_megatron.config import DistributedConfig, ParallelConfig
from nano_megatron.distributed import DistributedRuntime
from nano_megatron.parallel import GroupKey, GroupPlan, GroupSpec, ParallelAxis, ParallelContext


def test_single_process_context_is_explicit_and_collective_free() -> None:
    config = DistributedConfig(backend="gloo", device="cpu")
    with DistributedRuntime(config) as runtime:
        parallel = ParallelContext.create(
            runtime,
            ParallelConfig(data=None, sequence_parallel=True),
        )

        assert parallel.rank == 0
        assert parallel.world_size == 1
        assert parallel.config.data == 1
        assert parallel.sequence_parallel
        assert parallel.coordinate.tp == 0
        assert parallel.axis_group(ParallelAxis.TP) is parallel.tp
        assert parallel.dense_replica.size == 1
        assert parallel.expert_replica.size == 1
        assert parallel.batch_replica.size == 1
        assert parallel.tp.process_group is None
        assert parallel.is_pipeline_first_stage()
        assert parallel.is_pipeline_last_stage()
        assert parallel.pipeline_prev_rank() is None
        assert parallel.pipeline_next_rank() is None

        parallel.close()
        parallel.close()


def test_context_rejects_a_plan_that_redefines_builtin_semantics() -> None:
    config = DistributedConfig(backend="gloo", device="cpu")
    with DistributedRuntime(config) as runtime:
        invalid_plan = GroupPlan(
            GroupSpec(GroupKey.TP, frozenset({ParallelAxis.DP})),
        )
        with pytest.raises(ValueError, match="fixed Nano-Megatron semantics|missing required"):
            ParallelContext.create(runtime, ParallelConfig(data=1), invalid_plan)


def _two_rank_worker(rank: int, rendezvous: str, output_directory: str) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "2"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    config = DistributedConfig(
        backend="gloo",
        device="cpu",
        init_method=f"file://{rendezvous}",
    )
    with DistributedRuntime(config) as runtime:
        parallel = ParallelContext.create(
            runtime,
            ParallelConfig(tensor=2, data=1),
        )
        value = torch.tensor(float(rank + 1))
        dist.all_reduce(value, group=parallel.tp.process_group)

        result = {
            "rank": parallel.rank,
            "coordinate_tp": parallel.coordinate.tp,
            "tp_ranks": list(parallel.tp.ranks),
            "tp_rank": parallel.tp.rank,
            "dense_replica_ranks": list(parallel.dense_replica.ranks),
            "sum": value.item(),
            "tp_family": [list(ranks) for ranks in parallel.group_family(GroupKey.TP)],
        }
        Path(output_directory, f"rank-{rank}.json").write_text(json.dumps(result))
        parallel.close()


@pytest.mark.distributed
def test_two_rank_gloo_context_materializes_real_groups(tmp_path: Path) -> None:
    rendezvous = tmp_path / "rendezvous"
    output = tmp_path / "results"
    output.mkdir()

    mp.spawn(
        _two_rank_worker,
        args=(str(rendezvous), str(output)),
        nprocs=2,
        join=True,
    )

    results = [json.loads((output / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert [result["coordinate_tp"] for result in results] == [0, 1]
    assert [result["tp_rank"] for result in results] == [0, 1]
    assert all(result["tp_ranks"] == [0, 1] for result in results)
    assert all(result["tp_family"] == [[0, 1]] for result in results)
    assert all(result["sum"] == pytest.approx(3.0) for result in results)
    assert [result["dense_replica_ranks"] for result in results] == [[0], [1]]
