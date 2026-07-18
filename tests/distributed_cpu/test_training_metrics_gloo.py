from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from nano_megatron.config import DistributedConfig, ParallelConfig
from nano_megatron.distributed import DistributedRuntime
from nano_megatron.parallel import ParallelContext
from nano_megatron.training import aggregate_loss_metrics

_WORLD_SIZE = 4


def _worker(rank: int, rendezvous: str, output: str, replica_axis: str) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(_WORLD_SIZE), LOCAL_RANK=str(rank))
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    distributed = DistributedConfig(
        backend="gloo",
        device="cpu",
        init_method=f"file://{rendezvous}",
    )
    axis_sizes = {"tensor": 1, "data": 1, "context": 1, "expert": 1}
    axis_sizes[replica_axis] = 2
    parallel_config = ParallelConfig(
        pipeline=2,
        tensor=axis_sizes["tensor"],
        data=axis_sizes["data"],
        context=axis_sizes["context"],
        expert=axis_sizes["expert"],
    )
    with DistributedRuntime(distributed) as runtime:
        parallel = ParallelContext.create(runtime, parallel_config)
        local: dict[str, float] = {}
        if parallel.is_pipeline_last_stage():
            replica_index = parallel.coordinate.dp + parallel.coordinate.cp + parallel.coordinate.ep
            local = {
                "loss_sum": 4.0 + 5.0 * replica_index,
                "token_count": 2.0 + replica_index,
                "loss": (4.0 + 5.0 * replica_index) / (2.0 + replica_index),
            }
        metrics = aggregate_loss_metrics(local, parallel, device=torch.device("cpu"))
        Path(output, f"rank-{rank}.json").write_text(json.dumps(metrics))


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("replica_axis", "expected_sum", "expected_count", "expected_loss"),
    [
        ("data", 13.0, 5.0, 2.6),
        ("context", 13.0, 5.0, 2.6),
        ("expert", 13.0, 5.0, 2.6),
        ("tensor", 4.0, 2.0, 2.0),
    ],
)
def test_loss_totals_reduce_over_semantic_replicas_then_route_over_pp(
    tmp_path: Path,
    replica_axis: str,
    expected_sum: float,
    expected_count: float,
    expected_loss: float,
) -> None:
    output = tmp_path / f"metrics-{replica_axis}"
    output.mkdir()
    mp.spawn(
        _worker,
        args=(
            str(tmp_path / f"metrics-{replica_axis}.rendezvous"),
            str(output),
            replica_axis,
        ),
        nprocs=_WORLD_SIZE,
        join=True,
    )

    metrics = [
        json.loads((output / f"rank-{rank}.json").read_text())
        for rank in range(_WORLD_SIZE)
    ]
    assert all(item["loss_sum"] == pytest.approx(expected_sum) for item in metrics)
    assert all(item["token_count"] == pytest.approx(expected_count) for item in metrics)
    assert all(item["loss"] == pytest.approx(expected_loss) for item in metrics)
