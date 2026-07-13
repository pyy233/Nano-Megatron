from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from nano_megatron.config import (
    DataParallelConfig,
    DistributedConfig,
    OffloadConfig,
    OptimizerConfig,
    ParallelConfig,
)
from nano_megatron.data_parallel import DDPStrategy
from nano_megatron.distributed import DistributedRuntime
from nano_megatron.parallel import ParallelContext, ParameterDomain, ParameterDomainRegistry


class _NormFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tensor_shard = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.tp_replicated = nn.Parameter(torch.zeros((), dtype=torch.float64))


def _global_norm_worker(rank: int, world_size: int, rendezvous: str) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    runtime = DistributedRuntime(
        DistributedConfig(
            backend="gloo",
            device="cpu",
            init_method=f"file://{rendezvous}",
        )
    ).initialize()
    parallel = ParallelContext.create(
        runtime,
        ParallelConfig(tensor=2, pipeline=2, data=1),
    )
    try:
        model = _NormFixture()
        registry = ParameterDomainRegistry()
        registry.register(
            model.tensor_shard,
            ParameterDomain.DENSE,
            tensor_sharded=True,
            name="tensor_shard",
        )
        registry.register(
            model.tp_replicated,
            ParameterDomain.DENSE,
            tensor_sharded=False,
            name="tp_replicated",
        )
        strategy = DDPStrategy(
            config=DataParallelConfig(),
            offload=OffloadConfig(),
            parallel=parallel,
            parameter_domains=registry,
        )
        strategy.setup(model, OptimizerConfig(), registry)

        shard_value = 1.0 + 2.0 * parallel.coordinate.pp + parallel.coordinate.tp
        replicated_value = 5.0 + parallel.coordinate.pp
        model.tensor_shard.grad = torch.tensor(shard_value, dtype=torch.float64)
        model.tp_replicated.grad = torch.tensor(replicated_value, dtype=torch.float64)

        expected = math.sqrt(sum(value * value for value in (1, 2, 3, 4, 5, 6)))
        total_norm = strategy.clip_grad_norm(expected / 2.0)
        torch.testing.assert_close(
            total_norm,
            torch.tensor(expected, dtype=total_norm.dtype),
            atol=2.0e-6,
            rtol=2.0e-6,
        )
        coefficient = (expected / 2.0) / (expected + 1.0e-6)
        torch.testing.assert_close(
            model.tensor_shard.grad,
            torch.tensor(shard_value * coefficient, dtype=torch.float64),
            atol=2.0e-6,
            rtol=2.0e-6,
        )
        torch.testing.assert_close(
            model.tp_replicated.grad,
            torch.tensor(replicated_value * coefficient, dtype=torch.float64),
            atol=2.0e-6,
            rtol=2.0e-6,
        )
    finally:
        parallel.close()
        runtime.close()


@pytest.mark.distributed
def test_global_grad_norm_counts_tp_shards_pp_stages_and_one_replicated_copy(
    tmp_path: Path,
) -> None:
    rendezvous = tmp_path / "global-grad-norm.rendezvous"
    mp.spawn(
        _global_norm_worker,
        args=(4, str(rendezvous)),
        nprocs=4,
        join=True,
    )
