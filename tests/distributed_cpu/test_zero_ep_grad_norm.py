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
from nano_megatron.data_parallel import build_data_parallel_strategy
from nano_megatron.distributed import DistributedRuntime
from nano_megatron.parallel import ParallelContext, ParameterDomain, ParameterDomainRegistry


def _worker(rank: int, rendezvous: str, mode: str) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank))
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
        ParallelConfig(expert=2, data=1),
    )
    try:
        model = nn.Linear(4, 1, bias=False)
        registry = ParameterDomainRegistry()
        registry.register_module(model, ParameterDomain.DENSE)
        strategy = build_data_parallel_strategy(
            DataParallelConfig(mode=mode, bucket_bytes=128),
            OffloadConfig(),
            parallel,
            registry,
        )
        strategy.setup(model, OptimizerConfig(lr=0.01), registry)
        inputs = (
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            torch.tensor([[5.0, 6.0, 7.0, 8.0]]),
        )
        strategy.zero_grad()
        strategy.backward(model(inputs[rank]).sum())
        strategy.finalize_gradients()
        norm = strategy.clip_grad_norm(1_000.0)

        expected_gradient = (inputs[0] + inputs[1]) / 2
        expected_norm = math.sqrt(float(expected_gradient.square().sum()))
        torch.testing.assert_close(
            norm,
            torch.tensor(expected_norm),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
    finally:
        parallel.close()
        runtime.close()


@pytest.mark.distributed
@pytest.mark.parametrize("mode", ["zero1", "zero2"])
def test_zero_global_grad_norm_keeps_all_dense_ep_shards(tmp_path: Path, mode: str) -> None:
    rendezvous = tmp_path / f"{mode}-ep2.rendezvous"
    mp.spawn(
        _worker,
        args=(str(rendezvous), mode),
        nprocs=2,
        join=True,
    )
