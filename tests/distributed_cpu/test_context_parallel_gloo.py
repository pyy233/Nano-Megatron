from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nano_megatron.context_parallel import (
    AllGatherContextParallelAttention,
    RingContextParallelAttention,
)
from nano_megatron.nn.kernels import TorchKernelBackend
from nano_megatron.parallel import GroupKey, ParallelGroup


def _cp_worker(rank: int, world_size: int, init_file: str, backend: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        group = ParallelGroup(
            key=GroupKey.CP,
            ranks=tuple(range(world_size)),
            process_group=dist.group.WORLD,
            rank=rank,
            size=world_size,
            backend="gloo",
        )
        generator = torch.Generator().manual_seed(41)
        full_q = torch.randn(1, 4, 6, 4, generator=generator, dtype=torch.float64)
        full_k = torch.randn(1, 2, 6, 4, generator=generator, dtype=torch.float64)
        full_v = torch.randn(1, 2, 6, 4, generator=generator, dtype=torch.float64)

        reference_q = full_q.detach().clone().requires_grad_(True)
        reference_k = full_k.detach().clone().requires_grad_(True)
        reference_v = full_v.detach().clone().requires_grad_(True)
        reference_output = TorchKernelBackend().local_attention(
            reference_q,
            reference_k,
            reference_v,
            causal=True,
        )
        reference_output.square().sum().backward()

        local_q = full_q.chunk(world_size, dim=-2)[rank].detach().clone().requires_grad_(True)
        local_k = full_k.chunk(world_size, dim=-2)[rank].detach().clone().requires_grad_(True)
        local_v = full_v.chunk(world_size, dim=-2)[rank].detach().clone().requires_grad_(True)
        attention = (
            AllGatherContextParallelAttention(group)
            if backend == "all_gather"
            else RingContextParallelAttention(group)
        )
        local_output = attention(
            local_q,
            local_k,
            local_v,
            sequence_offset=rank * local_q.size(-2),
        )
        local_output.square().sum().backward()

        torch.testing.assert_close(
            local_output,
            reference_output.chunk(world_size, dim=-2)[rank],
            atol=2.0e-9,
            rtol=2.0e-7,
        )
        for actual, expected in (
            (local_q.grad, reference_q.grad.chunk(world_size, dim=-2)[rank]),
            (local_k.grad, reference_k.grad.chunk(world_size, dim=-2)[rank]),
            (local_v.grad, reference_v.grad.chunk(world_size, dim=-2)[rank]),
        ):
            assert actual is not None
            torch.testing.assert_close(actual, expected, atol=2.0e-8, rtol=2.0e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.parametrize("backend", ["all_gather", "ring"])
def test_cp_attention_matches_full_gqa_on_two_gloo_processes(backend: str) -> None:
    world_size = 2
    handle, init_file = tempfile.mkstemp(prefix=f"nano-megatron-cp-{backend}-")
    os.close(handle)
    os.unlink(init_file)
    try:
        mp.spawn(
            _cp_worker,
            args=(world_size, init_file, backend),
            nprocs=world_size,
            join=True,
        )
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)
