from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from nano_megatron.config import DistributedConfig, ParallelConfig
from nano_megatron.context_parallel import (
    AllGatherContextParallelAttention,
    RingContextParallelAttention,
)
from nano_megatron.distributed import DistributedRuntime
from nano_megatron.parallel import ParallelContext


def _parity_worker(rank: int, world_size: int, rendezvous: str) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    runtime = DistributedRuntime(
        DistributedConfig(
            backend="gloo",
            device="cpu",
            init_method=f"file://{rendezvous}",
        )
    ).initialize()
    parallel = ParallelContext.create(
        runtime,
        ParallelConfig(context=world_size, data=1),
    )
    try:
        local_sequence = 3
        cases = (
            (True, rank * local_sequence),
            (True, 0),
            (False, rank * local_sequence),
        )
        for case_index, (causal, sequence_offset) in enumerate(cases):
            generator = torch.Generator().manual_seed(100 + 17 * rank + case_index)
            inputs = (
                torch.randn(1, 4, local_sequence, 4, generator=generator),
                torch.randn(1, 2, local_sequence, 4, generator=generator),
                torch.randn(1, 2, local_sequence, 4, generator=generator),
            )
            reference_inputs = tuple(tensor.clone().requires_grad_() for tensor in inputs)
            ring_inputs = tuple(tensor.clone().requires_grad_() for tensor in inputs)
            output_gradient = torch.randn(
                1,
                4,
                local_sequence,
                4,
                generator=generator,
            )
            reference = AllGatherContextParallelAttention(parallel.cp)(
                *reference_inputs,
                causal=causal,
                sequence_offset=sequence_offset,
            )
            reference.backward(output_gradient)
            ring = RingContextParallelAttention(parallel.cp)(
                *ring_inputs,
                causal=causal,
                sequence_offset=sequence_offset,
            )
            ring.backward(output_gradient)

            torch.testing.assert_close(ring, reference, atol=3.0e-5, rtol=3.0e-5)
            for actual, expected in zip(ring_inputs, reference_inputs, strict=True):
                torch.testing.assert_close(
                    actual.grad,
                    expected.grad,
                    atol=4.0e-5,
                    rtol=4.0e-5,
                )
    finally:
        parallel.close()
        runtime.close()


@pytest.mark.distributed
def test_ring_attention_matches_all_gather_on_two_gloo_processes(tmp_path: Path) -> None:
    rendezvous = tmp_path / "cp-ring.rendezvous"
    mp.spawn(
        _parity_worker,
        args=(2, str(rendezvous)),
        nprocs=2,
        join=True,
    )
