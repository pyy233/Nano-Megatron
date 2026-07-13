from __future__ import annotations

import os
import tempfile
from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn

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

pytestmark = [
    pytest.mark.distributed,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.device_count() < 2
        or not dist.is_nccl_available(),
        reason="requires two CUDA devices and NCCL",
    ),
]


class _RegressionStage(nn.Module):
    """Small model whose reverse parameter order yields three independent buckets."""

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(4, 4, bias=False)
        self.hidden_projection = nn.Linear(4, 4, bias=False)
        self.output_projection = nn.Linear(4, 2, bias=False)

    def forward(self, hidden_states: Tensor | None, batch: dict[str, Tensor]) -> Tensor:
        assert hidden_states is None
        hidden = torch.tanh(self.input_projection(batch["input"]))
        hidden = torch.sin(self.hidden_projection(hidden))
        prediction = self.output_projection(hidden)
        return (prediction - batch["target"]).square().mean()


def _batch(rank: int, step: int, microbatch: int, device: torch.device) -> dict[str, Tensor]:
    sample = torch.arange(12, dtype=torch.float32, device=device).view(3, 4)
    target = torch.arange(6, dtype=torch.float32, device=device).view(3, 2)
    input_scale = 0.025 * (1 + rank + 2 * step + microbatch)
    target_shift = 0.1 * (rank - step + 2 * microbatch)
    return {
        "input": (sample + 1.0) * input_scale,
        "target": target * 0.04 + target_shift,
    }


def _reference_backward(
    reference: _RegressionStage,
    *,
    step: int,
    world_size: int,
    microbatch_count: int,
    device: torch.device,
) -> None:
    divisor = world_size * microbatch_count
    for replica_rank in range(world_size):
        for microbatch in range(microbatch_count):
            loss = reference(None, _batch(replica_rank, step, microbatch, device))
            (loss / divisor).backward()


def _expected_bucket_gradient(
    bucket: object,
    reference_parameters: dict[str, nn.Parameter],
) -> Tensor:
    expected = torch.zeros(
        bucket.padded_numel,
        dtype=bucket.dtype,
        device=bucket.device,
    )
    for item in bucket.slices:
        gradient = reference_parameters[item.name].grad
        assert gradient is not None
        expected[item.start : item.end].copy_(gradient.reshape(-1))
    return expected


def _assert_overlap_started(strategy: object, mode: str) -> None:
    reducer = strategy.gradient_reducer
    assert reducer is not None
    assert len(reducer.buckets) == 3
    assert reducer.start_count == len(reducer.buckets)
    assert reducer.wait_count == 0
    assert not reducer.finalized
    assert len(reducer.requests) == len(reducer.buckets)

    expected_reduction = "reduce_scatter" if mode == "zero2" else "all_reduce"
    assert all(request.reduction == expected_reduction for request in reducer.requests)
    assert all(request.work is not None for request in reducer.requests)
    assert all(not request.completed for request in reducer.requests)

    if mode == "zero2":
        # This distinguishes the native NCCL reduce-scatter path from the Gloo
        # all-reduce-and-slice compatibility path.
        assert all(request.output_buffer is not None for request in reducer.requests)
        assert all(
            str(request.bucket.group.backend).lower() == "nccl" for request in reducer.requests
        )


def _assert_reduced_gradients(
    strategy: object,
    model: _RegressionStage,
    reference: _RegressionStage,
    mode: str,
) -> None:
    reducer = strategy.gradient_reducer
    assert reducer is not None
    assert reducer.finalized
    assert reducer.start_count == len(reducer.buckets)
    assert reducer.wait_count == len(reducer.buckets)
    assert all(request.completed for request in reducer.requests)

    reference_parameters = dict(reference.named_parameters())
    for bucket in reducer.buckets:
        expected = _expected_bucket_gradient(bucket, reference_parameters)
        if mode == "zero2":
            assert bucket.full_gradient is None
            assert bucket.local_gradient is not None
            actual = bucket.all_gather_shards(bucket.local_gradient)
        else:
            assert bucket.full_gradient is not None
            actual = bucket.full_gradient
        torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=2.0e-6)

    if mode == "zero2":
        assert all(parameter.grad is None for parameter in model.parameters())
    else:
        for (name, actual), expected in zip(
            model.named_parameters(),
            reference.parameters(),
            strict=True,
        ):
            assert actual.grad is not None, name
            assert expected.grad is not None, name
            torch.testing.assert_close(
                actual.grad,
                expected.grad,
                atol=2.0e-6,
                rtol=2.0e-6,
            )


def _run_mode(
    mode: str,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    parallel: ParallelContext,
) -> None:
    torch.manual_seed(1729)
    model = _RegressionStage().to(device)
    registry = ParameterDomainRegistry()
    registry.register_module(model, ParameterDomain.DENSE)
    optimizer_config = OptimizerConfig(
        lr=0.02,
        betas=(0.8, 0.9),
        eps=1.0e-6,
        weight_decay=0.01,
    )
    strategy = build_data_parallel_strategy(
        DataParallelConfig(
            mode=mode,
            bucket_bytes=64,
            overlap_grad_reduce=True,
        ),
        OffloadConfig(),
        parallel,
        registry,
    )
    wrapped = strategy.setup(model, optimizer_config, registry)
    reference = deepcopy(model)
    reference_optimizer = torch.optim.AdamW(
        reference.parameters(),
        lr=optimizer_config.lr,
        betas=optimizer_config.betas,
        eps=optimizer_config.eps,
        weight_decay=optimizer_config.weight_decay,
    )

    microbatch_count = 2
    for step in range(2):
        strategy.zero_grad()
        reference_optimizer.zero_grad(set_to_none=True)
        for microbatch in range(microbatch_count):
            batch = _batch(rank, step, microbatch, device)
            with strategy.microbatch_context(is_last_microbatch=microbatch == microbatch_count - 1):
                loss = wrapped(None, batch) / microbatch_count
                strategy.backward(loss)

        # Hooks on the final accumulated backward must launch NCCL collectives
        # without waiting for them.  This state assertion is deterministic and
        # does not rely on a timing threshold or a particular GPU's speed.
        _assert_overlap_started(strategy, mode)
        _reference_backward(
            reference,
            step=step,
            world_size=world_size,
            microbatch_count=microbatch_count,
            device=device,
        )

        strategy.finalize_gradients()
        _assert_reduced_gradients(strategy, model, reference, mode)
        strategy.optimizer_step()
        reference_optimizer.step()
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, atol=3.0e-6, rtol=3.0e-6)


def _worker(rank: int, world_size: int, rendezvous: str) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    runtime = DistributedRuntime(
        DistributedConfig(
            backend="nccl",
            device="cuda",
            init_method=f"file://{rendezvous}",
        )
    ).initialize()
    parallel = ParallelContext.create(runtime, ParallelConfig(data=world_size))
    try:
        device = torch.device(runtime.device)
        for mode in ("ddp", "zero1", "zero2"):
            _run_mode(
                mode,
                rank=rank,
                world_size=world_size,
                device=device,
                parallel=parallel,
            )
            dist.barrier()
    finally:
        parallel.close()
        runtime.close()


def test_data_parallel_gradient_overlap_nccl() -> None:
    world_size = 2
    handle, rendezvous = tempfile.mkstemp(prefix="nano-megatron-nccl-dp-overlap-")
    os.close(handle)
    os.unlink(rendezvous)
    try:
        mp.spawn(
            _worker,
            args=(world_size, rendezvous),
            nprocs=world_size,
            join=True,
        )
    finally:
        if os.path.exists(rendezvous):
            os.unlink(rendezvous)
