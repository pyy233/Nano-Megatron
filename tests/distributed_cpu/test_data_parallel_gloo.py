from __future__ import annotations

import math
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

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
from nano_megatron.pipeline_parallel import GPipeSchedule


class _SingleStageParallel:
    @staticmethod
    def is_pipeline_first_stage() -> bool:
        return True

    @staticmethod
    def is_pipeline_last_stage() -> bool:
        return True


class _RegressionStage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 1, bias=False)

    def forward(self, hidden_states, batch):
        del hidden_states
        return self.projection(batch["input"]).square().mean()


class _TwoLayerRegressionStage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(2, 4, bias=False)
        self.output_projection = nn.Linear(4, 1, bias=False)

    def forward(self, hidden_states, batch):
        del hidden_states
        hidden = torch.tanh(self.input_projection(batch["input"]))
        return self.output_projection(hidden).square().mean()


def _zero_worker(rank: int, world_size: int, rendezvous: str, mode: str) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    runtime = DistributedRuntime(
        DistributedConfig(
            backend="gloo",
            device="cpu",
            init_method=f"file://{rendezvous}",
        )
    ).initialize()
    parallel = ParallelContext.create(runtime, ParallelConfig(data=world_size))
    try:
        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        registry = ParameterDomainRegistry()
        registry.register_module(model, ParameterDomain.DENSE)
        strategy = build_data_parallel_strategy(
            DataParallelConfig(mode=mode, bucket_bytes=128),
            OffloadConfig(),
            parallel,
            registry,
        )
        strategy.setup(
            model,
            OptimizerConfig(lr=0.01, weight_decay=0.0),
            registry,
        )

        scale = float(1 + 2 * rank)
        strategy.backward((model.weight * scale).sum())
        strategy.finalize_gradients()
        assert len(strategy.buckets) == 1
        local_gradient = strategy.buckets[0].local_gradient
        assert local_gradient is not None
        torch.testing.assert_close(local_gradient, torch.tensor([2.0]))
        expected_norm = math.sqrt(8.0)
        norm = strategy.clip_grad_norm(1.0)
        coefficient = 1.0 / (expected_norm + 1.0e-6)
        torch.testing.assert_close(norm, torch.tensor(expected_norm))
        torch.testing.assert_close(
            strategy.buckets[0].local_gradient,
            torch.tensor([2.0 * coefficient]),
        )
        strategy.optimizer_step()

        gathered = [torch.empty_like(model.weight) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, model.weight)
        for replica in gathered[1:]:
            torch.testing.assert_close(replica, gathered[0])
    finally:
        parallel.close()
        runtime.close()


def _zero3_worker(rank: int, world_size: int, rendezvous: str) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    runtime = DistributedRuntime(
        DistributedConfig(
            backend="gloo",
            device="cpu",
            init_method=f"file://{rendezvous}",
        )
    ).initialize()
    parallel = ParallelContext.create(runtime, ParallelConfig(data=world_size))
    try:
        torch.manual_seed(11)
        model = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 2))
        reference = deepcopy(model)
        registry = ParameterDomainRegistry()
        registry.register_module(model, ParameterDomain.DENSE)
        strategy = build_data_parallel_strategy(
            DataParallelConfig(mode="zero3"),
            OffloadConfig(),
            parallel,
            registry,
        )
        optimizer_config = OptimizerConfig(lr=0.01)
        wrapped = strategy.setup(model, optimizer_config, registry)
        reference_optimizer = torch.optim.AdamW(
            reference.parameters(),
            lr=optimizer_config.lr,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
        )
        assert strategy.uses_fsdp2
        inputs = torch.randn(3, 4)
        with strategy.microbatch_context(is_last_microbatch=True):
            strategy.backward(wrapped(inputs).square().mean())
        reference(inputs).square().mean().backward()
        expected_norm = (
            torch.stack(
                [parameter.grad.float().square().sum() for parameter in reference.parameters()]
            )
            .sum()
            .sqrt()
        )
        max_norm = expected_norm / 2.0
        norm = strategy.clip_grad_norm(float(max_norm))
        torch.testing.assert_close(norm, expected_norm, atol=2.0e-5, rtol=2.0e-5)
        coefficient = torch.clamp(max_norm / (expected_norm + 1.0e-6), max=1.0)
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            assert actual.grad is not None and expected.grad is not None
            full_tensor = getattr(actual.grad, "full_tensor", None)
            actual_gradient = full_tensor() if callable(full_tensor) else actual.grad
            expected.grad.mul_(coefficient)
            torch.testing.assert_close(
                actual_gradient,
                expected.grad,
                atol=2.0e-5,
                rtol=2.0e-5,
            )
        strategy.optimizer_step()
        reference_optimizer.step()
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            full_tensor = getattr(actual, "full_tensor", None)
            actual_parameter = full_tensor() if callable(full_tensor) else actual
            torch.testing.assert_close(
                actual_parameter,
                expected,
                atol=2.0e-5,
                rtol=2.0e-5,
            )
        assert "optimizer_step" in strategy.trace
    finally:
        parallel.close()
        runtime.close()


def _ddp_accumulation_worker(rank: int, world_size: int, rendezvous: str) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    runtime = DistributedRuntime(
        DistributedConfig(
            backend="gloo",
            device="cpu",
            init_method=f"file://{rendezvous}",
        )
    ).initialize()
    parallel = ParallelContext.create(runtime, ParallelConfig(data=world_size))
    try:
        model = _RegressionStage()
        with torch.no_grad():
            model.projection.weight.copy_(torch.tensor([[0.5, -0.25]]))
        reference = _RegressionStage()
        reference.load_state_dict(model.state_dict())
        optimizer_config = OptimizerConfig(lr=0.01, weight_decay=0.0)
        registry = ParameterDomainRegistry()
        registry.register_module(model, ParameterDomain.DENSE)
        strategy = build_data_parallel_strategy(
            DataParallelConfig(mode="ddp"),
            OffloadConfig(),
            parallel,
            registry,
        )
        wrapped = strategy.setup(model, optimizer_config, registry)
        microbatches = [
            {
                "input": torch.tensor(
                    [[rank + index + 1.0, 0.5 * (index + 1)]],
                    dtype=torch.float32,
                )
            }
            for index in range(3)
        ]
        GPipeSchedule(_SingleStageParallel()).forward_backward(
            stage=wrapped,
            microbatches=microbatches,
            data_parallel=strategy,
        )
        strategy.finalize_gradients()
        strategy.optimizer_step()

        reference_optimizer = torch.optim.AdamW(
            reference.parameters(),
            lr=optimizer_config.lr,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
        )
        for replica_rank in range(world_size):
            for index in range(3):
                value = torch.tensor(
                    [[replica_rank + index + 1.0, 0.5 * (index + 1)]],
                    dtype=torch.float32,
                )
                reference(None, {"input": value}).div_(3 * world_size).backward()
        reference_optimizer.step()

        assert strategy.gradient_sync_count == 1
        torch.testing.assert_close(
            model.projection.weight,
            reference.projection.weight,
            rtol=1.0e-6,
            atol=1.0e-7,
        )
    finally:
        parallel.close()
        runtime.close()


def _ddp_reduction_precision_worker(rank: int, world_size: int, rendezvous: str) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    runtime = DistributedRuntime(
        DistributedConfig(
            backend="gloo",
            device="cpu",
            init_method=f"file://{rendezvous}",
        )
    ).initialize()
    parallel = ParallelContext.create(runtime, ParallelConfig(data=world_size))
    try:
        model = nn.Linear(2, 1, bias=False).to(dtype=torch.bfloat16)
        registry = ParameterDomainRegistry()
        registry.register_module(model, ParameterDomain.DENSE)
        strategy = build_data_parallel_strategy(
            DataParallelConfig(mode="ddp"),
            OffloadConfig(),
            parallel,
            registry,
        )
        strategy.configure_precision(
            SimpleNamespace(
                params="bfloat16",
                compute="bfloat16",
                grad_reduce="float32",
            )
        )
        wrapped = strategy.setup(model, OptimizerConfig(lr=0.01), registry)
        value = torch.tensor([[rank + 1.0, 2.0 - rank]], dtype=torch.bfloat16)
        strategy.backward(wrapped(value).float().square().mean())
        strategy.finalize_gradients()
        assert strategy.gradient_sync_count == 1
        assert all(
            bucket.local_gradient is not None
            and bucket.local_gradient.dtype is torch.float32
            for bucket in strategy._buckets
        )
        strategy.optimizer_step()

        gathered = [torch.empty_like(model.weight) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, model.weight)
        for replica in gathered[1:]:
            torch.testing.assert_close(replica, gathered[0])
    finally:
        parallel.close()
        runtime.close()


def _zero3_accumulation_worker(rank: int, world_size: int, rendezvous: str) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    runtime = DistributedRuntime(
        DistributedConfig(
            backend="gloo",
            device="cpu",
            init_method=f"file://{rendezvous}",
        )
    ).initialize()
    parallel = ParallelContext.create(runtime, ParallelConfig(data=world_size))
    try:
        torch.manual_seed(101)
        model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
        reference = deepcopy(model)
        registry = ParameterDomainRegistry()
        registry.register_module(model, ParameterDomain.DENSE)
        strategy = build_data_parallel_strategy(
            DataParallelConfig(mode="zero3"),
            OffloadConfig(),
            parallel,
            registry,
        )
        wrapped = strategy.setup(model, OptimizerConfig(weight_decay=0.0), registry)

        for microbatch in range(2):
            inputs = torch.full((2, 3), rank + microbatch + 1.0)
            loss = wrapped(inputs).square().mean() / 2
            with strategy.microbatch_context(is_last_microbatch=microbatch == 1):
                strategy.backward(loss)

        for replica_rank in range(world_size):
            for microbatch in range(2):
                inputs = torch.full((2, 3), replica_rank + microbatch + 1.0)
                (reference(inputs).square().mean() / (2 * world_size)).backward()

        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            assert actual.grad is not None and expected.grad is not None
            full_tensor = getattr(actual.grad, "full_tensor", None)
            actual_gradient = full_tensor() if callable(full_tensor) else actual.grad
            torch.testing.assert_close(
                actual_gradient,
                expected.grad,
                atol=3.0e-6,
                rtol=3.0e-5,
            )
    finally:
        parallel.close()
        runtime.close()


def _gradient_overlap_parity_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    mode: str,
    overlap: bool,
) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    runtime = DistributedRuntime(
        DistributedConfig(
            backend="gloo",
            device="cpu",
            init_method=f"file://{rendezvous}",
        )
    ).initialize()
    parallel = ParallelContext.create(runtime, ParallelConfig(data=world_size))
    try:
        torch.manual_seed(909)
        model = _TwoLayerRegressionStage()
        reference = deepcopy(model)
        registry = ParameterDomainRegistry()
        registry.register_module(model, ParameterDomain.DENSE)
        optimizer_config = OptimizerConfig(lr=0.01, weight_decay=0.0)
        strategy = build_data_parallel_strategy(
            DataParallelConfig(
                mode=mode,
                bucket_bytes=16,
                overlap_grad_reduce=overlap,
            ),
            OffloadConfig(),
            parallel,
            registry,
        )
        wrapped = strategy.setup(model, optimizer_config, registry)
        strategy.zero_grad()
        microbatches = [
            {
                "input": torch.tensor(
                    [[rank + microbatch + 1.0, 0.25 * (microbatch + 1)]],
                    dtype=torch.float32,
                )
            }
            for microbatch in range(2)
        ]
        GPipeSchedule(_SingleStageParallel()).forward_backward(
            stage=wrapped,
            microbatches=microbatches,
            data_parallel=strategy,
        )

        reducer = strategy.gradient_reducer
        assert reducer is not None
        assert len(reducer.buckets) == 2
        if overlap:
            assert reducer.start_count == 2
            assert reducer.wait_count == 0
            assert all(not request.completed for request in reducer.requests)
        else:
            assert reducer.start_count == 0

        strategy.finalize_gradients()
        assert reducer.start_count == 2
        assert reducer.wait_count == 2

        reference_optimizer = torch.optim.AdamW(
            reference.parameters(),
            lr=optimizer_config.lr,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
        )
        for replica_rank in range(world_size):
            for microbatch in range(2):
                value = torch.tensor(
                    [[replica_rank + microbatch + 1.0, 0.25 * (microbatch + 1)]],
                    dtype=torch.float32,
                )
                reference(None, {"input": value}).div_(2 * world_size).backward()

        expected_norm = (
            torch.stack(
                [parameter.grad.float().square().sum() for parameter in reference.parameters()]
            )
            .sum()
            .sqrt()
        )
        norm = strategy.clip_grad_norm(1.0e6)
        torch.testing.assert_close(norm, expected_norm, atol=1.0e-6, rtol=1.0e-6)

        strategy.optimizer_step()
        reference_optimizer.step()
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, atol=1.0e-7, rtol=1.0e-6)
    finally:
        parallel.close()
        runtime.close()


@pytest.mark.distributed
@pytest.mark.parametrize("mode", ["zero1", "zero2"])
def test_zero_flat_shards_on_two_gloo_processes(tmp_path: Path, mode: str) -> None:
    rendezvous = tmp_path / f"{mode}.rendezvous"
    mp.spawn(
        _zero_worker,
        args=(2, str(rendezvous), mode),
        nprocs=2,
        join=True,
    )


@pytest.mark.distributed
def test_zero3_fsdp2_adapter_on_two_gloo_processes(tmp_path: Path) -> None:
    rendezvous = tmp_path / "zero3.rendezvous"
    mp.spawn(
        _zero3_worker,
        args=(2, str(rendezvous)),
        nprocs=2,
        join=True,
    )


@pytest.mark.distributed
def test_ddp_accumulation_communicates_once_on_two_gloo_processes(tmp_path: Path) -> None:
    rendezvous = tmp_path / "ddp-accumulation.rendezvous"
    mp.spawn(
        _ddp_accumulation_worker,
        args=(2, str(rendezvous)),
        nprocs=2,
        join=True,
    )


@pytest.mark.distributed
def test_ddp_uses_configured_fp32_reduction_on_two_gloo_processes(tmp_path: Path) -> None:
    rendezvous = tmp_path / "ddp-reduction-precision.rendezvous"
    mp.spawn(
        _ddp_reduction_precision_worker,
        args=(2, str(rendezvous)),
        nprocs=2,
        join=True,
    )


@pytest.mark.distributed
def test_zero3_accumulation_matches_global_reference_on_two_gloo_processes(
    tmp_path: Path,
) -> None:
    rendezvous = tmp_path / "zero3-accumulation.rendezvous"
    mp.spawn(
        _zero3_accumulation_worker,
        args=(2, str(rendezvous)),
        nprocs=2,
        join=True,
    )


@pytest.mark.distributed
@pytest.mark.parametrize("mode", ["ddp", "zero1", "zero2"])
@pytest.mark.parametrize("overlap", [False, True])
def test_bucket_gradient_overlap_matches_global_reference_on_two_gloo_processes(
    tmp_path: Path,
    mode: str,
    overlap: bool,
) -> None:
    rendezvous = tmp_path / f"{mode}-overlap-{overlap}.rendezvous"
    mp.spawn(
        _gradient_overlap_parity_worker,
        args=(2, str(rendezvous), mode, overlap),
        nprocs=2,
        join=True,
    )
