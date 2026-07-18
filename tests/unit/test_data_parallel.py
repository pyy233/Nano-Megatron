from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nano_megatron.config import (
    DataParallelConfig,
    DistributedConfig,
    OffloadConfig,
    OptimizerConfig,
    ParallelConfig,
)
from nano_megatron.data_parallel import (
    DDPStrategy,
    ReplicatedStrategy,
    Zero1Strategy,
    Zero2Strategy,
    Zero3Strategy,
    build_data_parallel_strategy,
)
from nano_megatron.data_parallel._common import _scalar_collective_device
from nano_megatron.distributed import DistributedRuntime
from nano_megatron.parallel import ParallelContext, ParameterDomain, ParameterDomainRegistry


def _parallel_context() -> tuple[DistributedRuntime, ParallelContext]:
    runtime = DistributedRuntime(DistributedConfig(backend="gloo", device="cpu")).initialize()
    parallel = ParallelContext.create(runtime, ParallelConfig(data=1))
    return runtime, parallel


def _registered(model: nn.Module) -> ParameterDomainRegistry:
    registry = ParameterDomainRegistry()
    registry.register_module(model, ParameterDomain.DENSE)
    return registry


class _VirtualShardChunk(nn.Linear):
    def __init__(self) -> None:
        super().__init__(2, 2, bias=False)
        self.gradient_sync_calls: list[tuple[bool, bool]] = []

    def set_requires_gradient_sync(self, value: bool, *, recurse: bool = True) -> None:
        self.gradient_sync_calls.append((value, recurse))


class _VirtualShardContainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.chunks = nn.ModuleList([_VirtualShardChunk(), _VirtualShardChunk()])

    def sharding_units(self) -> tuple[nn.Module, ...]:
        return tuple(self.chunks)


def test_norm_scalar_collective_device_follows_the_explicit_backend() -> None:
    parallel = SimpleNamespace(runtime=SimpleNamespace(device=torch.device("cuda", 1)))
    nccl_group = SimpleNamespace(backend="nccl")
    gloo_group = SimpleNamespace(backend="gloo")

    assert _scalar_collective_device(
        nccl_group,
        parallel,
        torch.device("cpu"),
    ) == torch.device("cuda", 1)
    assert _scalar_collective_device(
        gloo_group,
        parallel,
        torch.device("cuda", 1),
    ) == torch.device("cpu")


@pytest.mark.parametrize(
    ("mode", "strategy_type"),
    [
        ("ddp", DDPStrategy),
        ("zero1", Zero1Strategy),
        ("zero2", Zero2Strategy),
        ("zero3", Zero3Strategy),
    ],
)
def test_factory_and_single_process_step(mode: str, strategy_type: type) -> None:
    runtime, parallel = _parallel_context()
    try:
        model = nn.Linear(3, 2)
        registry = _registered(model)
        strategy = build_data_parallel_strategy(
            DataParallelConfig(mode=mode, bucket_bytes=128),
            OffloadConfig(),
            parallel,
            registry,
        )
        assert isinstance(strategy, strategy_type)
        wrapped = strategy.setup(
            model,
            OptimizerConfig(lr=0.01, weight_decay=0.0),
            registry,
        )
        strategy.set_learning_rate(0.005)
        assert strategy.learning_rate == pytest.approx(0.005)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        loss = wrapped(torch.ones(4, 3)).square().mean()
        strategy.backward(loss)
        norm = strategy.clip_grad_norm(1.0)
        assert torch.isfinite(norm)
        strategy.optimizer_step()
        assert any(
            not torch.equal(old, new)
            for old, new in zip(before, model.parameters(), strict=True)
        )
        if isinstance(strategy, Zero3Strategy):
            assert not strategy.uses_fsdp2
    finally:
        parallel.close()
        runtime.close()


@pytest.mark.parametrize("strategy_type", [Zero1Strategy, Zero2Strategy])
def test_zero_size_one_matches_adamw_with_gradient_accumulation(strategy_type: type) -> None:
    runtime, parallel = _parallel_context()
    try:
        torch.manual_seed(7)
        model = nn.Sequential(nn.Linear(4, 3), nn.Tanh(), nn.Linear(3, 2))
        reference = deepcopy(model)
        registry = _registered(model)
        optimizer_config = OptimizerConfig(lr=0.003, weight_decay=0.1)
        strategy = strategy_type(
            config=DataParallelConfig(mode=strategy_type.mode, bucket_bytes=80),
            offload=OffloadConfig(optimizer_state=True),
            parallel=parallel,
            parameter_domains=registry,
        )
        strategy.setup(model, optimizer_config, registry)
        reference_optimizer = torch.optim.AdamW(
            reference.parameters(),
            lr=optimizer_config.lr,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
        )

        inputs = (torch.randn(2, 4), torch.randn(3, 4))
        for index, batch in enumerate(inputs):
            with strategy.microbatch_context(is_last_microbatch=index == len(inputs) - 1):
                strategy.backward(model(batch).square().mean())
            reference(batch).square().mean().backward()
        strategy.optimizer_step()
        reference_optimizer.step()

        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-6)
        state = strategy.state_dict()
        assert state["step"] == 1
        assert state["buckets"]
    finally:
        parallel.close()
        runtime.close()


def test_replicated_strategy_rejects_missing_parameter_registration() -> None:
    runtime, parallel = _parallel_context()
    try:
        model = nn.Linear(2, 2)
        registry = ParameterDomainRegistry()
        strategy = ReplicatedStrategy(
            config=DataParallelConfig(),
            offload=OffloadConfig(),
            parallel=parallel,
            parameter_domains=registry,
        )
        with pytest.raises(ValueError, match="every trainable parameter"):
            strategy.setup(model, OptimizerConfig(), registry)
    finally:
        parallel.close()
        runtime.close()


@pytest.mark.parametrize("strategy_type", [Zero1Strategy, Zero2Strategy])
def test_final_microbatch_context_handles_direct_autograd_backward(strategy_type: type) -> None:
    runtime, parallel = _parallel_context()
    try:
        model = nn.Linear(2, 2)
        registry = _registered(model)
        strategy = strategy_type(
            config=DataParallelConfig(mode=strategy_type.mode),
            offload=OffloadConfig(),
            parallel=parallel,
            parameter_domains=registry,
        )
        strategy.setup(model, OptimizerConfig(), registry)
        output = model(torch.ones(1, 2))
        with strategy.microbatch_context(is_last_microbatch=True):
            torch.autograd.backward(output, torch.ones_like(output))
        strategy.finalize_gradients()
        assert all(bucket.local_gradient is not None for bucket in strategy.buckets)
    finally:
        parallel.close()
        runtime.close()


def test_offload_policy_rejects_incompatible_mode() -> None:
    runtime, parallel = _parallel_context()
    try:
        registry = _registered(nn.Linear(2, 2))
        with pytest.raises(ValueError, match="optimizer_state"):
            DDPStrategy(
                config=DataParallelConfig(),
                offload=OffloadConfig(optimizer_state=True),
                parallel=parallel,
                parameter_domains=registry,
            )
    finally:
        parallel.close()
        runtime.close()


def test_zero3_rejects_offload_when_the_replica_mesh_has_size_one() -> None:
    runtime, parallel = _parallel_context()
    try:
        model = nn.Linear(2, 2)
        registry = _registered(model)
        strategy = Zero3Strategy(
            config=DataParallelConfig(mode="zero3"),
            offload=OffloadConfig(zero3_params_and_grads=True),
            parallel=parallel,
            parameter_domains=registry,
        )
        with pytest.raises(ValueError, match="active replica mesh"):
            strategy.setup(model, OptimizerConfig(), registry)
    finally:
        parallel.close()
        runtime.close()


def _fsdp2_module():
    import torch.distributed.fsdp as fsdp

    if hasattr(fsdp, "fully_shard"):
        return fsdp
    from torch.distributed._composable import fsdp as composable_fsdp

    return composable_fsdp


def test_zero3_forwards_fsdp2_reshard_precision_and_pin_memory_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fsdp = _fsdp2_module()

    captured: dict[str, object] = {}
    sharded_modules: list[nn.Module] = []

    class _FakeCPUOffloadPolicy:
        def __init__(self, *, pin_memory: bool = True) -> None:
            self.pin_memory = pin_memory

    class _FakeMixedPrecisionPolicy:
        def __init__(self, *, reduce_dtype: torch.dtype) -> None:
            self.reduce_dtype = reduce_dtype

    def _fully_shard(module: nn.Module, **options: object) -> nn.Module:
        sharded_modules.append(module)
        captured.update(options)
        return module

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(fsdp, "CPUOffloadPolicy", _FakeCPUOffloadPolicy)
    monkeypatch.setattr(fsdp, "MixedPrecisionPolicy", _FakeMixedPrecisionPolicy)
    monkeypatch.setattr(fsdp, "fully_shard", _fully_shard)

    group = SimpleNamespace(
        size=2,
        rank=0,
        ranks=(0, 1),
        process_group=object(),
    )
    parallel = SimpleNamespace(
        group=lambda _key: group,
        mesh=lambda _key: object(),
    )
    model = nn.Linear(2, 2)
    registry = _registered(model)
    strategy = Zero3Strategy(
        config=DataParallelConfig(mode="zero3", reshard_after_forward=False),
        offload=OffloadConfig(
            zero3_params_and_grads=True,
            pin_memory=False,
        ),
        parallel=parallel,
        parameter_domains=registry,
    )
    strategy.configure_precision(
        SimpleNamespace(
            params="float32",
            compute="float32",
            grad_reduce="float32",
        )
    )

    assert strategy.setup(model, OptimizerConfig(), registry) is model
    assert strategy.uses_fsdp2
    assert sharded_modules == [model]
    assert captured["reshard_after_forward"] is False
    assert captured["mp_policy"].reduce_dtype is torch.float32
    assert captured["offload_policy"].pin_memory is False


def test_zero3_fully_shards_virtual_chunks_and_keeps_outer_checkpoint_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.distributed.checkpoint.state_dict as dcp_state_dict

    fsdp = _fsdp2_module()

    sharded_modules: list[nn.Module] = []

    def _fully_shard(module: nn.Module, **_options: object) -> nn.Module:
        sharded_modules.append(module)
        module.weight = nn.Parameter(module.weight.detach().clone())  # type: ignore[attr-defined]
        return module

    checkpoint_call: dict[str, object] = {}

    def _get_state_dict(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        checkpoint_call.update(model=model, optimizer=optimizer)
        return {"model": torch.ones(())}, {"optimizer": object()}

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(fsdp, "fully_shard", _fully_shard)
    monkeypatch.setattr(dcp_state_dict, "get_state_dict", _get_state_dict)

    group = SimpleNamespace(
        size=2,
        rank=0,
        ranks=(0, 1),
        process_group=object(),
    )
    parallel = SimpleNamespace(
        group=lambda _key: group,
        mesh=lambda _key: object(),
    )
    model = _VirtualShardContainer()
    original_parameters = tuple(model.parameters())
    registry = _registered(model)
    strategy = Zero3Strategy(
        config=DataParallelConfig(mode="zero3"),
        offload=OffloadConfig(),
        parallel=parallel,
        parameter_domains=registry,
    )

    assert strategy.setup(model, OptimizerConfig(), registry) is model
    assert strategy.model is model
    assert sharded_modules == list(model.chunks)
    assert [item.name for item in strategy._domain_parameters] == [
        "chunks.0.weight",
        "chunks.1.weight",
    ]
    assert all(
        item.parameter is dict(model.named_parameters())[item.name]
        for item in strategy._domain_parameters
    )
    assert not set(original_parameters).intersection(model.parameters())
    assert strategy.optimizer is not None
    assert {
        parameter
        for group_options in strategy.optimizer.param_groups
        for parameter in group_options["params"]
    } == set(model.parameters())

    first, second = model.chunks
    with strategy.microbatch_context(is_last_microbatch=False, unit=first):
        pass
    assert first.gradient_sync_calls == [(False, True), (True, True)]
    assert second.gradient_sync_calls == []

    with strategy.microbatch_context(is_last_microbatch=True, unit=second):
        pass
    assert first.gradient_sync_calls == [(False, True), (True, True)]
    assert second.gradient_sync_calls == [(True, True)]

    with (
        pytest.raises(ValueError, match="not one of the fully-sharded modules"),
        strategy.microbatch_context(
            is_last_microbatch=True,
            unit=nn.Linear(2, 2),
        ),
    ):
        pass

    state = strategy.distributed_checkpoint_state_dict()
    assert checkpoint_call == {"model": model, "optimizer": strategy.optimizer}
    assert set(state) == {"model", "optimizer"}


def test_zero3_size_one_leaves_virtual_chunk_container_unwrapped() -> None:
    runtime, parallel = _parallel_context()
    try:
        model = _VirtualShardContainer()
        registry = _registered(model)
        strategy = Zero3Strategy(
            config=DataParallelConfig(mode="zero3"),
            offload=OffloadConfig(),
            parallel=parallel,
            parameter_domains=registry,
        )

        assert strategy.setup(model, OptimizerConfig(), registry) is model
        assert strategy.model is model
        assert not strategy.uses_fsdp2
        with strategy.microbatch_context(
            is_last_microbatch=False,
            unit=model.chunks[0],
        ):
            pass
        assert all(not chunk.gradient_sync_calls for chunk in model.chunks)
    finally:
        parallel.close()
        runtime.close()


@pytest.mark.parametrize("strategy_type", [Zero1Strategy, Zero2Strategy])
def test_zero_reduces_gradients_in_configured_precision(strategy_type: type) -> None:
    runtime, parallel = _parallel_context()
    try:
        model = nn.Linear(3, 2).to(dtype=torch.bfloat16)
        registry = _registered(model)
        strategy = strategy_type(
            config=DataParallelConfig(mode=strategy_type.mode),
            offload=OffloadConfig(),
            parallel=parallel,
            parameter_domains=registry,
        )
        strategy.configure_precision(
            SimpleNamespace(
                params="bfloat16",
                compute="bfloat16",
                grad_reduce="float32",
            )
        )
        strategy.setup(model, OptimizerConfig(), registry)
        strategy.backward(model(torch.ones(2, 3, dtype=torch.bfloat16)).float().square().mean())
        strategy.finalize_gradients()

        assert all(
            bucket.local_gradient is not None
            and bucket.local_gradient.dtype is torch.float32
            for bucket in strategy.buckets
        )
    finally:
        parallel.close()
        runtime.close()


@pytest.mark.parametrize("strategy_type", [DDPStrategy, Zero1Strategy, Zero2Strategy])
def test_overlap_size_one_zero_grad_and_finalize_are_idempotent(strategy_type: type) -> None:
    runtime, parallel = _parallel_context()
    try:
        model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
        registry = _registered(model)
        strategy = strategy_type(
            config=DataParallelConfig(
                mode=strategy_type.mode,
                bucket_bytes=64,
                overlap_grad_reduce=True,
            ),
            offload=OffloadConfig(),
            parallel=parallel,
            parameter_domains=registry,
        )
        strategy.setup(model, OptimizerConfig(weight_decay=0.0), registry)

        for _ in range(2):
            strategy.zero_grad()
            with strategy.microbatch_context(is_last_microbatch=True):
                strategy.backward(model(torch.randn(2, 3)).square().mean())
            reducer = strategy.gradient_reducer
            assert reducer is not None
            assert reducer.start_count == len(reducer.buckets)
            assert not reducer.finalized

            strategy.finalize_gradients()
            strategy.finalize_gradients()
            assert reducer.finalized
            assert reducer.wait_count == len(reducer.buckets)
            strategy.optimizer_step()
    finally:
        parallel.close()
        runtime.close()
