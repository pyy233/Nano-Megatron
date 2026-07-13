from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.distributed.checkpoint.state_dict import get_state_dict

from nano_megatron.checkpoint import CheckpointManager, CheckpointManifest
from nano_megatron.checkpoint import manager as checkpoint_manager_module
from nano_megatron.config import (
    CheckpointConfig,
    DataParallelConfig,
    DistributedConfig,
    OffloadConfig,
    OptimizerConfig,
    ParallelConfig,
)
from nano_megatron.data_parallel import build_data_parallel_strategy
from nano_megatron.distributed import DistributedRuntime
from nano_megatron.parallel import ParallelContext, ParameterDomain, ParameterDomainRegistry


class _FakeRNG:
    def __init__(self, value: int) -> None:
        self.value = value

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.value = int(state["value"])


def _assert_nested_close(actual: Any, expected: Any, *, ignore_metadata: bool = False) -> None:
    if isinstance(expected, Tensor):
        assert isinstance(actual, Tensor)
        torch.testing.assert_close(actual, expected)
        return
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping)
        expected_keys = set(expected)
        if ignore_metadata:
            expected_keys.discard("metadata")
        actual_keys = set(actual)
        if ignore_metadata:
            actual_keys.discard("metadata")
        assert actual_keys == expected_keys
        for key in expected_keys:
            _assert_nested_close(
                actual[key],
                expected[key],
                ignore_metadata=ignore_metadata,
            )
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_close(
                actual_item,
                expected_item,
                ignore_metadata=ignore_metadata,
            )
        return
    assert actual == expected


def _clone_local_state(value: Any) -> Any:
    if isinstance(value, Tensor):
        to_local = getattr(value, "to_local", None)
        local = to_local() if callable(to_local) else value
        return local.detach().clone()
    if isinstance(value, Mapping):
        return {key: _clone_local_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_local_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_local_state(item) for item in value)
    return copy.deepcopy(value)


def _assert_local_state_close(actual: Any, expected: Any) -> None:
    if isinstance(actual, Tensor):
        to_local = getattr(actual, "to_local", None)
        local = to_local() if callable(to_local) else actual
        torch.testing.assert_close(local, expected)
        return
    if isinstance(actual, Mapping):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_local_state_close(actual[key], expected[key])
        return
    if isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_local_state_close(actual_item, expected_item)
        return
    assert actual == expected


def _initialize_runtime(rank: int, world_size: int, rendezvous: str) -> DistributedRuntime:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    return DistributedRuntime(
        DistributedConfig(
            backend="gloo",
            device="cpu",
            init_method=f"file://{rendezvous}",
        )
    ).initialize()


def _take_optimizer_step(model: nn.Module, strategy: Any, scale: float) -> None:
    strategy.zero_grad()
    loss = sum(parameter.square().sum() * scale for parameter in model.parameters())
    strategy.backward(loss)
    strategy.optimizer_step()
    strategy.zero_grad()


def _take_ddp_step(model: nn.Module, strategy: Any, rank: int) -> None:
    strategy.zero_grad()
    with strategy.forward_microbatch_context(synchronize_gradients=True):
        output = model(torch.full((2, 3), float(rank + 1)))
    strategy.backward(output.square().mean())
    strategy.optimizer_step()
    strategy.zero_grad()


def _tp_pp_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    checkpoint_root: str,
    axis: str,
) -> None:
    runtime = _initialize_runtime(rank, world_size, rendezvous)
    config = (
        ParallelConfig(tensor=world_size, data=1)
        if axis == "tp"
        else ParallelConfig(pipeline=world_size, data=1)
    )
    parallel = ParallelContext.create(runtime, config)
    try:
        if axis == "tp":
            model: nn.Module = nn.Linear(3, 2)
        elif rank == 0:
            model = nn.Sequential(nn.Linear(2, 3))
        else:
            # Deliberately reuse the same state-dict keys with a different shape.
            model = nn.Sequential(nn.Linear(3, 1))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(rank + 1.25)

        registry = ParameterDomainRegistry()
        registry.register_module(
            model,
            ParameterDomain.DENSE,
            tensor_sharded=axis == "tp",
            tensor_shard_dim=0 if axis == "tp" else None,
        )
        strategy = build_data_parallel_strategy(
            DataParallelConfig(mode="ddp"),
            OffloadConfig(),
            parallel,
            registry,
        )
        strategy.setup(model, OptimizerConfig(lr=0.01), registry)
        _take_optimizer_step(model, strategy, scale=float(rank + 1))
        expected_model = copy.deepcopy(model.state_dict())
        expected_optimizer = copy.deepcopy(strategy.state_dict())

        manager = CheckpointManager(
            config=CheckpointConfig(directory=Path(checkpoint_root) / axis, save_interval=1),
            parallel=parallel,
        )
        torch.manual_seed(900 + rank)
        expected_generator = torch.Generator().set_state(torch.get_rng_state())
        expected_random = torch.rand(4, generator=expected_generator)
        rng = _FakeRNG(30 + rank)
        checkpoint = manager.save(
            5,
            model=model,
            data_parallel=strategy,
            trainer_state={"step": 5},
            rng=rng,
        )
        manifest = CheckpointManifest.read(checkpoint / "manifest.json")
        assert manifest.storage_backend == "rank_local_torch.save"
        assert len(manifest.rank_local_shards) == world_size
        assert len(manifest.rank_runtime_states) == world_size
        assert len({shard.state_file for shard in manifest.rank_local_shards}) == world_size
        assert not (checkpoint / "data_parallel.pt").exists()
        assert all(
            (checkpoint / shard.state_file).is_file()
            for shard in manifest.rank_local_shards
        )
        assert manifest.topology_compatibility is not None
        assert manifest.topology_compatibility["fixed_axes"] == ["tp", "pp", "cp"]
        if axis == "tp":
            assert all(
                metadata.sharded_axes
                for shard in manifest.rank_local_shards
                for metadata in shard.tensor_metadata.values()
            )
            for shard in manifest.rank_local_shards:
                for metadata in shard.tensor_metadata.values():
                    assert metadata.global_shape[0] == metadata.local_shape[0] * world_size
                    assert metadata.global_offset[0] == metadata.local_shape[0] * shard.tp

        _take_optimizer_step(model, strategy, scale=7.0)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(100.0)
        torch.manual_seed(5000 + rank)
        rng.value = -1
        trainer_state = manager.load(
            checkpoint,
            model=model,
            data_parallel=strategy,
            rng=rng,
        )
        _assert_nested_close(model.state_dict(), expected_model)
        _assert_nested_close(strategy.state_dict(), expected_optimizer)
        assert trainer_state == {"step": 5}
        torch.testing.assert_close(torch.rand(4), expected_random)
        assert rng.value == 30 + rank
    finally:
        parallel.close()
        runtime.close()


@pytest.mark.distributed
@pytest.mark.parametrize("axis", ["tp", "pp"])
def test_rank_local_checkpoint_preserves_tp_and_pp_state(tmp_path: Path, axis: str) -> None:
    rendezvous = tmp_path / f"{axis}.rendezvous"
    mp.spawn(
        _tp_pp_worker,
        args=(2, str(rendezvous), str(tmp_path), axis),
        nprocs=2,
        join=True,
    )


def _zero_worker(rank: int, world_size: int, rendezvous: str, checkpoint_root: str) -> None:
    runtime = _initialize_runtime(rank, world_size, rendezvous)
    parallel = ParallelContext.create(runtime, ParallelConfig(data=world_size))
    try:
        for mode in ("ddp", "zero1", "zero2"):
            torch.manual_seed(71)
            model = nn.Linear(3, 2)
            registry = ParameterDomainRegistry()
            registry.register_module(model, ParameterDomain.DENSE)
            strategy = build_data_parallel_strategy(
                DataParallelConfig(mode=mode, bucket_bytes=64),
                OffloadConfig(),
                parallel,
                registry,
            )
            wrapped = strategy.setup(model, OptimizerConfig(lr=0.01), registry)
            if mode == "ddp":
                _take_ddp_step(wrapped, strategy, rank)
            else:
                _take_optimizer_step(model, strategy, scale=float(rank + 1))
            expected_model = copy.deepcopy(model.state_dict())
            expected_optimizer = copy.deepcopy(strategy.state_dict())

            directory = Path(checkpoint_root) / mode
            manager = CheckpointManager(
                config=CheckpointConfig(directory=directory, save_interval=1),
                parallel=parallel,
            )
            checkpoint = manager.save(
                3,
                model=wrapped,
                data_parallel=strategy,
                trainer_state={"step": 3, "mode": mode},
            )
            manifest = CheckpointManifest.read(checkpoint / "manifest.json")
            assert len(manifest.rank_local_shards) == 1
            assert manifest.topology_compatibility is not None
            assert "dp" in manifest.topology_compatibility["resizable_axes"]
            if rank == 0:
                torch.save(
                    {"model": expected_model, "data_parallel": expected_optimizer},
                    directory / "expected.pt",
                )
            if mode == "ddp":
                _take_ddp_step(wrapped, strategy, rank)
            else:
                _take_optimizer_step(model, strategy, scale=9.0)
            manager.load(checkpoint, model=wrapped, data_parallel=strategy)
            _assert_nested_close(model.state_dict(), expected_model)
            _assert_nested_close(
                strategy.state_dict(),
                expected_optimizer,
                ignore_metadata=True,
            )
    finally:
        parallel.close()
        runtime.close()


@pytest.mark.distributed
def test_ddp_zero12_checkpoint_round_trip_and_dp_reshard(tmp_path: Path) -> None:
    rendezvous = tmp_path / "zero.rendezvous"
    mp.spawn(
        _zero_worker,
        args=(2, str(rendezvous), str(tmp_path)),
        nprocs=2,
        join=True,
    )

    for mode in ("ddp", "zero1", "zero2"):
        runtime = DistributedRuntime(
            DistributedConfig(backend="gloo", device="cpu")
        ).initialize()
        parallel = ParallelContext.create(runtime, ParallelConfig(data=1))
        try:
            model = nn.Linear(3, 2)
            registry = ParameterDomainRegistry()
            registry.register_module(model, ParameterDomain.DENSE)
            strategy = build_data_parallel_strategy(
                DataParallelConfig(mode=mode, bucket_bytes=64),
                OffloadConfig(),
                parallel,
                registry,
            )
            strategy.setup(model, OptimizerConfig(lr=0.01), registry)
            directory = tmp_path / mode
            manager = CheckpointManager(
                config=CheckpointConfig(directory=directory, save_interval=1),
                parallel=parallel,
            )
            trainer_state = manager.load(
                directory / "step_00000003",
                model=model,
                data_parallel=strategy,
            )
            expected = torch.load(directory / "expected.pt", weights_only=False)
            _assert_nested_close(model.state_dict(), expected["model"])
            _assert_nested_close(
                strategy.state_dict(),
                expected["data_parallel"],
                ignore_metadata=True,
            )
            assert trainer_state == {"step": 3, "mode": mode}
        finally:
            parallel.close()
            runtime.close()


def _async_rank_local_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    checkpoint_root: str,
) -> None:
    runtime = _initialize_runtime(rank, world_size, rendezvous)
    parallel = ParallelContext.create(runtime, ParallelConfig(data=world_size))
    manager: CheckpointManager | None = None
    try:
        torch.manual_seed(307)
        model = nn.Linear(3, 2)
        registry = ParameterDomainRegistry()
        registry.register_module(model, ParameterDomain.DENSE)
        strategy = build_data_parallel_strategy(
            DataParallelConfig(mode="zero2", bucket_bytes=64),
            OffloadConfig(),
            parallel,
            registry,
        )
        strategy.setup(model, OptimizerConfig(lr=0.01), registry)
        _take_optimizer_step(model, strategy, scale=float(rank + 1))
        expected_model = copy.deepcopy(model.state_dict())
        expected_optimizer = copy.deepcopy(strategy.state_dict())

        manager = CheckpointManager(
            config=CheckpointConfig(
                directory=Path(checkpoint_root) / "async_zero2",
                save_interval=1,
                async_save=True,
            ),
            parallel=parallel,
        )
        checkpoint = manager.save(
            8,
            model=model,
            data_parallel=strategy,
            trainer_state={"step": 8},
        )
        assert not (checkpoint / ".complete").exists()

        # Training collectives are allowed to continue while rank-local files are
        # written.  Final checkpoint collectives are deliberately deferred to flush().
        _take_optimizer_step(model, strategy, scale=float(rank + 9))
        manager.flush()
        assert (checkpoint / ".complete").is_file()
        assert manager.latest() == checkpoint

        manifest = CheckpointManifest.read(checkpoint / "manifest.json")
        assert manifest.storage_backend == "rank_local_torch.save"
        assert len(manifest.rank_local_shards) == 1
        assert len(manifest.rank_runtime_states) == world_size
        assert manager.load(
            checkpoint,
            model=model,
            data_parallel=strategy,
        ) == {"step": 8}
        _assert_nested_close(model.state_dict(), expected_model)
        _assert_nested_close(
            strategy.state_dict(),
            expected_optimizer,
            ignore_metadata=True,
        )
        manager.close()
    finally:
        if manager is not None:
            with suppress(BaseException):
                manager.close()
        parallel.close()
        runtime.close()


@pytest.mark.distributed
def test_async_rank_local_checkpoint_allows_training_before_flush(tmp_path: Path) -> None:
    rendezvous = tmp_path / "async.rendezvous"
    mp.spawn(
        _async_rank_local_worker,
        args=(2, str(rendezvous), str(tmp_path)),
        nprocs=2,
        join=True,
    )


def _async_failure_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    checkpoint_root: str,
) -> None:
    runtime = _initialize_runtime(rank, world_size, rendezvous)
    parallel = ParallelContext.create(runtime, ParallelConfig(data=world_size))
    manager: CheckpointManager | None = None
    original_write = checkpoint_manager_module._write_torch_payloads
    try:
        model = nn.Linear(3, 2)
        registry = ParameterDomainRegistry()
        registry.register_module(model, ParameterDomain.DENSE)
        strategy = build_data_parallel_strategy(
            DataParallelConfig(mode="zero2", bucket_bytes=64),
            OffloadConfig(),
            parallel,
            registry,
        )
        strategy.setup(model, OptimizerConfig(lr=0.01), registry)

        if rank == 1:

            def fail_write(payloads: list[tuple[Any, Path]]) -> None:
                del payloads
                raise OSError("rank-local disk failure")

            checkpoint_manager_module._write_torch_payloads = fail_write

        manager = CheckpointManager(
            config=CheckpointConfig(
                directory=Path(checkpoint_root) / "async_failure",
                save_interval=1,
                async_save=True,
            ),
            parallel=parallel,
        )
        checkpoint = manager.save(
            9,
            model=model,
            data_parallel=strategy,
            trainer_state={"step": 9},
        )
        with pytest.raises(RuntimeError, match="rank 1.*rank-local disk failure"):
            manager.flush()
        assert not (checkpoint / ".complete").exists()
        assert not (checkpoint / "manifest.json").exists()
        with pytest.raises(RuntimeError, match="rank-local disk failure"):
            manager.close()
    finally:
        checkpoint_manager_module._write_torch_payloads = original_write
        if manager is not None:
            with suppress(BaseException):
                manager.close()
        parallel.close()
        runtime.close()


@pytest.mark.distributed
def test_async_checkpoint_propagates_remote_write_failure(tmp_path: Path) -> None:
    rendezvous = tmp_path / "async-failure.rendezvous"
    mp.spawn(
        _async_failure_worker,
        args=(2, str(rendezvous), str(tmp_path)),
        nprocs=2,
        join=True,
    )


def _existing_target_preflight_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    checkpoint_root: str,
    async_save: bool,
) -> None:
    runtime = _initialize_runtime(rank, world_size, rendezvous)
    # PP2 makes every DP subgroup local, so this regression also proves that
    # preflight error propagation uses the runtime WORLD group.
    parallel = ParallelContext.create(runtime, ParallelConfig(pipeline=world_size))
    manager: CheckpointManager | None = None
    try:
        model = nn.Linear(3, 2)
        registry = ParameterDomainRegistry()
        registry.register_module(model, ParameterDomain.DENSE)
        strategy = build_data_parallel_strategy(
            DataParallelConfig(mode="zero2", bucket_bytes=64),
            OffloadConfig(),
            parallel,
            registry,
        )
        strategy.setup(model, OptimizerConfig(lr=0.01), registry)

        directory = Path(checkpoint_root)
        manager = CheckpointManager(
            config=CheckpointConfig(
                directory=directory,
                save_interval=1,
                async_save=async_save,
            ),
            parallel=parallel,
        )
        with pytest.raises(
            RuntimeError,
            match="checkpoint save preflight failed.*rank 0: FileExistsError",
        ):
            manager.save(
                1,
                model=model,
                data_parallel=strategy,
                trainer_state={"step": 1},
            )

        # A preflight error is synchronous: no pending work exists and it is not
        # sticky like a latent background write failure.  The manager remains usable.
        assert manager._pending is None
        assert manager._async_error is None
        manager.flush()
        recovered = manager.save(
            2,
            model=model,
            data_parallel=strategy,
            trainer_state={"step": 2},
        )
        manager.flush()
        assert (recovered / ".complete").is_file()
        manager.close()
    finally:
        if manager is not None:
            with suppress(BaseException):
                manager.close()
        parallel.close()
        runtime.close()


@pytest.mark.distributed
@pytest.mark.parametrize("async_save", [False, True], ids=["sync", "async"])
def test_existing_incomplete_target_preflight_error_reaches_every_rank(
    tmp_path: Path,
    async_save: bool,
) -> None:
    directory = tmp_path / ("async_preflight" if async_save else "sync_preflight")
    incomplete = directory / "step_00000001"
    incomplete.mkdir(parents=True)
    (incomplete / "partial-payload").write_text("incomplete\n")

    rendezvous = tmp_path / f"preflight-{async_save}.rendezvous"
    mp.spawn(
        _existing_target_preflight_worker,
        args=(2, str(rendezvous), str(directory), async_save),
        nprocs=2,
        join=True,
    )

    assert (incomplete / "partial-payload").read_text() == "incomplete\n"
    assert not (incomplete / ".complete").exists()
    assert (directory / "step_00000002" / ".complete").is_file()


def _dp_expand_rng_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    checkpoint: str,
    saved_seed: int,
) -> None:
    runtime = _initialize_runtime(rank, world_size, rendezvous)
    parallel = ParallelContext.create(runtime, ParallelConfig(data=world_size))
    try:
        model = nn.Linear(3, 2)
        registry = ParameterDomainRegistry()
        registry.register_module(model, ParameterDomain.DENSE)
        strategy = build_data_parallel_strategy(
            DataParallelConfig(mode="ddp"),
            OffloadConfig(),
            parallel,
            registry,
        )
        strategy.setup(model, OptimizerConfig(lr=0.01), registry)
        manager = CheckpointManager(
            config=CheckpointConfig(directory=Path(checkpoint).parent, save_interval=1),
            parallel=parallel,
        )

        torch.manual_seed(8000 + rank)
        initialized_torch_rng = torch.get_rng_state().clone()
        rng = _FakeRNG(100 + rank)
        if rank == 0:
            trainer_state = manager.load(
                checkpoint,
                model=model,
                data_parallel=strategy,
                rng=rng,
            )
            expected_torch_rng = torch.Generator().manual_seed(saved_seed).get_state()
            torch.testing.assert_close(torch.get_rng_state(), expected_torch_rng)
            assert rng.value == 55
        else:
            with pytest.warns(RuntimeWarning, match="newly introduced DP/EP replica"):
                trainer_state = manager.load(
                    checkpoint,
                    model=model,
                    data_parallel=strategy,
                    rng=rng,
                )
            torch.testing.assert_close(torch.get_rng_state(), initialized_torch_rng)
            assert rng.value == 100 + rank
        assert trainer_state == {"step": 4}
    finally:
        parallel.close()
        runtime.close()


@pytest.mark.distributed
def test_dp_expansion_keeps_new_replica_rng_state(tmp_path: Path) -> None:
    runtime = DistributedRuntime(
        DistributedConfig(backend="gloo", device="cpu")
    ).initialize()
    parallel = ParallelContext.create(runtime, ParallelConfig(data=1))
    try:
        model = nn.Linear(3, 2)
        registry = ParameterDomainRegistry()
        registry.register_module(model, ParameterDomain.DENSE)
        strategy = build_data_parallel_strategy(
            DataParallelConfig(mode="ddp"),
            OffloadConfig(),
            parallel,
            registry,
        )
        strategy.setup(model, OptimizerConfig(lr=0.01), registry)
        manager = CheckpointManager(
            config=CheckpointConfig(directory=tmp_path / "expand", save_interval=1),
            parallel=parallel,
        )
        saved_seed = 1207
        torch.manual_seed(saved_seed)
        checkpoint = manager.save(
            4,
            model=model,
            data_parallel=strategy,
            trainer_state={"step": 4},
            rng=_FakeRNG(55),
        )
    finally:
        parallel.close()
        runtime.close()

    rendezvous = tmp_path / "expand.rendezvous"
    mp.spawn(
        _dp_expand_rng_worker,
        args=(2, str(rendezvous), str(checkpoint), saved_seed),
        nprocs=2,
        join=True,
    )


def _build_zero3(parallel: ParallelContext) -> tuple[nn.Module, Any, OptimizerConfig]:
    torch.manual_seed(123)
    model = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 2))
    registry = ParameterDomainRegistry()
    registry.register_module(model, ParameterDomain.DENSE)
    optimizer_config = OptimizerConfig(lr=0.01)
    strategy = build_data_parallel_strategy(
        DataParallelConfig(mode="zero3"),
        OffloadConfig(),
        parallel,
        registry,
    )
    wrapped = strategy.setup(model, optimizer_config, registry)
    return wrapped, strategy, optimizer_config


def _zero3_reference(*, steps: int) -> tuple[nn.Module, torch.optim.AdamW]:
    torch.manual_seed(123)
    model = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 2))
    config = OptimizerConfig(lr=0.01)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        for replica_rank in range(2):
            value = float(replica_rank + 1 + step * 2)
            (model(torch.full((2, 4), value)).square().mean() / 2).backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def _zero3_save_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    checkpoint_root: str,
) -> None:
    runtime = _initialize_runtime(rank, world_size, rendezvous)
    parallel = ParallelContext.create(runtime, ParallelConfig(data=world_size))
    try:
        model, strategy, _ = _build_zero3(parallel)
        strategy.backward(model(torch.full((2, 4), float(rank + 1))).square().mean())
        strategy.optimizer_step()
        strategy.zero_grad()

        expected_model = []
        for parameter in model.parameters():
            full_tensor = getattr(parameter, "full_tensor", None)
            expected_model.append(
                (full_tensor() if callable(full_tensor) else parameter).detach().clone()
            )
        assert strategy.optimizer is not None
        _, optimizer_state = get_state_dict(model, strategy.optimizer)
        expected_optimizer = _clone_local_state(optimizer_state)

        manager = CheckpointManager(
            config=CheckpointConfig(
                directory=Path(checkpoint_root) / "zero3_dp2",
                save_interval=1,
                async_save=True,
            ),
            parallel=parallel,
        )
        checkpoint = manager.save(
            3,
            model=model,
            data_parallel=strategy,
            trainer_state={"step": 3},
        )
        assert not (checkpoint / ".complete").exists()
        manager.flush()
        manifest = CheckpointManifest.read(checkpoint / "manifest.json")
        assert manifest.storage_backend == "fsdp2_dcp"
        assert manifest.topology_compatibility is not None
        assert manifest.topology_compatibility["fixed_axes"] == ["tp", "pp", "cp", "ep"]
        assert manifest.topology_compatibility["resizable_axes"] == ["dp"]
        assert not manifest.topology_compatibility[
            "optimizer_layout_requires_same_parameter_order"
        ]
        assert manifest.topology_compatibility["shared_filesystem_required"]
        assert (checkpoint / "fsdp2_states" / "dense_tp0000_pp0000").is_dir()

        strategy.backward(model(torch.full((2, 4), float(rank + 7))).square().mean())
        strategy.optimizer_step()
        strategy.zero_grad()
        trainer_state = manager.load(
            checkpoint,
            model=model,
            data_parallel=strategy,
        )
        assert trainer_state == {"step": 3}
        for actual, expected in zip(model.parameters(), expected_model, strict=True):
            full_tensor = getattr(actual, "full_tensor", None)
            actual = full_tensor() if callable(full_tensor) else actual
            torch.testing.assert_close(actual, expected)
        _, restored_optimizer = get_state_dict(model, strategy.optimizer)
        _assert_local_state_close(restored_optimizer, expected_optimizer)
        manager.close()
    finally:
        parallel.close()
        runtime.close()


def _zero3_expand_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    checkpoint: str,
) -> None:
    runtime = _initialize_runtime(rank, world_size, rendezvous)
    parallel = ParallelContext.create(runtime, ParallelConfig(data=world_size))
    try:
        model, strategy, _ = _build_zero3(parallel)
        manager = CheckpointManager(
            config=CheckpointConfig(directory=Path(checkpoint).parent, save_interval=1),
            parallel=parallel,
        )
        if rank == 0:
            trainer_state = manager.load(
                checkpoint,
                model=model,
                data_parallel=strategy,
            )
        else:
            with pytest.warns(RuntimeWarning, match="newly introduced DP/EP replica"):
                trainer_state = manager.load(
                    checkpoint,
                    model=model,
                    data_parallel=strategy,
                )
        assert trainer_state == {"step": 4}

        strategy.zero_grad()
        strategy.backward(model(torch.full((2, 4), float(rank + 3))).square().mean())
        strategy.optimizer_step()
        reference, _ = _zero3_reference(steps=2)
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            full_tensor = getattr(actual, "full_tensor", None)
            actual = full_tensor() if callable(full_tensor) else actual
            torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-5)
    finally:
        parallel.close()
        runtime.close()


@pytest.mark.distributed
def test_zero3_dcp_round_trip_and_dp_reshard(tmp_path: Path) -> None:
    rendezvous = tmp_path / "zero3-save.rendezvous"
    mp.spawn(
        _zero3_save_worker,
        args=(2, str(rendezvous), str(tmp_path)),
        nprocs=2,
        join=True,
    )

    runtime = DistributedRuntime(
        DistributedConfig(backend="gloo", device="cpu")
    ).initialize()
    parallel = ParallelContext.create(runtime, ParallelConfig(data=1))
    try:
        model, strategy, _ = _build_zero3(parallel)
        manager = CheckpointManager(
            config=CheckpointConfig(directory=tmp_path / "zero3_dp1", save_interval=1),
            parallel=parallel,
        )
        trainer_state = manager.load(
            tmp_path / "zero3_dp2" / "step_00000003",
            model=model,
            data_parallel=strategy,
        )
        assert trainer_state == {"step": 3}
        reference, reference_optimizer = _zero3_reference(steps=1)
        assert strategy.optimizer is not None
        _assert_nested_close(model.state_dict(), reference.state_dict())
        _assert_nested_close(
            get_state_dict(model, strategy.optimizer),
            get_state_dict(reference, reference_optimizer),
        )
        checkpoint = manager.save(
            4,
            model=model,
            data_parallel=strategy,
            trainer_state={"step": 4},
        )
    finally:
        parallel.close()
        runtime.close()

    rendezvous = tmp_path / "zero3-load.rendezvous"
    mp.spawn(
        _zero3_expand_worker,
        args=(2, str(rendezvous), str(checkpoint)),
        nprocs=2,
        join=True,
    )


def _zero3_pp_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    checkpoint_root: str,
) -> None:
    runtime = _initialize_runtime(rank, world_size, rendezvous)
    parallel = ParallelContext.create(runtime, ParallelConfig(pipeline=2, data=2))
    try:
        if parallel.pp.rank == 0:
            model: nn.Module = nn.Sequential(nn.Linear(2, 3))
        else:
            # Same canonical keys, deliberately different stage-local shape.
            model = nn.Sequential(nn.Linear(3, 1))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(parallel.pp.rank + 0.25)
        registry = ParameterDomainRegistry()
        registry.register_module(model, ParameterDomain.DENSE)
        strategy = build_data_parallel_strategy(
            DataParallelConfig(mode="zero3"),
            OffloadConfig(),
            parallel,
            registry,
        )
        model = strategy.setup(model, OptimizerConfig(lr=0.01), registry)
        input_features = 2 if parallel.pp.rank == 0 else 3
        inputs = torch.full((2, input_features), float(parallel.pp.rank + 1))
        strategy.backward(model(inputs).square().mean())
        strategy.optimizer_step()
        strategy.zero_grad()

        expected_model = []
        for parameter in model.parameters():
            full_tensor = getattr(parameter, "full_tensor", None)
            expected_model.append(
                (full_tensor() if callable(full_tensor) else parameter).detach().clone()
            )
        assert strategy.optimizer is not None
        _, optimizer_state = get_state_dict(model, strategy.optimizer)
        expected_optimizer = _clone_local_state(optimizer_state)

        manager = CheckpointManager(
            config=CheckpointConfig(
                directory=Path(checkpoint_root) / "zero3_pp2_dp2",
                save_interval=1,
            ),
            parallel=parallel,
        )
        checkpoint = manager.save(
            6,
            model=model,
            data_parallel=strategy,
            trainer_state={"step": 6},
        )
        assert (checkpoint / "fsdp2_states" / "dense_tp0000_pp0000").is_dir()
        assert (checkpoint / "fsdp2_states" / "dense_tp0000_pp0001").is_dir()

        strategy.backward(model(inputs + 7).square().mean())
        strategy.optimizer_step()
        strategy.zero_grad()
        assert manager.load(
            checkpoint,
            model=model,
            data_parallel=strategy,
        ) == {"step": 6}
        for actual, expected in zip(model.parameters(), expected_model, strict=True):
            full_tensor = getattr(actual, "full_tensor", None)
            actual = full_tensor() if callable(full_tensor) else actual
            torch.testing.assert_close(actual, expected)
        _, restored_optimizer = get_state_dict(model, strategy.optimizer)
        _assert_local_state_close(restored_optimizer, expected_optimizer)
    finally:
        parallel.close()
        runtime.close()


@pytest.mark.distributed
def test_zero3_dcp_uses_separate_pp_stage_subdirectories(tmp_path: Path) -> None:
    rendezvous = tmp_path / "zero3-pp.rendezvous"
    mp.spawn(
        _zero3_pp_worker,
        args=(4, str(rendezvous), str(tmp_path)),
        nprocs=4,
        join=True,
    )
