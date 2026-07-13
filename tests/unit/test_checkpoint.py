from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nano_megatron.checkpoint import (
    CheckpointManager,
    CheckpointManifest,
    ShardedState,
    ShardMetadata,
)
from nano_megatron.checkpoint import manager as checkpoint_manager_module
from nano_megatron.checkpoint.manifest import validate_topology
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
from nano_megatron.parallel import (
    ParallelAxis,
    ParallelContext,
    ParameterDomain,
    ParameterDomainRegistry,
)


class _FakeRNG:
    def __init__(self, value: int) -> None:
        self.value = value

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.value = int(state["value"])


def _setup(
    tmp_path: Path,
    *,
    async_save: bool = False,
    keep_last: int = 2,
    run_config: object | None = None,
):
    runtime = DistributedRuntime(DistributedConfig(backend="gloo", device="cpu")).initialize()
    parallel = ParallelContext.create(runtime, ParallelConfig(data=1))
    model = nn.Sequential(nn.Linear(3, 4), nn.GELU(), nn.Linear(4, 2))
    registry = ParameterDomainRegistry()
    registry.register_module(model, ParameterDomain.DENSE)
    strategy = build_data_parallel_strategy(
        DataParallelConfig(mode="zero2", bucket_bytes=128),
        OffloadConfig(),
        parallel,
        registry,
    )
    strategy.setup(model, OptimizerConfig(lr=0.01), registry)
    manager = CheckpointManager(
        config=CheckpointConfig(
            directory=tmp_path,
            save_interval=1,
            async_save=async_save,
            keep_last=keep_last,
        ),
        parallel=parallel,
        run_config=run_config,
    )
    return runtime, parallel, model, strategy, manager


@pytest.mark.parametrize("async_save", [False, True], ids=["sync", "async"])
def test_checkpoint_manifest_uses_actual_model_virtual_pipeline_layout(
    tmp_path: Path,
    async_save: bool,
) -> None:
    runtime, parallel, model, strategy, manager = _setup(
        tmp_path,
        async_save=async_save,
    )
    model.layout = SimpleNamespace(virtual_stages_per_rank=2)
    try:
        checkpoint = manager.save(
            1,
            model=model,
            data_parallel=strategy,
            trainer_state={"step": 1},
        )
        manager.flush()

        manifest = CheckpointManifest.read(checkpoint / "manifest.json")
        assert manifest.virtual_stages_per_rank == 2
        assert manifest.topology_compatibility is not None
        assert manifest.topology_compatibility["virtual_stages_per_rank"] == 2
        assert manager.load(
            checkpoint,
            model=model,
            data_parallel=strategy,
        ) == {"step": 1}
    finally:
        manager.close()
        parallel.close()
        runtime.close()


@pytest.mark.parametrize("async_save", [False, True], ids=["sync", "async"])
def test_checkpoint_rejects_run_config_virtual_pipeline_mismatch(
    tmp_path: Path,
    async_save: bool,
) -> None:
    run_config = SimpleNamespace(
        pipeline=SimpleNamespace(virtual_stages_per_rank=1)
    )
    runtime, parallel, model, strategy, manager = _setup(
        tmp_path,
        async_save=async_save,
        run_config=run_config,
    )
    model.layout = SimpleNamespace(virtual_stages_per_rank=2)
    try:
        with pytest.raises(ValueError, match="does not match model.layout: 1 != 2"):
            manager.save(
                1,
                model=model,
                data_parallel=strategy,
                trainer_state={"step": 1},
            )
        assert not (tmp_path / "step_00000001").exists()
        manager.flush()
    finally:
        manager.close()
        parallel.close()
        runtime.close()


def test_shard_metadata_round_trip() -> None:
    metadata = ShardMetadata(
        logical_key="layers.0.weight",
        global_shape=(8, 4),
        local_shape=(4, 4),
        global_offset=(4, 0),
        sharded_axes=(ParallelAxis.TP,),
        parameter_domain=ParameterDomain.DENSE,
        replica_coordinate=(0, 0, 0),
    )
    assert ShardMetadata.from_dict(metadata.to_dict()) == metadata


def test_sharded_state_uses_registered_tp_shard_dimension() -> None:
    model = nn.Linear(4, 3, bias=False)
    registry = ParameterDomainRegistry()
    registry.register(
        model.weight,
        ParameterDomain.DENSE,
        tensor_sharded=True,
        tensor_shard_dim=1,
    )
    parallel = SimpleNamespace(
        tp=SimpleNamespace(rank=1, size=2),
        coordinate=SimpleNamespace(dp=0, ep=0, cp=0),
    )
    state = ShardedState.from_model(
        model,
        parameter_domains=registry,
        parallel=parallel,
    )
    metadata = state.metadata["weight"]
    assert metadata.local_shape == (3, 4)
    assert metadata.global_shape == (3, 8)
    assert metadata.global_offset == (0, 4)
    assert metadata.sharded_axes == (ParallelAxis.TP,)


def test_checkpoint_round_trip_restores_model_optimizer_and_trainer_state(
    tmp_path: Path,
) -> None:
    runtime, parallel, model, strategy, manager = _setup(tmp_path)
    try:
        strategy.backward(model(torch.randn(5, 3)).square().mean())
        strategy.optimizer_step()
        strategy.zero_grad()
        expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
        torch.manual_seed(1234)
        expected_generator = torch.Generator().set_state(torch.get_rng_state())
        expected_random = torch.rand(4, generator=expected_generator)
        rng = _FakeRNG(17)
        checkpoint = manager.save(
            3,
            model=model,
            data_parallel=strategy,
            trainer_state={"step": 3, "consumed_tokens": 120},
            rng=rng,
        )
        assert manager.latest() == checkpoint
        manifest = CheckpointManifest.read(checkpoint / "manifest.json")
        assert manifest.step == 3
        assert manifest.schema_version == 2
        assert manifest.data_parallel_mode == "zero2"
        assert manifest.tensor_metadata
        assert len(manifest.rank_runtime_states) == 1
        assert manifest.group_plan is not None
        assert manifest.group_plan["specs"][0]["varying_axes"] == ["tp"]
        assert manifest.topology_compatibility == {
            "fixed_axes": ["tp", "pp", "cp"],
            "model_shard_key": ["tp", "pp", "ep"],
            "optimizer_layout_requires_same_parameter_order": True,
            "rank_order_may_change": True,
            "resizable_axes": ["dp", "ep"],
            "new_replica_rng_policy": "keep_initialized_state_with_warning",
            "shared_filesystem_required": False,
            "rng_exact_for_saved_coordinates": True,
            "virtual_stages_per_rank": 1,
            "zero_bucket_boundaries_must_match": True,
        }
        assert manifest.virtual_stages_per_rank == 1
        saved_sizes = {**manifest.parallel_sizes, "dp": 2}
        validate_topology(replace(manifest, parallel_sizes=saved_sizes), parallel)
        validate_topology(
            replace(
                manifest,
                data_parallel_mode="zero3",
                parallel_sizes=saved_sizes,
            ),
            parallel,
        )
        with pytest.raises(ValueError, match="cannot change EP size"):
            validate_topology(
                replace(
                    manifest,
                    data_parallel_mode="zero3",
                    parallel_sizes={**manifest.parallel_sizes, "ep": 2},
                ),
                parallel,
            )

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        torch.manual_seed(9999)
        rng.value = -1
        replace(manifest, virtual_stages_per_rank=2).write(
            checkpoint / "manifest.json"
        )
        with pytest.raises(ValueError, match="virtual_stages_per_rank"):
            manager.load(checkpoint, model=model, data_parallel=strategy, rng=rng)
        manifest.write(checkpoint / "manifest.json")
        with pytest.raises(ValueError, match="no RNG object"):
            manager.load(checkpoint, model=model, data_parallel=strategy)
        trainer_state = manager.load(
            checkpoint,
            model=model,
            data_parallel=strategy,
            rng=rng,
        )
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[key])
        assert trainer_state == {"step": 3, "consumed_tokens": 120}
        assert strategy.state_dict()["step"] == 1
        torch.testing.assert_close(torch.rand(4), expected_random)
        assert rng.value == 17
    finally:
        parallel.close()
        runtime.close()


def test_checkpoint_retention(tmp_path: Path) -> None:
    runtime, parallel, model, strategy, manager = _setup(tmp_path)
    try:
        for step in range(3):
            manager.save(
                step,
                model=model,
                data_parallel=strategy,
                trainer_state={"step": step},
            )
        assert [path.name for path in sorted(tmp_path.glob("step_*"))] == [
            "step_00000001",
            "step_00000002",
        ]
    finally:
        parallel.close()
        runtime.close()


def test_async_checkpoint_returns_before_io_and_uses_captured_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, parallel, model, strategy, manager = _setup(tmp_path, async_save=True)
    io_started = Event()
    allow_io = Event()
    original_write = checkpoint_manager_module._write_torch_payloads

    def blocking_write(payloads: list[tuple[object, Path]]) -> None:
        io_started.set()
        if not allow_io.wait(timeout=10):
            raise TimeoutError("test did not release async checkpoint I/O")
        original_write(payloads)

    monkeypatch.setattr(checkpoint_manager_module, "_write_torch_payloads", blocking_write)
    try:
        expected_model = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        trainer_state = {"step": 7, "nested": {"tokens": 128}}
        rng = _FakeRNG(41)
        checkpoint = manager.save(
            7,
            model=model,
            data_parallel=strategy,
            trainer_state=trainer_state,
            rng=rng,
        )

        assert io_started.wait(timeout=10)
        assert not (checkpoint / ".complete").exists()
        assert not (checkpoint / "manifest.json").exists()
        assert manager.latest() is None
        with pytest.raises(ValueError, match="incomplete"):
            manager.load(checkpoint, model=model, data_parallel=strategy, rng=rng)

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(100.0)
        trainer_state["step"] = 99
        trainer_state["nested"]["tokens"] = 999
        rng.value = -1

        allow_io.set()
        manager.flush()
        assert (checkpoint / ".complete").is_file()
        assert manager.latest() == checkpoint

        restored = manager.load(
            checkpoint,
            model=model,
            data_parallel=strategy,
            rng=rng,
        )
        assert restored == {"step": 7, "nested": {"tokens": 128}}
        assert rng.value == 41
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected_model[key])

        manager.flush()
        manager.close()
        manager.close()
    finally:
        allow_io.set()
        with suppress(BaseException):
            manager.close()
        parallel.close()
        runtime.close()


def test_async_checkpoint_finishes_previous_save_before_starting_next(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, parallel, model, strategy, manager = _setup(tmp_path, async_save=True)
    io_started = Event()
    allow_io = Event()
    original_write = checkpoint_manager_module._write_torch_payloads

    def blocking_write(payloads: list[tuple[object, Path]]) -> None:
        io_started.set()
        if not allow_io.wait(timeout=10):
            raise TimeoutError("test did not release async checkpoint I/O")
        original_write(payloads)

    monkeypatch.setattr(checkpoint_manager_module, "_write_torch_payloads", blocking_write)
    try:
        first = manager.save(
            1,
            model=model,
            data_parallel=strategy,
            trainer_state={"step": 1},
        )
        assert io_started.wait(timeout=10)

        with ThreadPoolExecutor(max_workers=1) as caller:
            next_save = caller.submit(
                manager.save,
                2,
                model=model,
                data_parallel=strategy,
                trainer_state={"step": 2},
            )
            assert not next_save.done()
            allow_io.set()
            second = next_save.result(timeout=10)

        assert (first / ".complete").is_file()
        assert not (second / ".complete").exists()
        manager.flush()
        assert (second / ".complete").is_file()
    finally:
        allow_io.set()
        with suppress(BaseException):
            manager.close()
        parallel.close()
        runtime.close()


def test_async_checkpoint_failure_is_sticky_and_never_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, parallel, model, strategy, manager = _setup(tmp_path, async_save=True)

    def fail_write(payloads: list[tuple[object, Path]]) -> None:
        del payloads
        raise OSError("simulated disk failure")

    monkeypatch.setattr(checkpoint_manager_module, "_write_torch_payloads", fail_write)
    checkpoint = manager.save(
        3,
        model=model,
        data_parallel=strategy,
        trainer_state={"step": 3},
    )
    try:
        with pytest.raises(RuntimeError, match="simulated disk failure"):
            manager.flush()
        assert not (checkpoint / ".complete").exists()
        assert not (checkpoint / "manifest.json").exists()
        assert manager.latest() is None

        with pytest.raises(RuntimeError, match="simulated disk failure"):
            manager.save(
                4,
                model=model,
                data_parallel=strategy,
                trainer_state={"step": 4},
            )
        assert not (tmp_path / "step_00000004").exists()
        with pytest.raises(RuntimeError, match="simulated disk failure"):
            manager.close()
        with pytest.raises(RuntimeError, match="simulated disk failure"):
            manager.close()
    finally:
        with suppress(BaseException):
            manager.close()
        parallel.close()
        runtime.close()


def test_async_retention_ignores_incomplete_directories(tmp_path: Path) -> None:
    incomplete = tmp_path / "step_00000000"
    incomplete.mkdir()
    runtime, parallel, model, strategy, manager = _setup(
        tmp_path,
        async_save=True,
        keep_last=1,
    )
    try:
        first = manager.save(
            1,
            model=model,
            data_parallel=strategy,
            trainer_state={"step": 1},
        )
        manager.flush()
        second = manager.save(
            2,
            model=model,
            data_parallel=strategy,
            trainer_state={"step": 2},
        )
        manager.flush()

        assert incomplete.is_dir()
        assert not first.exists()
        assert second.is_dir()
        assert manager.latest() == second
    finally:
        manager.close()
        parallel.close()
        runtime.close()


def test_async_zero3_builds_cpu_checkpoint_groups_for_nccl(
    tmp_path: Path,
) -> None:
    created: list[tuple[tuple[int, ...], str, object]] = []
    destroyed: list[object] = []

    def new_group(ranks: tuple[int, ...], *, backend: str) -> object:
        process_group = object()
        created.append((ranks, backend, process_group))
        return process_group

    runtime = SimpleNamespace(
        world_size=4,
        new_group=new_group,
        destroy_group=destroyed.append,
    )
    families = ((0, 1), (2, 3))
    parallel = SimpleNamespace(
        rank=0,
        runtime=runtime,
        group_family=lambda key: families,
    )
    manager = CheckpointManager(
        config=CheckpointConfig(directory=tmp_path, async_save=True),
        parallel=parallel,
    )
    nccl_process_group = SimpleNamespace(_device_types=(torch.device("cuda"),))
    selected = manager._async_dcp_process_group(
        SimpleNamespace(
            process_group=nccl_process_group,
            backend="nccl",
            ranks=(0, 1),
        )
    )

    assert [(ranks, backend) for ranks, backend, _ in created] == [
        ((0, 1), "gloo"),
        ((2, 3), "gloo"),
    ]
    assert selected is created[0][2]
    manager.close()
    assert destroyed == [created[1][2], created[0][2]]
