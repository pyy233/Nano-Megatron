from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from nano_megatron.config import DistributedConfig
from nano_megatron.distributed import DistributedRuntime


class _RecordingDistributed:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.GroupMember = SimpleNamespace(NON_GROUP_MEMBER=object())

    @staticmethod
    def is_initialized() -> bool:
        return True

    def new_group(self, **kwargs):
        self.calls.append(kwargs)
        return object()

    def barrier(self, **kwargs) -> None:
        self.calls.append(kwargs)


class _LegacyDistributed(_RecordingDistributed):
    def new_group(self, *, ranks, backend, timeout):
        self.calls.append({"ranks": ranks, "backend": backend, "timeout": timeout})
        return object()


def _active_runtime(
    *, backend: str, device: torch.device
) -> tuple[DistributedRuntime, _RecordingDistributed]:
    runtime = DistributedRuntime(DistributedConfig(backend=backend, device=device.type))
    distributed = _RecordingDistributed()
    runtime._active = True
    runtime._dist = distributed
    runtime._rank = 0
    runtime._world_size = 2
    runtime._local_rank = 0
    runtime._backend = backend
    runtime._device_type = device.type
    runtime._device = device
    return runtime, distributed


def test_nccl_world_initialization_uses_current_device_without_eager_device_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "init_process_group", lambda **kwargs: calls.append(kwargs))
    runtime = DistributedRuntime(DistributedConfig(backend="nccl", device="cuda"))

    runtime.initialize()

    assert calls[0]["backend"] == "nccl"
    assert "device_id" not in calls[0]


def test_nccl_subgroup_creation_does_not_request_unsafe_eager_device_binding() -> None:
    runtime, distributed = _active_runtime(
        backend="nccl",
        device=torch.device("cuda", 3),
    )

    runtime.new_group((0, 1))

    assert "device_id" not in distributed.calls[0]


def test_nccl_subgroup_non_member_does_not_receive_a_cuda_device_id() -> None:
    runtime, distributed = _active_runtime(
        backend="nccl",
        device=torch.device("cuda", 0),
    )
    runtime._world_size = 3

    runtime.new_group((1, 2))

    assert "device_id" not in distributed.calls[0]


def test_gloo_subgroup_does_not_receive_a_cuda_device_id() -> None:
    runtime, distributed = _active_runtime(
        backend="gloo",
        device=torch.device("cpu"),
    )

    runtime.new_group((0, 1))

    assert "device_id" not in distributed.calls[0]


def test_nccl_barrier_uses_the_bound_local_cuda_device() -> None:
    runtime, distributed = _active_runtime(
        backend="nccl",
        device=torch.device("cuda", 3),
    )
    runtime._local_rank = 3

    runtime.barrier()

    assert distributed.calls[0] == {"device_ids": [3]}


def test_gloo_barrier_does_not_receive_cuda_device_ids() -> None:
    runtime, distributed = _active_runtime(
        backend="gloo",
        device=torch.device("cpu"),
    )

    runtime.barrier()

    assert distributed.calls[0] == {}


def test_legacy_nccl_subgroup_without_device_id_remains_usable() -> None:
    runtime, _ = _active_runtime(
        backend="nccl",
        device=torch.device("cuda", 0),
    )
    distributed = _LegacyDistributed()
    runtime._dist = distributed

    runtime.new_group((0, 1))

    assert "device_id" not in distributed.calls[0]
