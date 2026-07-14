from __future__ import annotations

from types import SimpleNamespace

import torch

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


def test_nccl_subgroup_is_eagerly_bound_to_the_local_cuda_device() -> None:
    runtime, distributed = _active_runtime(
        backend="nccl",
        device=torch.device("cuda", 3),
    )

    runtime.new_group((0, 1))

    assert distributed.calls[0]["device_id"] == torch.device("cuda", 3)


def test_gloo_subgroup_does_not_receive_a_cuda_device_id() -> None:
    runtime, distributed = _active_runtime(
        backend="gloo",
        device=torch.device("cpu"),
    )

    runtime.new_group((0, 1))

    assert "device_id" not in distributed.calls[0]


def test_legacy_nccl_subgroup_without_device_id_remains_usable() -> None:
    runtime, _ = _active_runtime(
        backend="nccl",
        device=torch.device("cuda", 0),
    )
    distributed = _LegacyDistributed()
    runtime._dist = distributed

    runtime.new_group((0, 1))

    assert "device_id" not in distributed.calls[0]
