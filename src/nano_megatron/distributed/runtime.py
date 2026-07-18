"""Explicit ownership of the default ``torch.distributed`` runtime.

There is deliberately no module-level singleton.  Objects which need rank or
group state receive a :class:`DistributedRuntime` (usually indirectly through
``ParallelContext``) in their constructor.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import timedelta
from types import ModuleType
from typing import Any

from nano_megatron.config.schema import DistributedConfig


class DistributedRuntimeError(RuntimeError):
    pass


class DistributedUnavailableError(DistributedRuntimeError):
    pass


class DistributedRuntime:
    """Initialize, expose, and conditionally destroy the default process group."""

    def __init__(self, config: DistributedConfig) -> None:
        if not isinstance(config, DistributedConfig):
            raise TypeError("DistributedRuntime requires a DistributedConfig")
        self.config = config
        self._active = False
        self._owns_default_group = False
        self._torch: ModuleType | None = None
        self._dist: ModuleType | None = None
        self._rank: int | None = None
        self._world_size: int | None = None
        self._local_rank: int | None = None
        self._backend: str | None = None
        self._device_type: str | None = None
        self._device: Any | None = None

    def initialize(self) -> DistributedRuntime:
        if self._active:
            return self

        rank = _environment_integer("RANK", default=0)
        world_size = _environment_integer("WORLD_SIZE", default=1, minimum=1)
        local_rank = _environment_integer("LOCAL_RANK", default=rank)

        try:
            import torch
            import torch.distributed as dist
        except ImportError as error:
            device_type = "cpu" if self.config.device == "auto" else self.config.device
            backend = "gloo" if self.config.backend == "auto" else self.config.backend
            if world_size != 1 or device_type != "cpu" or backend != "gloo":
                raise DistributedUnavailableError(
                    "PyTorch is required for multi-process or accelerator distributed runtime"
                ) from error
            # A dependency-free local runtime keeps topology/group-plan tests
            # usable before torch is installed.  It cannot create DeviceMesh or
            # perform collectives.
            self._rank = rank
            self._world_size = world_size
            self._local_rank = local_rank
            self._backend = backend
            self._device_type = device_type
            self._device = "cpu"
            self._active = True
            return self

        self._torch = torch
        self._dist = dist
        device_type = self._resolve_device_type(torch)
        backend = self._resolve_backend(device_type)

        if device_type == "cuda":
            if not torch.cuda.is_available():
                raise DistributedRuntimeError("distributed.device=cuda but CUDA is unavailable")
            device_count = torch.cuda.device_count()
            if not 0 <= local_rank < device_count:
                raise DistributedRuntimeError(
                    f"LOCAL_RANK={local_rank} is outside the {device_count} visible CUDA devices"
                )
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            try:
                device = torch.device(device_type)
            except RuntimeError as error:
                raise DistributedRuntimeError(
                    f"unsupported distributed device type {device_type!r}"
                ) from error

        if dist.is_initialized():
            actual_rank = dist.get_rank()
            actual_world_size = dist.get_world_size()
            actual_backend = str(dist.get_backend())
            if world_size != actual_world_size and "WORLD_SIZE" in os.environ:
                raise DistributedRuntimeError(
                    f"existing process group world size {actual_world_size} != "
                    f"WORLD_SIZE {world_size}"
                )
            if rank != actual_rank and "RANK" in os.environ:
                raise DistributedRuntimeError(
                    f"existing process group rank {actual_rank} != RANK {rank}"
                )
            if self.config.backend != "auto" and backend != actual_backend:
                raise DistributedRuntimeError(
                    f"existing process group backend {actual_backend!r} != requested {backend!r}"
                )
            rank = actual_rank
            world_size = actual_world_size
            backend = actual_backend
        elif self._should_initialize_default_group(world_size):
            kwargs: dict[str, Any] = {
                "backend": backend,
                "init_method": self.config.init_method,
                "timeout": timedelta(minutes=self.config.timeout_minutes),
                "rank": rank,
                "world_size": world_size,
            }
            # ``torch.cuda.set_device(local_rank)`` above is the authoritative
            # binding. Do not also pass WORLD device_id: Torch 2.6 implements
            # that eager path with NCCL communicator splitting, which can
            # corrupt later subgroups on real 8-GPU topologies.
            try:
                dist.init_process_group(**kwargs)
            except Exception as error:
                raise DistributedRuntimeError(
                    f"failed to initialize torch.distributed with backend={backend!r}, "
                    f"rank={rank}, world_size={world_size}: {error}"
                ) from error
            self._owns_default_group = True

        self._rank = rank
        self._world_size = world_size
        self._local_rank = local_rank
        self._backend = backend
        self._device_type = device_type
        self._device = device
        self._active = True
        return self

    def close(self) -> None:
        if not self._active:
            return
        if self._owns_default_group and self._dist is not None and self._dist.is_initialized():
            self._dist.destroy_process_group()
        self._active = False
        self._owns_default_group = False

    @property
    def is_initialized(self) -> bool:
        """Whether this explicit runtime object is active."""

        return self._active

    @property
    def process_group_initialized(self) -> bool:
        return bool(self._dist is not None and self._dist.is_initialized())

    @property
    def rank(self) -> int:
        self._require_active()
        assert self._rank is not None
        return self._rank

    @property
    def world_size(self) -> int:
        self._require_active()
        assert self._world_size is not None
        return self._world_size

    @property
    def local_rank(self) -> int:
        self._require_active()
        assert self._local_rank is not None
        return self._local_rank

    @property
    def backend(self) -> str:
        self._require_active()
        assert self._backend is not None
        return self._backend

    @property
    def device_type(self) -> str:
        self._require_active()
        assert self._device_type is not None
        return self._device_type

    @property
    def device(self) -> Any:
        self._require_active()
        return self._device

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        self._require_active()
        if self.world_size == 1:
            return
        assert self._dist is not None
        kwargs = {}
        if self.device_type == "cuda" and "nccl" in self.backend.lower():
            kwargs["device_ids"] = [self.local_rank]
        self._dist.barrier(**kwargs)

    def all_gather_object(self, value: Any) -> list[Any]:
        """Gather one Python value across the explicitly owned WORLD group."""

        self._require_active()
        if self.world_size == 1:
            return [value]
        assert self._dist is not None and self._dist.is_initialized()
        gathered: list[Any] = [None] * self.world_size
        self._dist.all_gather_object(gathered, value)
        return gathered

    def new_group(self, ranks: Iterable[int], *, backend: str | None = None) -> Any | None:
        self._require_active()
        concrete_ranks = tuple(ranks)
        if not concrete_ranks:
            raise ValueError("cannot create an empty process group")
        if len(set(concrete_ranks)) != len(concrete_ranks):
            raise ValueError("process-group ranks cannot contain duplicates")
        if any(rank < 0 or rank >= self.world_size for rank in concrete_ranks):
            raise ValueError(
                f"process-group ranks must be in [0, {self.world_size}), got {concrete_ranks}"
            )
        if not self.process_group_initialized:
            if self.world_size == 1 and concrete_ranks == (0,):
                return None
            raise DistributedRuntimeError(
                "a default torch.distributed process group is required to create subgroups"
            )
        assert self._dist is not None
        selected_backend = backend or self.backend
        kwargs: dict[str, Any] = {
            "ranks": list(concrete_ranks),
            "backend": selected_backend,
            "timeout": timedelta(minutes=self.config.timeout_minutes),
        }
        # Do not pass device_id to subgroup creation. PyTorch implements eager
        # NCCL subgroup initialization with ncclCommSplit when the default
        # WORLD group already owns a bound device. On real multi-dimensional
        # topologies this path can inspect/free a null communicator for ranks
        # outside a subgroup (Torch 2.4 warns; Torch 2.6/NCCL 2.21 may abort).
        # Pipeline/model communicators are instead initialized by explicit
        # ordered warmups before their first P2P exchange.
        return self._dist.new_group(**kwargs)

    def is_group_member(self, process_group: Any | None) -> bool:
        if process_group is None:
            return self.world_size == 1
        if self._dist is None:
            return False
        return process_group is not self._dist.GroupMember.NON_GROUP_MEMBER

    def destroy_group(self, process_group: Any | None) -> None:
        if process_group is None or self._dist is None or not self._dist.is_initialized():
            return
        if process_group is self._dist.GroupMember.NON_GROUP_MEMBER:
            return
        self._dist.destroy_process_group(process_group)

    def create_device_mesh(
        self,
        process_group: Any | None,
        *,
        ranks: tuple[int, ...],
        name: str,
    ) -> Any:
        self._require_active()
        if process_group is None:
            raise DistributedUnavailableError(
                "DeviceMesh requires an initialized torch.distributed process group; "
                "a size-one ZeRO-3 configuration should use its local degeneration path"
            )
        try:
            from torch.distributed.device_mesh import DeviceMesh
        except ImportError as error:
            raise DistributedUnavailableError(
                "this PyTorch build does not provide torch.distributed.device_mesh"
            ) from error
        if not hasattr(DeviceMesh, "from_group"):
            raise DistributedUnavailableError(
                "DeviceMesh.from_group is required so Nano-Megatron can reuse its explicit group"
            )
        try:
            return DeviceMesh.from_group(
                process_group,
                device_type=self.device_type,
                mesh=list(ranks),
                mesh_dim_names=(name,),
            )
        except TypeError:
            # Compatibility with PyTorch releases whose from_group omits the
            # explicit mesh argument.
            return DeviceMesh.from_group(
                process_group,
                device_type=self.device_type,
                mesh_dim_names=(name,),
            )

    def __enter__(self) -> DistributedRuntime:
        return self.initialize()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def _resolve_device_type(self, torch: ModuleType) -> str:
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        # Accept cuda:0 in config while local rank remains the authoritative
        # binding under torchrun.
        return self.config.device.split(":", 1)[0]

    def _resolve_backend(self, device_type: str) -> str:
        backend = self.config.backend
        if backend == "auto":
            return "nccl" if device_type == "cuda" else "gloo"
        if backend == "nccl" and device_type != "cuda":
            raise DistributedRuntimeError("NCCL backend requires distributed.device=cuda")
        return backend

    def _should_initialize_default_group(self, world_size: int) -> bool:
        if world_size > 1 or self.config.init_method != "env://":
            return True
        # Under torchrun these are present even for nproc-per-node=1.  In a
        # normal Python process, skip c10d initialization and use cheap local
        # groups instead of inventing a global rendezvous port.
        return "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ

    def _require_active(self) -> None:
        if not self._active:
            raise DistributedRuntimeError(
                "DistributedRuntime is not initialized; use it as a context manager "
                "or call initialize()"
            )


def _environment_integer(name: str, *, default: int, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise DistributedRuntimeError(f"{name} must be an integer, got {raw!r}") from error
    if value < minimum:
        raise DistributedRuntimeError(f"{name} must be >= {minimum}, got {value}")
    return value
