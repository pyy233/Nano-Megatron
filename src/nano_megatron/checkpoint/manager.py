from __future__ import annotations

import copy
import shutil
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from nano_megatron.data_parallel import DataParallelStrategy

from .manifest import (
    CheckpointManifest,
    RankLocalShard,
    RankRuntimeState,
    make_manifest,
    validate_topology,
)
from .mapping import ShardedState

TrainerState = dict[str, Any]


class CheckpointManager:
    """DCP-backed model checkpointing with a portable single-process fallback."""

    def __init__(
        self,
        *,
        config: object,
        parallel: object,
        run_config: object | None = None,
    ) -> None:
        self.config = config
        self.parallel = parallel
        self.run_config = run_config
        self.directory = Path(getattr(config, "directory", "checkpoints"))
        self.keep_last = int(getattr(config, "keep_last", 2))
        if self.keep_last < 0:
            raise ValueError("checkpoint.keep_last cannot be negative")
        if bool(getattr(config, "async_save", False)):
            raise NotImplementedError(
                "asynchronous checkpoint finalization is not implemented in phase one"
            )

    def save(
        self,
        step: int,
        *,
        model: nn.Module,
        data_parallel: DataParallelStrategy,
        trainer_state: Mapping[str, Any],
        path: Path | str | None = None,
        rng: object | None = None,
    ) -> Path:
        if step < 0:
            raise ValueError("checkpoint step cannot be negative")
        registry = data_parallel.parameter_domains
        if registry is None:
            raise ValueError("checkpointing requires an explicit ParameterDomainRegistry")

        target = Path(path) if path is not None else self.directory / f"step_{step:08d}"
        if self._is_coordinator():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.mkdir(parents=False, exist_ok=False)
        self._barrier()
        captured_runtime_state = _capture_runtime_state(rng)

        mode = str(getattr(data_parallel, "mode", "unknown"))
        rank_local_shards: tuple[RankLocalShard, ...] = ()
        manifest_metadata = {}
        optimizer_state: Mapping[str, Any] | None = None
        if mode == "zero3":
            self._save_zero3_state(target, data_parallel)
            backend = "fsdp2_dcp"
        else:
            canonical_model = _canonical_model(model)
            sharded_model = ShardedState.from_model(
                canonical_model,
                parameter_domains=registry,
                parallel=self.parallel,
            )
            manifest_metadata = sharded_model.metadata
            if self._distributed_world_size() == 1:
                backend = self._save_model(target, sharded_model)
                optimizer_state = data_parallel.state_dict()
            else:
                # ZeRO-1/2 transiently gather each logical bucket in state_dict().  This
                # keeps the file independent from the current DP shard ownership.
                optimizer_state = data_parallel.state_dict()
                local_shard = self._write_rank_local_state(
                    target,
                    model=canonical_model,
                    sharded_model=sharded_model,
                    data_parallel_state=optimizer_state,
                )
                rank_local_shards = self._collect_rank_local_shards(local_shard)
                backend = "rank_local_torch.save"
                manifest_metadata = {}

        local_runtime_state = self._write_rank_runtime_state(
            target,
            captured_runtime_state,
        )
        rank_runtime_states = self._collect_rank_runtime_states(local_runtime_state)

        if self._is_coordinator():
            if optimizer_state is not None and backend != "rank_local_torch.save":
                torch.save(dict(optimizer_state), target / "data_parallel.pt")
            torch.save(dict(trainer_state), target / "trainer_state.pt")
            manifest = make_manifest(
                step=step,
                storage_backend=backend,
                parallel=self.parallel,
                metadata=manifest_metadata,
                data_parallel_mode=mode,
                rank_local_shards=rank_local_shards,
                rank_runtime_states=rank_runtime_states,
                run_config=self.run_config,
            )
            manifest.write(target / "manifest.json")
            (target / ".complete").write_text("complete\n")
            if target.parent.resolve() == self.directory.resolve():
                self._remove_old_checkpoints(exclude=target)
        self._barrier()
        return target

    def load(
        self,
        path: Path | str,
        *,
        model: nn.Module,
        data_parallel: DataParallelStrategy,
        rng: object | None = None,
    ) -> TrainerState:
        path = Path(path)
        if not (path / ".complete").is_file():
            raise ValueError(f"checkpoint is incomplete or missing: {path}")
        manifest = CheckpointManifest.read(path / "manifest.json")
        validate_topology(manifest, self.parallel)
        current_mode = str(getattr(data_parallel, "mode", "unknown"))
        if (
            manifest.data_parallel_mode != "unknown"
            and manifest.data_parallel_mode != current_mode
        ):
            raise ValueError(
                "checkpoint data-parallel mode does not match the current strategy: "
                f"{manifest.data_parallel_mode!r} != {current_mode!r}"
            )
        if manifest.storage_backend == "fsdp2_dcp":
            self._load_zero3_state(path, data_parallel)
        elif manifest.storage_backend == "rank_local_torch.save":
            canonical_model = _canonical_model(model)
            self._load_rank_local_state(
                path,
                manifest,
                model=canonical_model,
                data_parallel=data_parallel,
            )
        else:
            canonical_model = _canonical_model(model)
            self._load_model(path, manifest.storage_backend, canonical_model)
            optimizer_state = _torch_load(path / "data_parallel.pt", map_location="cpu")
            if not isinstance(optimizer_state, Mapping):
                raise ValueError("data_parallel.pt does not contain a mapping")
            data_parallel.load_state_dict(optimizer_state)
        self._load_rank_runtime_state(path, manifest, rng=rng)

        trainer_state = _torch_load(path / "trainer_state.pt", map_location="cpu")
        if not isinstance(trainer_state, Mapping):
            raise ValueError("trainer_state.pt does not contain a mapping")
        return dict(trainer_state)

    def latest(self) -> Path | None:
        if not self.directory.exists():
            return None
        complete = sorted(
            path
            for path in self.directory.glob("step_*")
            if path.is_dir() and (path / ".complete").is_file()
        )
        return complete[-1] if complete else None

    def _zero3_state_path(
        self,
        target: Path,
        data_parallel: DataParallelStrategy,
    ) -> Path:
        domain = getattr(data_parallel, "checkpoint_domain", None)
        domain_name = str(getattr(domain, "value", domain))
        if domain_name not in {"dense", "expert"}:
            raise RuntimeError(f"invalid ZeRO-3 checkpoint domain: {domain_name!r}")
        coordinate = self.parallel.coordinate
        shard_id = f"{domain_name}_tp{coordinate.tp:04d}_pp{coordinate.pp:04d}"
        if domain_name == "expert":
            shard_id += f"_ep{coordinate.ep:04d}"
        return target / "fsdp2_states" / shard_id

    def _zero3_state_group(self, data_parallel: DataParallelStrategy) -> Any:
        group = getattr(data_parallel, "checkpoint_group", None)
        if group is None:
            raise RuntimeError("ZeRO-3 strategy does not expose a checkpoint replica group")
        return group

    def _save_zero3_state(
        self,
        target: Path,
        data_parallel: DataParallelStrategy,
    ) -> None:
        try:
            import torch.distributed.checkpoint as dcp
        except ImportError as error:
            raise RuntimeError("ZeRO-3 checkpointing requires PyTorch DCP") from error
        state_dict = getattr(data_parallel, "distributed_checkpoint_state_dict", None)
        if not callable(state_dict):
            raise RuntimeError("ZeRO-3 strategy does not expose a DCP state dict")
        group = self._zero3_state_group(data_parallel)
        dcp.save(
            state_dict(),
            checkpoint_id=self._zero3_state_path(target, data_parallel),
            process_group=group.process_group,
            no_dist=group.process_group is None,
        )

    def _load_zero3_state(
        self,
        path: Path,
        data_parallel: DataParallelStrategy,
    ) -> None:
        try:
            import torch.distributed.checkpoint as dcp
        except ImportError as error:
            raise RuntimeError("ZeRO-3 checkpointing requires PyTorch DCP") from error
        state_dict = getattr(data_parallel, "distributed_checkpoint_state_dict", None)
        load_state_dict = getattr(
            data_parallel,
            "load_distributed_checkpoint_state_dict",
            None,
        )
        if not callable(state_dict) or not callable(load_state_dict):
            raise RuntimeError("ZeRO-3 strategy does not expose DCP checkpoint methods")
        group = self._zero3_state_group(data_parallel)
        state = state_dict()
        dcp.load(
            state,
            checkpoint_id=self._zero3_state_path(path, data_parallel),
            process_group=group.process_group,
            no_dist=group.process_group is None,
        )
        load_state_dict(state)

    def _save_model(self, target: Path, state: ShardedState) -> str:
        try:
            import torch.distributed.checkpoint as dcp
        except ImportError:
            dcp = None  # type: ignore[assignment]

        if dcp is not None:
            try:
                dcp.save(
                    state_dict={"model": state.state},
                    checkpoint_id=target / "dcp",
                )
                return "torch.distributed.checkpoint"
            except Exception:
                if self._distributed_world_size() > 1:
                    raise
        if self._distributed_world_size() > 1:
            raise RuntimeError("distributed checkpoint save requires PyTorch DCP")
        torch.save(state.state, target / "model.pt")
        return "torch.save"

    def _write_rank_local_state(
        self,
        target: Path,
        *,
        model: nn.Module,
        sharded_model: ShardedState,
        data_parallel_state: Mapping[str, Any],
    ) -> RankLocalShard | None:
        coordinate = self.parallel.coordinate
        # DP and CP are replica axes for phase-one GPT parameters. One writer
        # per (TP, PP, EP) coordinate prevents both duplicate files and PP/TP key collisions.
        if coordinate.dp != 0 or coordinate.cp != 0:
            return None
        shard_id = f"tp{coordinate.tp:04d}_pp{coordinate.pp:04d}_ep{coordinate.ep:04d}"
        relative_path = Path("rank_states") / f"{shard_id}.pt"
        state_path = target / relative_path
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": sharded_model.state,
                "data_parallel": dict(data_parallel_state),
            },
            state_path,
        )
        partition = getattr(model, "partition", None)
        return RankLocalShard(
            shard_id=shard_id,
            writer_rank=int(self.parallel.rank),
            tp=int(coordinate.tp),
            pp=int(coordinate.pp),
            ep=int(coordinate.ep),
            cp=int(coordinate.cp),
            dp=int(coordinate.dp),
            state_file=relative_path.as_posix(),
            tensor_metadata=sharded_model.metadata,
            layer_start=None if partition is None else int(partition.start_layer),
            layer_end=None if partition is None else int(partition.end_layer),
        )

    def _collect_rank_local_shards(
        self,
        local_shard: RankLocalShard | None,
    ) -> tuple[RankLocalShard, ...]:
        gathered = self.parallel.runtime.all_gather_object(local_shard)
        shards = tuple(
            sorted(
                (shard for shard in gathered if shard is not None),
                key=lambda shard: (shard.pp, shard.tp, shard.ep),
            )
        )
        expected = int(self.parallel.tp.size * self.parallel.pp.size * self.parallel.ep.size)
        if len(shards) != expected:
            raise RuntimeError(
                f"collected {len(shards)} rank-local checkpoint shards, expected {expected}"
            )
        shard_ids = [shard.shard_id for shard in shards]
        if len(set(shard_ids)) != len(shard_ids):
            raise RuntimeError(f"duplicate rank-local checkpoint shard ids: {shard_ids}")
        return shards

    def _load_rank_local_state(
        self,
        path: Path,
        manifest: CheckpointManifest,
        *,
        model: nn.Module,
        data_parallel: DataParallelStrategy,
    ) -> None:
        shard = self._select_rank_local_shard(manifest)
        state_file = Path(shard.state_file)
        if state_file.is_absolute() or ".." in state_file.parts:
            raise ValueError(f"invalid rank-local state path in manifest: {shard.state_file!r}")
        payload = _torch_load(path / state_file, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise ValueError(f"rank-local state {state_file} does not contain a mapping")
        model_state = payload.get("model")
        optimizer_state = payload.get("data_parallel")
        if not isinstance(model_state, Mapping):
            raise ValueError(f"rank-local state {state_file} is missing model state")
        if not isinstance(optimizer_state, Mapping):
            raise ValueError(f"rank-local state {state_file} is missing data-parallel state")
        model.load_state_dict(model_state)
        data_parallel.load_state_dict(optimizer_state)

    def _select_rank_local_shard(self, manifest: CheckpointManifest) -> RankLocalShard:
        coordinate = self.parallel.coordinate
        candidates = [
            shard
            for shard in manifest.rank_local_shards
            if shard.tp == coordinate.tp and shard.pp == coordinate.pp
        ]
        exact = [shard for shard in candidates if shard.ep == coordinate.ep]
        if exact:
            return exact[0]
        has_expert = any(
            metadata.parameter_domain.value == "expert"
            for shard in candidates
            for metadata in shard.tensor_metadata.values()
        )
        if not has_expert and candidates:
            # Dense EP is a replica axis. A newly added EP coordinate may reuse
            # any saved dense shard; choose the lowest coordinate deterministically.
            return min(candidates, key=lambda shard: shard.ep)
        raise ValueError(
            "checkpoint has no rank-local shard for coordinate "
            f"tp={coordinate.tp}, pp={coordinate.pp}, ep={coordinate.ep}"
        )

    def _write_rank_runtime_state(
        self,
        target: Path,
        state: Mapping[str, Any],
    ) -> RankRuntimeState:
        coordinate = self.parallel.coordinate
        state_id = (
            f"tp{coordinate.tp:04d}_pp{coordinate.pp:04d}_cp{coordinate.cp:04d}_"
            f"ep{coordinate.ep:04d}_dp{coordinate.dp:04d}"
        )
        relative_path = Path("rank_runtime") / f"{state_id}.pt"
        state_path = target / relative_path
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state), state_path)
        return RankRuntimeState(
            state_id=state_id,
            writer_rank=int(self.parallel.rank),
            tp=int(coordinate.tp),
            pp=int(coordinate.pp),
            cp=int(coordinate.cp),
            ep=int(coordinate.ep),
            dp=int(coordinate.dp),
            state_file=relative_path.as_posix(),
        )

    def _collect_rank_runtime_states(
        self,
        local_state: RankRuntimeState,
    ) -> tuple[RankRuntimeState, ...]:
        if self._distributed_world_size() == 1:
            return (local_state,)
        gathered = self.parallel.runtime.all_gather_object(local_state)
        states = tuple(
            sorted(
                (state for state in gathered if state is not None),
                key=lambda state: (state.pp, state.tp, state.cp, state.ep, state.dp),
            )
        )
        if len(states) != self._distributed_world_size():
            raise RuntimeError(
                f"collected {len(states)} rank runtime states, "
                f"expected {self._distributed_world_size()}"
            )
        state_ids = [state.state_id for state in states]
        if len(set(state_ids)) != len(state_ids):
            raise RuntimeError(f"duplicate rank runtime state ids: {state_ids}")
        return states

    def _load_rank_runtime_state(
        self,
        path: Path,
        manifest: CheckpointManifest,
        *,
        rng: object | None,
    ) -> None:
        if not manifest.rank_runtime_states:
            return
        coordinate = self.parallel.coordinate
        matches = [
            state
            for state in manifest.rank_runtime_states
            if (
                state.tp,
                state.pp,
                state.cp,
                state.ep,
                state.dp,
            )
            == (
                coordinate.tp,
                coordinate.pp,
                coordinate.cp,
                coordinate.ep,
                coordinate.dp,
            )
        ]
        if len(matches) != 1:
            fixed_coordinate_exists = any(
                (state.tp, state.pp, state.cp)
                == (coordinate.tp, coordinate.pp, coordinate.cp)
                for state in manifest.rank_runtime_states
            )
            saved_dp = manifest.parallel_sizes.get("dp")
            saved_ep = manifest.parallel_sizes.get("ep")
            replica_mesh_changed = (
                saved_dp != self.parallel.dp.size
                or saved_ep != self.parallel.ep.size
            )
            if fixed_coordinate_exists and replica_mesh_changed:
                warnings.warn(
                    "checkpoint has no RNG state for a newly introduced DP/EP replica; "
                    "keeping the RNG initialized from the current config, so resume is "
                    "valid but not bitwise-equivalent for this coordinate",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
            raise ValueError(
                "checkpoint has no unique RNG state for coordinate "
                f"tp={coordinate.tp}, pp={coordinate.pp}, cp={coordinate.cp}, "
                f"ep={coordinate.ep}, dp={coordinate.dp}"
            )
        state_file = Path(matches[0].state_file)
        if state_file.is_absolute() or ".." in state_file.parts:
            raise ValueError(f"invalid rank runtime state path: {matches[0].state_file!r}")
        state = _torch_load(path / state_file, map_location="cpu")
        if not isinstance(state, Mapping):
            raise ValueError(f"rank runtime state {state_file} does not contain a mapping")
        _restore_runtime_state(state, rng)

    def _load_model(self, path: Path, backend: str, model: nn.Module) -> None:
        if backend == "torch.distributed.checkpoint":
            try:
                import torch.distributed.checkpoint as dcp
            except ImportError as error:
                raise RuntimeError(
                    "this checkpoint requires torch.distributed.checkpoint"
                ) from error
            state = {"model": model.state_dict()}
            dcp.load(state_dict=state, checkpoint_id=path / "dcp")
            model.load_state_dict(state["model"])
            return
        if backend != "torch.save":
            raise ValueError(f"unknown checkpoint storage backend: {backend!r}")
        state = _torch_load(path / "model.pt", map_location="cpu")
        if not isinstance(state, Mapping):
            raise ValueError("model.pt does not contain a state dict")
        model.load_state_dict(state)

    def _remove_old_checkpoints(self, *, exclude: Path) -> None:
        if self.keep_last == 0:
            return
        checkpoints = sorted(
            path
            for path in self.directory.glob("step_*")
            if path.is_dir() and (path / ".complete").is_file() and path != exclude
        )
        total_to_remove = max(0, len(checkpoints) + 1 - self.keep_last)
        for path in checkpoints[:total_to_remove]:
            shutil.rmtree(path)

    def _distributed_world_size(self) -> int:
        return int(self.parallel.runtime.world_size)

    def _is_coordinator(self) -> bool:
        return int(self.parallel.rank) == 0

    def _barrier(self) -> None:
        self.parallel.runtime.barrier()


def _torch_load(path: Path, *, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _canonical_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def _capture_runtime_state(rng: object | None) -> dict[str, Any]:
    state: dict[str, Any] = {"torch_cpu_rng": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda_rng"] = torch.cuda.get_rng_state(torch.cuda.current_device())
    if rng is not None:
        state_dict = getattr(rng, "state_dict", None)
        if not callable(state_dict):
            raise TypeError("explicit RNG must implement state_dict()")
        state["explicit_rng"] = copy.deepcopy(state_dict())
    return state


def _restore_runtime_state(state: Mapping[str, Any], rng: object | None) -> None:
    cpu_rng = state.get("torch_cpu_rng")
    if not isinstance(cpu_rng, torch.Tensor):
        raise ValueError("rank runtime state is missing torch CPU RNG")
    cuda_rng = state.get("torch_cuda_rng")
    if cuda_rng is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if not isinstance(cuda_rng, torch.Tensor):
            raise ValueError("checkpoint CUDA RNG state is not a tensor")
    explicit_rng = state.get("explicit_rng")
    if explicit_rng is not None:
        if rng is None:
            raise ValueError("checkpoint contains explicit RNG state but no RNG object was passed")
        load_state_dict = getattr(rng, "load_state_dict", None)
        if not callable(load_state_dict):
            raise TypeError("explicit RNG must implement load_state_dict()")

    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng, torch.cuda.current_device())
    if explicit_rng is not None:
        load_state_dict(explicit_rng)
