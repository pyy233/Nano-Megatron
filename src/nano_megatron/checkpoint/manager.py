from __future__ import annotations

import copy
import shutil
import warnings
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from nano_megatron.data_parallel import DataParallelStrategy
from nano_megatron.parallel import GroupKey

from .manifest import (
    CheckpointManifest,
    RankLocalShard,
    RankRuntimeState,
    make_manifest,
    validate_topology,
)
from .mapping import ShardedState

TrainerState = dict[str, Any]


def _call_dcp(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a DCP entry point while tolerating older optional parameters."""

    try:
        parameters = signature(function).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    accepts_kwargs = any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters)
    if not accepts_kwargs and "no_dist" not in {parameter.name for parameter in parameters}:
        kwargs.pop("no_dist", None)
    return function(*args, **kwargs)


@dataclass(frozen=True)
class _PendingCheckpoint:
    target: Path
    manifest: CheckpointManifest
    handles: tuple[Any, ...]


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
        self.async_save = bool(getattr(config, "async_save", False))
        self._pending: _PendingCheckpoint | None = None
        self._async_error: BaseException | None = None
        self._closed = False
        self._async_dcp_groups_ready = False
        self._async_dcp_groups: dict[tuple[int, ...], Any] = {}
        self._owned_async_process_groups: list[Any] = []
        self._executor = (
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="nano-megatron-checkpoint",
            )
            if self.async_save
            else None
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
        self._ensure_open()
        if self.async_save:
            # Only one checkpoint may own a snapshot at a time.  Waiting here also
            # makes a background failure visible before a later step is started.
            self.flush()
            return self._save_async(
                step,
                model=model,
                data_parallel=data_parallel,
                trainer_state=trainer_state,
                path=path,
                rng=rng,
            )
        return self._save_sync(
            step,
            model=model,
            data_parallel=data_parallel,
            trainer_state=trainer_state,
            path=path,
            rng=rng,
        )

    def _save_sync(
        self,
        step: int,
        *,
        model: nn.Module,
        data_parallel: DataParallelStrategy,
        trainer_state: Mapping[str, Any],
        path: Path | str | None,
        rng: object | None,
    ) -> Path:
        if step < 0:
            raise ValueError("checkpoint step cannot be negative")
        registry = data_parallel.parameter_domains
        if registry is None:
            raise ValueError("checkpointing requires an explicit ParameterDomainRegistry")
        virtual_stages_per_rank = self._virtual_stages_per_rank(model)

        target = Path(path) if path is not None else self.directory / f"step_{step:08d}"
        self._preflight_save_target(target)
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
                virtual_stages_per_rank=virtual_stages_per_rank,
            )
            manifest.write(target / "manifest.json")
            (target / ".complete").write_text("complete\n")
            if target.parent.resolve() == self.directory.resolve():
                self._remove_old_checkpoints(exclude=target)
        self._barrier()
        return target

    def _save_async(
        self,
        step: int,
        *,
        model: nn.Module,
        data_parallel: DataParallelStrategy,
        trainer_state: Mapping[str, Any],
        path: Path | str | None,
        rng: object | None,
    ) -> Path:
        if step < 0:
            raise ValueError("checkpoint step cannot be negative")
        registry = data_parallel.parameter_domains
        if registry is None:
            raise ValueError("checkpointing requires an explicit ParameterDomainRegistry")
        virtual_stages_per_rank = self._virtual_stages_per_rank(model)

        target = Path(path) if path is not None else self.directory / f"step_{step:08d}"
        self._preflight_save_target(target)

        mode = str(getattr(data_parallel, "mode", "unknown"))
        rank_local_shards: tuple[RankLocalShard, ...] = ()
        manifest_metadata = {}
        torch_payloads: list[tuple[Any, Path]] = []
        dcp_plan: tuple[dict[str, Any], Path, Any | None, bool] | None = None

        if mode == "zero3":
            dcp_plan = self._prepare_zero3_async_state(target, data_parallel)
            backend = "fsdp2_dcp"
        else:
            canonical_model = _canonical_model(model)
            sharded_model = ShardedState.from_model(
                canonical_model,
                parameter_domains=registry,
                parallel=self.parallel,
            )
            manifest_metadata = sharded_model.metadata
            optimizer_state = _snapshot_for_save(data_parallel.state_dict())
            if self._distributed_world_size() == 1:
                backend, dcp_plan = self._prepare_model_async_state(target, sharded_model)
                if dcp_plan is None:
                    torch_payloads.append(
                        (_snapshot_for_save(sharded_model.state), target / "model.pt")
                    )
                torch_payloads.append((optimizer_state, target / "data_parallel.pt"))
            else:
                local_shard = self._rank_local_shard(
                    canonical_model,
                    sharded_model=sharded_model,
                )
                if local_shard is not None:
                    torch_payloads.append(
                        (
                            _snapshot_for_save(
                                {
                                    "model": sharded_model.state,
                                    "data_parallel": optimizer_state,
                                }
                            ),
                            target / local_shard.state_file,
                        )
                    )
                rank_local_shards = self._collect_rank_local_shards(local_shard)
                backend = "rank_local_torch.save"
                manifest_metadata = {}

        captured_runtime_state = _snapshot_for_save(_capture_runtime_state(rng))
        local_runtime_state = self._rank_runtime_state()
        torch_payloads.append((captured_runtime_state, target / local_runtime_state.state_file))
        rank_runtime_states = self._collect_rank_runtime_states(local_runtime_state)

        if self._is_coordinator():
            torch_payloads.append(
                (_snapshot_for_save(dict(trainer_state)), target / "trainer_state.pt")
            )
        manifest = make_manifest(
            step=step,
            storage_backend=backend,
            parallel=self.parallel,
            metadata=manifest_metadata,
            data_parallel_mode=mode,
            rank_local_shards=rank_local_shards,
            rank_runtime_states=rank_runtime_states,
            run_config=self.run_config,
            virtual_stages_per_rank=virtual_stages_per_rank,
        )

        handles: list[Any] = []
        try:
            if dcp_plan is not None:
                handles.append(self._start_dcp_async_save(*dcp_plan))
            if torch_payloads:
                assert self._executor is not None
                handles.append(self._executor.submit(_write_torch_payloads, torch_payloads))
        except BaseException:
            for handle in handles:
                with suppress(BaseException):
                    _wait_async_handle(handle)
            raise

        self._pending = _PendingCheckpoint(
            target=target,
            manifest=manifest,
            handles=tuple(handles),
        )
        return target

    def flush(self) -> None:
        """Finish the pending asynchronous save and publish it atomically.

        The method is a no-op for synchronous managers and when no save is
        pending.  A background error is sticky so it cannot be accidentally
        ignored by a later ``save()`` or ``close()`` call.
        """

        if self._async_error is not None:
            raise self._async_error
        pending = self._pending
        if pending is None:
            return

        try:
            self._finish_pending(pending)
        except BaseException as error:
            self._async_error = error
            raise
        finally:
            self._pending = None

    def close(self) -> None:
        """Flush pending work and release the checkpoint worker thread."""

        if self._closed:
            if self._async_error is not None:
                raise self._async_error
            return
        try:
            self.flush()
        finally:
            try:
                if self._executor is not None:
                    self._executor.shutdown(wait=True)
                for process_group in reversed(self._owned_async_process_groups):
                    self.parallel.runtime.destroy_group(process_group)
            finally:
                self._owned_async_process_groups.clear()
                self._async_dcp_groups.clear()
                self._closed = True

    def __enter__(self) -> CheckpointManager:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

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
        current_virtual_stages = _model_virtual_stages_per_rank(model)
        if manifest.virtual_stages_per_rank != current_virtual_stages:
            raise ValueError(
                "checkpoint restore cannot change pipeline.virtual_stages_per_rank: "
                f"{manifest.virtual_stages_per_rank} -> {current_virtual_stages}"
            )
        current_mode = str(getattr(data_parallel, "mode", "unknown"))
        if manifest.data_parallel_mode != "unknown" and manifest.data_parallel_mode != current_mode:
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

    def _finish_pending(self, pending: _PendingCheckpoint) -> None:
        local_error: BaseException | None = None
        for handle in pending.handles:
            try:
                _wait_async_handle(handle)
            except BaseException as error:
                if local_error is None:
                    local_error = error

        self._collective_error(
            local_error,
            message=(f"asynchronous checkpoint payload write failed for {pending.target}"),
        )

        finalization_error: BaseException | None = None
        if self._is_coordinator():
            try:
                _atomic_manifest_write(
                    pending.manifest,
                    pending.target / "manifest.json",
                )
                _atomic_text_write(pending.target / ".complete", "complete\n")
                if pending.target.parent.resolve() == self.directory.resolve():
                    self._remove_old_checkpoints(exclude=pending.target)
            except BaseException as error:
                finalization_error = error

        self._collective_error(
            finalization_error,
            message=f"asynchronous checkpoint finalization failed for {pending.target}",
        )

    def _collective_error(
        self,
        error: BaseException | None,
        *,
        message: str,
    ) -> None:
        """Raise one WORLD-consistent error from a possibly rank-local failure."""

        description = None if error is None else f"{type(error).__name__}: {error}"
        gathered = self.parallel.runtime.all_gather_object(description)
        failures = [
            (rank, remote_description)
            for rank, remote_description in enumerate(gathered)
            if remote_description is not None
        ]
        if not failures:
            return
        details = "; ".join(f"rank {rank}: {description}" for rank, description in failures)
        collective_error = RuntimeError(f"{message}: {details}")
        if error is not None:
            raise collective_error from error
        raise collective_error

    def _preflight_save_target(self, target: Path) -> None:
        local_error: BaseException | None = None
        if self._is_coordinator():
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.mkdir(parents=False, exist_ok=False)
            except BaseException as error:
                local_error = error

        # This WORLD object collective both publishes a coordinator filesystem
        # failure and replaces the old barrier after successful directory creation.
        self._collective_error(
            local_error,
            message=f"checkpoint save preflight failed for {target}",
        )

    def _virtual_stages_per_rank(self, model: nn.Module) -> int:
        actual = _model_virtual_stages_per_rank(model)
        configured = _run_config_virtual_stages_per_rank(self.run_config)
        if configured is not None and configured != actual:
            raise ValueError(
                "checkpoint run_config pipeline.virtual_stages_per_rank does not "
                f"match model.layout: {configured} != {actual}"
            )
        return actual

    def _prepare_zero3_async_state(
        self,
        target: Path,
        data_parallel: DataParallelStrategy,
    ) -> tuple[dict[str, Any], Path, Any | None, bool]:
        try:
            import torch.distributed.checkpoint as dcp
        except ImportError as error:
            raise RuntimeError("ZeRO-3 checkpointing requires PyTorch DCP") from error
        if not callable(getattr(dcp, "async_save", None)):
            raise RuntimeError("ZeRO-3 async checkpointing requires DCP async_save()")
        state_dict = getattr(data_parallel, "distributed_checkpoint_state_dict", None)
        if not callable(state_dict):
            raise RuntimeError("ZeRO-3 strategy does not expose a DCP state dict")
        group = self._zero3_state_group(data_parallel)
        process_group = self._async_dcp_process_group(group)
        return (
            state_dict(),
            self._zero3_state_path(target, data_parallel),
            process_group,
            process_group is None,
        )

    def _async_dcp_process_group(self, group: Any) -> Any | None:
        process_group = group.process_group
        if process_group is None or _process_group_supports_cpu(
            process_group,
            backend=str(getattr(group, "backend", "")),
        ):
            return process_group
        self._materialize_async_dcp_groups()
        ranks = tuple(int(rank) for rank in group.ranks)
        try:
            return self._async_dcp_groups[ranks]
        except KeyError as error:
            raise RuntimeError(
                f"could not create a CPU checkpoint process group for ZeRO-3 replica ranks {ranks}"
            ) from error

    def _materialize_async_dcp_groups(self) -> None:
        if self._async_dcp_groups_ready:
            return
        families: set[tuple[int, ...]] = set()
        for key in (GroupKey.DENSE_REPLICA, GroupKey.EXPERT_REPLICA):
            families.update(
                tuple(int(rank) for rank in ranks) for ranks in self.parallel.group_family(key)
            )

        # All ranks execute this loop in the same order.  c10d requires even
        # non-members to participate when new process groups are created.
        for ranks in sorted(families):
            try:
                process_group = self.parallel.runtime.new_group(ranks, backend="gloo")
            except BaseException as error:
                raise RuntimeError(
                    "DCP async_save requires a CPU-capable process group; failed "
                    f"to create a Gloo checkpoint group for ranks {ranks}"
                ) from error
            self._owned_async_process_groups.append(process_group)
            if int(self.parallel.rank) in ranks:
                self._async_dcp_groups[ranks] = process_group
        self._async_dcp_groups_ready = True

    def _prepare_model_async_state(
        self,
        target: Path,
        state: ShardedState,
    ) -> tuple[str, tuple[dict[str, Any], Path, Any | None, bool] | None]:
        try:
            import torch.distributed.checkpoint as dcp
        except ImportError:
            dcp = None  # type: ignore[assignment]
        if dcp is None or not callable(getattr(dcp, "async_save", None)):
            return "torch.save", None
        return (
            "torch.distributed.checkpoint",
            ({"model": state.state}, target / "dcp", None, True),
        )

    def _start_dcp_async_save(
        self,
        state: dict[str, Any],
        checkpoint_id: Path,
        process_group: Any | None,
        no_dist: bool,
    ) -> Any:
        try:
            import torch.distributed.checkpoint as dcp
        except ImportError as error:
            raise RuntimeError("async checkpointing requires PyTorch DCP") from error
        return _call_dcp(
            dcp.async_save,
            state,
            checkpoint_id=checkpoint_id,
            process_group=process_group,
            no_dist=no_dist,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("CheckpointManager is closed")

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
        _call_dcp(
            dcp.save,
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
        _call_dcp(
            dcp.load,
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
        shard = self._rank_local_shard(model, sharded_model=sharded_model)
        if shard is None:
            return None
        state_path = target / shard.state_file
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": sharded_model.state,
                "data_parallel": dict(data_parallel_state),
            },
            state_path,
        )
        return shard

    def _rank_local_shard(
        self,
        model: nn.Module,
        *,
        sharded_model: ShardedState,
    ) -> RankLocalShard | None:
        coordinate = self.parallel.coordinate
        # DP and CP are replica axes for phase-one GPT parameters. One writer
        # per (TP, PP, EP) coordinate prevents both duplicate files and PP/TP key collisions.
        if coordinate.dp != 0 or coordinate.cp != 0:
            return None
        shard_id = f"tp{coordinate.tp:04d}_pp{coordinate.pp:04d}_ep{coordinate.ep:04d}"
        relative_path = Path("rank_states") / f"{shard_id}.pt"
        partition = getattr(model, "partition", None)
        if partition is not None:
            layer_ranges = ((int(partition.start_layer), int(partition.end_layer)),)
        else:
            layer_ranges = tuple(
                (
                    int(chunk.partition.start_layer),
                    int(chunk.partition.end_layer),
                )
                for chunk in getattr(model, "chunks", ())
                if getattr(chunk, "partition", None) is not None
            )
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
            layer_ranges=layer_ranges,
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
        runtime_state = self._rank_runtime_state()
        state_path = target / runtime_state.state_file
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state), state_path)
        return runtime_state

    def _rank_runtime_state(self) -> RankRuntimeState:
        coordinate = self.parallel.coordinate
        state_id = (
            f"tp{coordinate.tp:04d}_pp{coordinate.pp:04d}_cp{coordinate.cp:04d}_"
            f"ep{coordinate.ep:04d}_dp{coordinate.dp:04d}"
        )
        relative_path = Path("rank_runtime") / f"{state_id}.pt"
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
                (state.tp, state.pp, state.cp) == (coordinate.tp, coordinate.pp, coordinate.cp)
                for state in manifest.rank_runtime_states
            )
            saved_dp = manifest.parallel_sizes.get("dp")
            saved_ep = manifest.parallel_sizes.get("ep")
            replica_mesh_changed = (
                saved_dp != self.parallel.dp.size or saved_ep != self.parallel.ep.size
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


def _wait_async_handle(handle: Any) -> None:
    if isinstance(handle, Future):
        handle.result()
        return
    staging = getattr(handle, "staging_completion", None)
    if staging is not None:
        staging.result()
    upload = getattr(handle, "upload_completion", None)
    if upload is not None:
        upload.result()
        return
    result = getattr(handle, "result", None)
    if not callable(result):
        raise TypeError("async checkpoint backend returned an object without a completion future")
    result()


def _process_group_supports_cpu(process_group: Any, *, backend: str) -> bool:
    if "gloo" in backend.lower():
        return True
    try:
        device_types = process_group._device_types
    except (AttributeError, RuntimeError):
        return False
    return any(torch.device(device).type == "cpu" for device in device_types)


def _snapshot_for_save(value: Any) -> Any:
    """Detach checkpoint payloads from state that training may mutate."""

    if isinstance(value, torch.Tensor):
        detached = value.detach()
        try:
            return detached.to(device="cpu", copy=True)
        except TypeError:
            return detached.cpu().clone()
    if isinstance(value, Mapping):
        return {copy.deepcopy(key): _snapshot_for_save(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_for_save(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_for_save(item) for item in value)
    return copy.deepcopy(value)


def _write_torch_payloads(payloads: list[tuple[Any, Path]]) -> None:
    for payload, path in payloads:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            torch.save(payload, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _atomic_manifest_write(manifest: CheckpointManifest, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        manifest.write(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text_write(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(value)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _torch_load(path: Path, *, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _canonical_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def _model_virtual_stages_per_rank(model: nn.Module) -> int:
    canonical_model = _canonical_model(model)
    layout = getattr(canonical_model, "layout", None)
    value = getattr(layout, "virtual_stages_per_rank", 1)
    return _validate_virtual_stages_per_rank(value, source="model.layout")


def _run_config_virtual_stages_per_rank(run_config: object | None) -> int | None:
    if run_config is None:
        return None
    if isinstance(run_config, Mapping):
        pipeline = run_config.get("pipeline")
    else:
        pipeline = getattr(run_config, "pipeline", None)
    if pipeline is None:
        return None
    if isinstance(pipeline, Mapping):
        value = pipeline.get("virtual_stages_per_rank")
    else:
        value = getattr(pipeline, "virtual_stages_per_rank", None)
    if value is None:
        return None
    return _validate_virtual_stages_per_rank(value, source="run_config.pipeline")


def _validate_virtual_stages_per_rank(value: object, *, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{source}.virtual_stages_per_rank must be an integer")
    if value < 1:
        raise ValueError(f"{source}.virtual_stages_per_rank must be at least 1")
    return value


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
