from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from nano_megatron.parallel import GroupKey, ParallelAxis, ParameterDomain

from .mapping import ShardMetadata

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class RankLocalShard:
    """One model/optimizer shard selected by topology coordinate, not global rank."""

    shard_id: str
    writer_rank: int
    tp: int
    pp: int
    ep: int
    cp: int
    dp: int
    state_file: str
    tensor_metadata: dict[str, ShardMetadata]
    layer_start: int | None = None
    layer_end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "writer_rank": self.writer_rank,
            "coordinate": {
                "tp": self.tp,
                "pp": self.pp,
                "ep": self.ep,
                "cp": self.cp,
                "dp": self.dp,
            },
            "state_file": self.state_file,
            "layer_range": (
                None
                if self.layer_start is None or self.layer_end is None
                else [self.layer_start, self.layer_end]
            ),
            "tensor_metadata": {
                key: metadata.to_dict() for key, metadata in self.tensor_metadata.items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RankLocalShard:
        coordinate = value["coordinate"]
        layer_range = value.get("layer_range")
        return cls(
            shard_id=str(value["shard_id"]),
            writer_rank=int(value["writer_rank"]),
            tp=int(coordinate["tp"]),
            pp=int(coordinate["pp"]),
            ep=int(coordinate["ep"]),
            cp=int(coordinate["cp"]),
            dp=int(coordinate["dp"]),
            state_file=str(value["state_file"]),
            tensor_metadata={
                str(key): ShardMetadata.from_dict(metadata)
                for key, metadata in value.get("tensor_metadata", {}).items()
            },
            layer_start=None if layer_range is None else int(layer_range[0]),
            layer_end=None if layer_range is None else int(layer_range[1]),
        )


@dataclass(frozen=True)
class RankRuntimeState:
    """RNG and other execution state owned by one full five-axis coordinate."""

    state_id: str
    writer_rank: int
    tp: int
    pp: int
    cp: int
    ep: int
    dp: int
    state_file: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "writer_rank": self.writer_rank,
            "coordinate": {
                "tp": self.tp,
                "pp": self.pp,
                "cp": self.cp,
                "ep": self.ep,
                "dp": self.dp,
            },
            "state_file": self.state_file,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RankRuntimeState:
        coordinate = value["coordinate"]
        return cls(
            state_id=str(value["state_id"]),
            writer_rank=int(value["writer_rank"]),
            tp=int(coordinate["tp"]),
            pp=int(coordinate["pp"]),
            cp=int(coordinate["cp"]),
            ep=int(coordinate["ep"]),
            dp=int(coordinate["dp"]),
            state_file=str(value["state_file"]),
        )


@dataclass(frozen=True)
class CheckpointManifest:
    schema_version: int
    framework: str
    torch_version: str
    step: int
    storage_backend: str
    parallel_sizes: dict[str, int]
    rank_order: tuple[str, ...]
    tensor_metadata: dict[str, ShardMetadata]
    data_parallel_mode: str = "unknown"
    rank_local_shards: tuple[RankLocalShard, ...] = ()
    rank_runtime_states: tuple[RankRuntimeState, ...] = ()
    topology_compatibility: dict[str, Any] | None = None
    run_config: Any | None = None
    group_plan: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "framework": self.framework,
            "torch_version": self.torch_version,
            "step": self.step,
            "storage_backend": self.storage_backend,
            "parallel_sizes": self.parallel_sizes,
            "rank_order": list(self.rank_order),
            "tensor_metadata": {
                key: metadata.to_dict() for key, metadata in self.tensor_metadata.items()
            },
            "data_parallel_mode": self.data_parallel_mode,
            "rank_local_shards": [shard.to_dict() for shard in self.rank_local_shards],
            "rank_runtime_states": [state.to_dict() for state in self.rank_runtime_states],
            "topology_compatibility": json_compatible(self.topology_compatibility),
            "run_config": json_compatible(self.run_config),
            "group_plan": json_compatible(self.group_plan),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointManifest:
        return cls(
            schema_version=int(value["schema_version"]),
            framework=str(value["framework"]),
            torch_version=str(value["torch_version"]),
            step=int(value["step"]),
            storage_backend=str(value["storage_backend"]),
            parallel_sizes={str(key): int(size) for key, size in value["parallel_sizes"].items()},
            rank_order=tuple(str(item) for item in value.get("rank_order", ())),
            tensor_metadata={
                str(key): ShardMetadata.from_dict(metadata)
                for key, metadata in value.get("tensor_metadata", {}).items()
            },
            data_parallel_mode=str(value.get("data_parallel_mode", "unknown")),
            rank_local_shards=tuple(
                RankLocalShard.from_dict(shard)
                for shard in value.get("rank_local_shards", ())
            ),
            rank_runtime_states=tuple(
                RankRuntimeState.from_dict(state)
                for state in value.get("rank_runtime_states", ())
            ),
            topology_compatibility=value.get("topology_compatibility"),
            run_config=value.get("run_config"),
            group_plan=value.get("group_plan"),
        )

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path: Path) -> CheckpointManifest:
        return cls.from_dict(json.loads(path.read_text()))


def parallel_sizes(parallel: object) -> dict[str, int]:
    keys = {
        ParallelAxis.TP: GroupKey.TP,
        ParallelAxis.PP: GroupKey.PP,
        ParallelAxis.CP: GroupKey.CP,
        ParallelAxis.EP: GroupKey.EP,
        ParallelAxis.DP: GroupKey.DP_AXIS,
    }
    return {axis.value: int(parallel.group(key).size) for axis, key in keys.items()}


def parallel_rank_order(parallel: object) -> tuple[str, ...]:
    topology = getattr(parallel, "topology", None)
    order = getattr(topology, "order", ())
    return tuple(str(getattr(axis, "value", axis)) for axis in order)


def make_manifest(
    *,
    step: int,
    storage_backend: str,
    parallel: object,
    metadata: Mapping[str, ShardMetadata],
    data_parallel_mode: str = "unknown",
    rank_local_shards: tuple[RankLocalShard, ...] = (),
    rank_runtime_states: tuple[RankRuntimeState, ...] = (),
    run_config: object | None = None,
) -> CheckpointManifest:
    group_plan_method = getattr(parallel, "group_plan", None)
    group_plan = group_plan_method() if callable(group_plan_method) else None
    all_metadata = list(metadata.values())
    for shard in rank_local_shards:
        all_metadata.extend(shard.tensor_metadata.values())
    has_expert = any(
        item.parameter_domain is ParameterDomain.EXPERT for item in all_metadata
    )
    fixed_axes = ["tp", "pp", "cp"]
    resizable_axes = ["dp"]
    if not has_expert:
        resizable_axes.append("ep")
    if data_parallel_mode == "zero3":
        resizable_axes = ["dp"]
        fixed_axes = ["tp", "pp", "cp", "ep"]
    fsdp2_dcp = storage_backend == "fsdp2_dcp"
    return CheckpointManifest(
        schema_version=SCHEMA_VERSION,
        framework="nano-megatron",
        torch_version=torch.__version__,
        step=step,
        storage_backend=storage_backend,
        parallel_sizes=parallel_sizes(parallel),
        rank_order=parallel_rank_order(parallel),
        tensor_metadata=dict(metadata),
        data_parallel_mode=data_parallel_mode,
        rank_local_shards=rank_local_shards,
        rank_runtime_states=rank_runtime_states,
        topology_compatibility={
            "fixed_axes": fixed_axes,
            "resizable_axes": resizable_axes,
            "rank_order_may_change": True,
            "model_shard_key": (
                ["parameter_domain", "tp", "pp", "expert_ep"]
                if fsdp2_dcp
                else ["tp", "pp", "ep"]
            ),
            "optimizer_layout_requires_same_parameter_order": not fsdp2_dcp,
            "zero_bucket_boundaries_must_match": data_parallel_mode in {"zero1", "zero2"},
            "shared_filesystem_required": bool(rank_local_shards) or fsdp2_dcp,
            "rng_exact_for_saved_coordinates": True,
            "new_replica_rng_policy": "keep_initialized_state_with_warning",
        },
        run_config=run_config,
        group_plan=group_plan,
    )


def validate_topology(manifest: CheckpointManifest, parallel: object) -> None:
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema {manifest.schema_version}; expected {SCHEMA_VERSION}"
        )
    current = parallel_sizes(parallel)
    for axis in (ParallelAxis.TP, ParallelAxis.PP, ParallelAxis.CP):
        saved_size = manifest.parallel_sizes.get(axis.value)
        if saved_size is not None and saved_size != current[axis.value]:
            raise ValueError(
                f"phase-one checkpoint restore cannot change {axis.value.upper()} size "
                f"({saved_size} -> {current[axis.value]})"
            )
    has_expert = any(
        metadata.parameter_domain is ParameterDomain.EXPERT
        for metadata in manifest.tensor_metadata.values()
    )
    has_expert = has_expert or any(
        metadata.parameter_domain is ParameterDomain.EXPERT
        for shard in manifest.rank_local_shards
        for metadata in shard.tensor_metadata.values()
    )
    saved_ep = manifest.parallel_sizes.get(ParallelAxis.EP.value)
    if has_expert and saved_ep is not None and saved_ep != current[ParallelAxis.EP.value]:
        raise ValueError("changing EP size for expert parameters requires an expert remap planner")
    if manifest.data_parallel_mode == "zero3":
        saved_ep = manifest.parallel_sizes.get(ParallelAxis.EP.value)
        if saved_ep is not None and saved_ep != current[ParallelAxis.EP.value]:
            raise ValueError(
                "phase-one ZeRO-3 restore cannot change EP size: "
                f"{saved_ep} -> {current[ParallelAxis.EP.value]}"
            )


def json_compatible(value: Any) -> Any:
    if is_dataclass(value):
        return {
            item.name: json_compatible(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        ordered = sorted(value, key=lambda item: str(getattr(item, "value", item)))
        return [json_compatible(item) for item in ordered]
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    return value
