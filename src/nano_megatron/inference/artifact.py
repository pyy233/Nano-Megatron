"""Validated single-device GPT artifacts exported from training checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from nano_megatron.checkpoint import CheckpointManifest, RankLocalShard, ShardMetadata
from nano_megatron.config import (
    GPTConfig,
    TrainConfig,
    config_from_dict,
    load_config,
    validate_config,
)
from nano_megatron.models.gpt import GPTModel
from nano_megatron.nn.kernels import TorchKernelBackend
from nano_megatron.parallel import ParallelAxis
from nano_megatron.tokenizer import ByteLevelBPETokenizer, validate_tokenizer_for_model

ARTIFACT_NAME = "nano_megatron.single_device_gpt"
ARTIFACT_VERSION = 1
_MANIFEST_FILE = "manifest.json"
_MODEL_FILE = "model.pt"
_TOKENIZER_DIR = "tokenizer"
_COMPLETE_FILE = ".complete"
_LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_SUPPORTED_SOURCE_BACKENDS = {"rank_local_torch.save", "torch.save"}
_SUPPORTED_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class InferenceArtifactError(ValueError):
    """Raised when a checkpoint or exported artifact violates its contract."""


@dataclass(frozen=True, slots=True)
class InferenceArtifact:
    """A fully validated model-only artifact loaded on CPU."""

    path: Path
    manifest: dict[str, Any]
    model_config: GPTConfig
    tokenizer: ByteLevelBPETokenizer
    state_dict: dict[str, Tensor]
    weight_dtype: torch.dtype


@dataclass(slots=True)
class _TensorAssembly:
    tensor: Tensor
    sharded: bool
    boxes: list[tuple[tuple[int, ...], tuple[int, ...]]]
    filled_numel: int = 0


@dataclass(frozen=True, slots=True)
class _LocalGroup:
    rank: int = 0
    size: int = 1
    process_group: object | None = None


@dataclass(frozen=True, slots=True)
class _LocalParallel:
    tp: _LocalGroup = _LocalGroup()
    pp: _LocalGroup = _LocalGroup()
    cp: _LocalGroup = _LocalGroup()
    sequence_parallel: bool = False


def _sha256_file(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InferenceArtifactError(f"could not read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise InferenceArtifactError(f"{description} must contain a JSON object: {path}")
    return value


def _safe_relative_file(root: Path, value: str, *, description: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise InferenceArtifactError(f"invalid {description} path: {value!r}")
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def _torch_load_mapping(path: Path, *, description: str) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise InferenceArtifactError(f"could not load {description} {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise InferenceArtifactError(f"{description} must contain a mapping: {path}")
    return value


def _model_config_dict(config: GPTConfig) -> dict[str, Any]:
    return {field.name: getattr(config, field.name) for field in fields(GPTConfig)}


def _parse_model_config(value: Any) -> GPTConfig:
    if not isinstance(value, Mapping):
        raise InferenceArtifactError("artifact model config must be an object")
    names = {field.name for field in fields(GPTConfig)}
    unknown = set(value).difference(names)
    missing = names.difference(value)
    if unknown or missing:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        raise InferenceArtifactError("invalid artifact model config (" + "; ".join(details) + ")")
    try:
        return GPTConfig(**{name: value[name] for name in names})
    except (TypeError, ValueError) as error:
        raise InferenceArtifactError(f"invalid artifact model config: {error}") from error


def _dtype_name(dtype: torch.dtype) -> str:
    for name, candidate in _SUPPORTED_DTYPES.items():
        if dtype is candidate:
            return name
    raise InferenceArtifactError(f"unsupported model weight dtype: {dtype}")


def _source_world_size(manifest: CheckpointManifest) -> int:
    sizes = manifest.parallel_sizes
    try:
        values = [int(sizes[axis]) for axis in ("tp", "pp", "cp", "ep", "dp")]
    except KeyError as error:
        raise InferenceArtifactError(
            f"checkpoint manifest is missing parallel size {error.args[0]!r}"
        ) from error
    if any(value < 1 for value in values):
        raise InferenceArtifactError(f"checkpoint has invalid parallel sizes: {sizes}")
    result = 1
    for value in values:
        result *= value
    return result


def _config_from_checkpoint(
    manifest: CheckpointManifest,
    *,
    config_path: Path | None,
) -> TrainConfig:
    embedded: TrainConfig | None = None
    if isinstance(manifest.run_config, Mapping):
        try:
            embedded = config_from_dict(manifest.run_config)
        except (TypeError, ValueError) as error:
            if config_path is None:
                raise InferenceArtifactError(
                    f"checkpoint run_config cannot be reconstructed: {error}"
                ) from error
    if config_path is None:
        if embedded is None:
            raise InferenceArtifactError(
                "checkpoint has no usable run_config; pass --config explicitly"
            )
        config = embedded
    else:
        config = load_config(
            config_path,
            world_size=_source_world_size(manifest),
            emit_warnings=False,
        )
        if embedded is not None and config.model != embedded.model:
            raise InferenceArtifactError(
                "explicit config model does not match checkpoint run_config model"
            )

    validate_config(config, world_size=_source_world_size(manifest))
    expected_sizes = {
        "tp": config.parallel.tensor,
        "pp": config.parallel.pipeline,
        "cp": config.parallel.context,
        "ep": config.parallel.expert,
    }
    if config.parallel.data is not None:
        expected_sizes["dp"] = config.parallel.data
    mismatches = {
        axis: (manifest.parallel_sizes.get(axis), expected)
        for axis, expected in expected_sizes.items()
        if manifest.parallel_sizes.get(axis) != expected
    }
    if mismatches:
        raise InferenceArtifactError(
            f"config parallel sizes do not match checkpoint manifest: {mismatches}"
        )
    if config.pipeline.virtual_stages_per_rank != manifest.virtual_stages_per_rank:
        raise InferenceArtifactError(
            "config virtual pipeline size does not match checkpoint manifest: "
            f"{config.pipeline.virtual_stages_per_rank} != "
            f"{manifest.virtual_stages_per_rank}"
        )
    return config


def _tokenizer_path_from_config(config: TrainConfig) -> Path | None:
    tokenizer = config.data.tokenizer
    if tokenizer is None and config.validation.data is not None:
        tokenizer = config.validation.data.tokenizer
    return None if tokenizer is None else tokenizer.path


def _load_source_tokenizer(
    config: TrainConfig,
    *,
    tokenizer_path: Path | None,
) -> tuple[ByteLevelBPETokenizer, Path]:
    source = tokenizer_path or _tokenizer_path_from_config(config)
    if source is None:
        raise InferenceArtifactError(
            "checkpoint config has no tokenizer; pass --tokenizer explicitly"
        )
    try:
        tokenizer = ByteLevelBPETokenizer.load(source)
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        if tokenizer_path is None:
            raise InferenceArtifactError(
                f"could not load checkpoint tokenizer {source}; pass --tokenizer to override: "
                f"{error}"
            ) from error
        raise
    validate_tokenizer_for_model(tokenizer, config.model)
    return tokenizer, Path(source)


def _remap_pipeline_key(key: str, shard: RankLocalShard | None) -> str:
    if not key.startswith("model."):
        raise InferenceArtifactError(
            f"checkpoint model key does not use the GPT pipeline wrapper prefix: {key!r}"
        )
    layer_match = _LAYER_KEY.fullmatch(key)
    if layer_match is None:
        return key.removeprefix("model.")
    if shard is None:
        return key.removeprefix("model.")
    if shard.layer_start is None or shard.layer_end is None:
        raise InferenceArtifactError(
            f"checkpoint layer key {key!r} has no pipeline layer range metadata"
        )
    local_index = int(layer_match.group(1))
    owned_layers = shard.layer_end - shard.layer_start
    if not 0 <= local_index < owned_layers:
        raise InferenceArtifactError(
            f"checkpoint layer key {key!r} is outside shard layer range "
            f"[{shard.layer_start}, {shard.layer_end})"
        )
    global_index = shard.layer_start + local_index
    return f"layers.{global_index}.{layer_match.group(2)}"


def _boxes_overlap(
    first: tuple[tuple[int, ...], tuple[int, ...]],
    second: tuple[tuple[int, ...], tuple[int, ...]],
) -> bool:
    first_start, first_shape = first
    second_start, second_shape = second
    return all(
        a_start < b_start + b_size and b_start < a_start + a_size
        for a_start, a_size, b_start, b_size in zip(
            first_start,
            first_shape,
            second_start,
            second_shape,
            strict=True,
        )
    )


def _add_tensor_piece(
    assemblies: dict[str, _TensorAssembly],
    *,
    key: str,
    tensor: Tensor,
    metadata: ShardMetadata,
) -> None:
    if tuple(tensor.shape) != metadata.local_shape:
        raise InferenceArtifactError(
            f"checkpoint tensor {key!r} shape {tuple(tensor.shape)} does not match "
            f"metadata {metadata.local_shape}"
        )
    if metadata.logical_key == "":
        raise InferenceArtifactError(f"checkpoint tensor {key!r} has an empty logical key")
    sharded = ParallelAxis.TP in metadata.sharded_axes
    if any(axis is not ParallelAxis.TP for axis in metadata.sharded_axes):
        axes = [axis.value for axis in metadata.sharded_axes]
        raise InferenceArtifactError(
            f"checkpoint tensor {key!r} uses unsupported parameter sharding axes {axes}"
        )

    existing = assemblies.get(key)
    if not sharded:
        if metadata.local_shape != metadata.global_shape or any(metadata.global_offset):
            raise InferenceArtifactError(
                f"replicated checkpoint tensor {key!r} has non-replicated metadata"
            )
        if existing is None:
            assemblies[key] = _TensorAssembly(
                tensor=tensor.detach().cpu().clone(),
                sharded=False,
                boxes=[],
            )
            return
        if existing.sharded:
            raise InferenceArtifactError(
                f"checkpoint tensor {key!r} mixes sharded and replicated pieces"
            )
        if existing.tensor.dtype != tensor.dtype or not torch.equal(existing.tensor, tensor.cpu()):
            raise InferenceArtifactError(
                f"replicated TP copies differ for checkpoint tensor {key!r}"
            )
        return

    if existing is None:
        existing = _TensorAssembly(
            tensor=torch.empty(metadata.global_shape, dtype=tensor.dtype, device="cpu"),
            sharded=True,
            boxes=[],
        )
        assemblies[key] = existing
    if not existing.sharded:
        raise InferenceArtifactError(
            f"checkpoint tensor {key!r} mixes replicated and sharded pieces"
        )
    if tuple(existing.tensor.shape) != metadata.global_shape:
        raise InferenceArtifactError(
            f"checkpoint tensor {key!r} has inconsistent global shapes"
        )
    if existing.tensor.dtype != tensor.dtype:
        raise InferenceArtifactError(f"checkpoint tensor {key!r} has inconsistent dtypes")

    box = (metadata.global_offset, metadata.local_shape)
    if any(_boxes_overlap(box, previous) for previous in existing.boxes):
        raise InferenceArtifactError(f"checkpoint tensor {key!r} has overlapping TP shards")
    slices = tuple(
        slice(offset, offset + size)
        for offset, size in zip(
            metadata.global_offset,
            metadata.local_shape,
            strict=True,
        )
    )
    existing.tensor[slices].copy_(tensor.detach().cpu())
    existing.boxes.append(box)
    existing.filled_numel += tensor.numel()


def _finalize_assemblies(
    assemblies: dict[str, _TensorAssembly],
    model_config: GPTConfig,
) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for key, assembly in assemblies.items():
        if assembly.sharded and assembly.filled_numel != assembly.tensor.numel():
            raise InferenceArtifactError(
                f"checkpoint tensor {key!r} TP shards cover {assembly.filled_numel} of "
                f"{assembly.tensor.numel()} elements"
            )
        state[key] = assembly.tensor

    expected = _expected_state_shapes(model_config)
    missing = set(expected).difference(state)
    unexpected = set(state).difference(expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unexpected:
            details.append("unexpected " + ", ".join(sorted(unexpected)))
        raise InferenceArtifactError(
            "merged model state keys are invalid (" + "; ".join(details) + ")"
        )
    for key, shape in expected.items():
        if tuple(state[key].shape) != shape:
            raise InferenceArtifactError(
                f"merged tensor {key!r} shape {tuple(state[key].shape)} != {shape}"
            )

    if model_config.tie_embeddings:
        embedding = state.get("embedding.weight")
        lm_head = state.get("lm_head.weight")
        if embedding is None or lm_head is None:
            raise InferenceArtifactError("tied GPT export is missing embedding or LM-head weight")
        if embedding.dtype != lm_head.dtype or not torch.equal(embedding, lm_head):
            raise InferenceArtifactError(
                "checkpoint tied embedding and LM-head weights are not synchronized"
            )
        state["lm_head.weight"] = embedding
    return state


def _expected_state_shapes(model_config: GPTConfig) -> dict[str, tuple[int, ...]]:
    try:
        with torch.device("meta"):
            model = GPTModel(
                model_config,
                parallel=_LocalParallel(),  # type: ignore[arg-type]
                kernels=TorchKernelBackend(),
            )
    except Exception as error:  # pragma: no cover - defensive context for unusual torch builds.
        raise InferenceArtifactError(
            f"could not construct exported GPT shape oracle: {error}"
        ) from error
    return {key: tuple(tensor.shape) for key, tensor in model.state_dict().items()}


def _merge_rank_local_checkpoint(
    checkpoint: Path,
    manifest: CheckpointManifest,
    model_config: GPTConfig,
) -> dict[str, Tensor]:
    if manifest.virtual_stages_per_rank != 1:
        raise InferenceArtifactError(
            "checkpoint export currently requires pipeline.virtual_stages_per_rank=1"
        )
    if manifest.parallel_sizes.get("ep", 1) != 1:
        raise InferenceArtifactError("checkpoint export currently requires EP=1")
    shards = tuple(shard for shard in manifest.rank_local_shards if shard.ep == 0)
    expected = manifest.parallel_sizes["tp"] * manifest.parallel_sizes["pp"]
    if len(shards) != expected:
        raise InferenceArtifactError(
            f"checkpoint contains {len(shards)} dense TP/PP shards, expected {expected}"
        )
    coordinates = {(shard.tp, shard.pp) for shard in shards}
    expected_coordinates = {
        (tp, pp)
        for pp in range(manifest.parallel_sizes["pp"])
        for tp in range(manifest.parallel_sizes["tp"])
    }
    if coordinates != expected_coordinates:
        raise InferenceArtifactError(
            f"checkpoint TP/PP shard coordinates are incomplete: {coordinates}"
        )

    assemblies: dict[str, _TensorAssembly] = {}
    for shard in sorted(shards, key=lambda item: (item.pp, item.tp)):
        state_path = _safe_relative_file(
            checkpoint,
            shard.state_file,
            description="rank-local checkpoint state",
        )
        payload = _torch_load_mapping(state_path, description="rank-local checkpoint state")
        model_state = payload.get("model")
        if not isinstance(model_state, Mapping):
            raise InferenceArtifactError(f"rank-local checkpoint state {state_path} has no model")
        if set(model_state) != set(shard.tensor_metadata):
            raise InferenceArtifactError(
                f"rank-local state keys do not match manifest metadata: {state_path}"
            )
        for local_key, value in model_state.items():
            if not isinstance(local_key, str) or not isinstance(value, Tensor):
                raise InferenceArtifactError(
                    f"rank-local model state must contain string Tensor entries: {state_path}"
                )
            final_key = _remap_pipeline_key(local_key, shard)
            _add_tensor_piece(
                assemblies,
                key=final_key,
                tensor=value,
                metadata=shard.tensor_metadata[local_key],
            )
        del payload
    return _finalize_assemblies(assemblies, model_config)


def _merge_single_process_checkpoint(
    checkpoint: Path,
    manifest: CheckpointManifest,
    model_config: GPTConfig,
) -> dict[str, Tensor]:
    if any(manifest.parallel_sizes.get(axis, 1) != 1 for axis in ("tp", "pp", "ep")):
        raise InferenceArtifactError(
            "torch.save checkpoint without rank-local shards must have TP=PP=EP=1"
        )
    payload = _torch_load_mapping(checkpoint / _MODEL_FILE, description="checkpoint model state")
    if set(payload) != set(manifest.tensor_metadata):
        raise InferenceArtifactError("checkpoint model keys do not match manifest metadata")
    assemblies: dict[str, _TensorAssembly] = {}
    for local_key, value in payload.items():
        if not isinstance(local_key, str) or not isinstance(value, Tensor):
            raise InferenceArtifactError(
                "checkpoint model state must contain string Tensor entries"
            )
        _add_tensor_piece(
            assemblies,
            key=_remap_pipeline_key(local_key, None),
            tensor=value,
            metadata=manifest.tensor_metadata[local_key],
        )
    return _finalize_assemblies(assemblies, model_config)


def _merge_checkpoint_model(
    checkpoint: Path,
    manifest: CheckpointManifest,
    model_config: GPTConfig,
) -> dict[str, Tensor]:
    if manifest.data_parallel_mode == "zero3":
        raise InferenceArtifactError("ZeRO-3 DCP checkpoint export is not implemented")
    if manifest.storage_backend not in _SUPPORTED_SOURCE_BACKENDS:
        raise InferenceArtifactError(
            f"checkpoint backend {manifest.storage_backend!r} cannot be exported; "
            "supported backends are rank_local_torch.save and torch.save"
        )
    if manifest.storage_backend == "rank_local_torch.save":
        return _merge_rank_local_checkpoint(checkpoint, manifest, model_config)
    return _merge_single_process_checkpoint(checkpoint, manifest, model_config)


def _validate_export_target(directory: Path) -> bool:
    if directory.is_symlink():
        raise FileExistsError(f"refusing to replace inference artifact symlink: {directory}")
    if not directory.exists():
        return False
    if not directory.is_dir():
        raise FileExistsError(f"inference artifact path already exists: {directory}")
    if any(directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty inference artifact directory: {directory}"
        )
    return True


def export_checkpoint(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    config_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
) -> InferenceArtifact:
    """Merge one training checkpoint into an atomically published single-device artifact."""

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_dir() or not (checkpoint / _COMPLETE_FILE).is_file():
        raise ValueError(f"checkpoint is incomplete or missing: {checkpoint}")
    manifest_path = checkpoint / _MANIFEST_FILE
    source_manifest = CheckpointManifest.read(manifest_path)
    if source_manifest.framework != "nano-megatron":
        raise InferenceArtifactError(
            f"unsupported checkpoint framework: {source_manifest.framework!r}"
        )
    config = _config_from_checkpoint(
        source_manifest,
        config_path=None if config_path is None else Path(config_path),
    )
    tokenizer, tokenizer_source = _load_source_tokenizer(
        config,
        tokenizer_path=None if tokenizer_path is None else Path(tokenizer_path),
    )
    state = _merge_checkpoint_model(checkpoint, source_manifest, config.model)
    floating_dtypes = {tensor.dtype for tensor in state.values() if tensor.is_floating_point()}
    if len(floating_dtypes) != 1:
        raise InferenceArtifactError(
            f"exported GPT weights must use exactly one floating dtype, got {floating_dtypes}"
        )
    weight_dtype = next(iter(floating_dtypes))
    dtype_name = _dtype_name(weight_dtype)

    destination = Path(output_path)
    destination_was_empty = _validate_export_target(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    removed_empty_destination = False
    try:
        weights_path = temporary / _MODEL_FILE
        torch.save(state, weights_path)
        tokenizer.save(temporary / _TOKENIZER_DIR)
        manifest = {
            "artifact": ARTIFACT_NAME,
            "format_version": ARTIFACT_VERSION,
            "framework": "nano-megatron",
            "model": _model_config_dict(config.model),
            "source": {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_manifest_sha256": _sha256_file(manifest_path),
                "data_parallel_mode": source_manifest.data_parallel_mode,
                "parallel_sizes": dict(source_manifest.parallel_sizes),
                "step": source_manifest.step,
                "storage_backend": source_manifest.storage_backend,
            },
            "tokenizer": {
                "directory": _TOKENIZER_DIR,
                "eos_token_id": tokenizer.eos_token_id,
                "fingerprint": tokenizer.fingerprint,
                "source": str(tokenizer_source.resolve()),
                "vocab_size": tokenizer.vocab_size,
            },
            "weights": {
                "dtype": dtype_name,
                "file": _MODEL_FILE,
                "sha256": _sha256_file(weights_path),
                "tensors": len(state),
            },
        }
        (temporary / _MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / _COMPLETE_FILE).write_text("complete\n", encoding="utf-8")
        del state
        load_inference_artifact(temporary)

        if destination_was_empty:
            destination.rmdir()
            removed_empty_destination = True
        elif destination.exists() or destination.is_symlink():
            raise FileExistsError(f"inference artifact path already exists: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        if removed_empty_destination and not destination.exists():
            destination.mkdir()
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return load_inference_artifact(destination)


def load_inference_artifact(path: str | Path) -> InferenceArtifact:
    """Load and fully validate one exported single-device artifact on CPU."""

    directory = Path(path)
    if not directory.is_dir() or not (directory / _COMPLETE_FILE).is_file():
        raise ValueError(f"inference artifact is incomplete or missing: {directory}")
    manifest = _read_json_object(directory / _MANIFEST_FILE, "inference manifest")
    if manifest.get("artifact") != ARTIFACT_NAME:
        raise InferenceArtifactError(
            f"unsupported inference artifact: {manifest.get('artifact')!r}"
        )
    if manifest.get("format_version") != ARTIFACT_VERSION:
        raise InferenceArtifactError(
            f"unsupported inference artifact version: {manifest.get('format_version')!r}"
        )
    model_config = _parse_model_config(manifest.get("model"))

    tokenizer_info = manifest.get("tokenizer")
    if not isinstance(tokenizer_info, Mapping):
        raise InferenceArtifactError("inference manifest tokenizer must be an object")
    tokenizer_directory = tokenizer_info.get("directory")
    if not isinstance(tokenizer_directory, str):
        raise InferenceArtifactError("inference manifest tokenizer.directory must be a string")
    tokenizer_relative = Path(tokenizer_directory)
    if tokenizer_relative.is_absolute() or ".." in tokenizer_relative.parts:
        raise InferenceArtifactError("inference tokenizer directory must be relative")
    tokenizer = ByteLevelBPETokenizer.load(directory / tokenizer_relative)
    validate_tokenizer_for_model(tokenizer, model_config)
    if tokenizer_info.get("fingerprint") != tokenizer.fingerprint:
        raise InferenceArtifactError("inference tokenizer fingerprint does not match manifest")
    if tokenizer_info.get("vocab_size") != tokenizer.vocab_size:
        raise InferenceArtifactError("inference tokenizer vocab size does not match manifest")
    if tokenizer_info.get("eos_token_id") != tokenizer.eos_token_id:
        raise InferenceArtifactError("inference tokenizer EOS id does not match manifest")

    weights_info = manifest.get("weights")
    if not isinstance(weights_info, Mapping):
        raise InferenceArtifactError("inference manifest weights must be an object")
    weights_file = weights_info.get("file")
    if not isinstance(weights_file, str):
        raise InferenceArtifactError("inference manifest weights.file must be a string")
    weights_path = _safe_relative_file(
        directory,
        weights_file,
        description="inference weights",
    )
    expected_sha = weights_info.get("sha256")
    if not isinstance(expected_sha, str) or _sha256_file(weights_path) != expected_sha:
        raise InferenceArtifactError("inference weights SHA256 does not match manifest")
    dtype_name = weights_info.get("dtype")
    if dtype_name not in _SUPPORTED_DTYPES:
        raise InferenceArtifactError(f"unsupported inference weight dtype: {dtype_name!r}")
    weight_dtype = _SUPPORTED_DTYPES[str(dtype_name)]
    raw_state = _torch_load_mapping(weights_path, description="inference weights")
    state: dict[str, Tensor] = {}
    for key, value in raw_state.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise InferenceArtifactError(
                "inference weights must contain only string Tensor entries"
            )
        if value.device.type != "cpu":
            raise InferenceArtifactError("inference weights must load onto CPU")
        if value.is_floating_point() and value.dtype is not weight_dtype:
            raise InferenceArtifactError(
                f"inference tensor {key!r} dtype {value.dtype} != {weight_dtype}"
            )
        state[key] = value
    if weights_info.get("tensors") != len(state):
        raise InferenceArtifactError("inference weights tensor count does not match manifest")
    expected_shapes = _expected_state_shapes(model_config)
    if set(state) != set(expected_shapes):
        raise InferenceArtifactError("inference weight keys do not match GPT model state")
    for key, shape in expected_shapes.items():
        if tuple(state[key].shape) != shape:
            raise InferenceArtifactError(
                f"inference tensor {key!r} shape {tuple(state[key].shape)} != {shape}"
            )
    if model_config.tie_embeddings:
        if not torch.equal(state["embedding.weight"], state["lm_head.weight"]):
            raise InferenceArtifactError("inference tied embedding weights differ")
        state["lm_head.weight"] = state["embedding.weight"]

    return InferenceArtifact(
        path=directory.resolve(),
        manifest=manifest,
        model_config=model_config,
        tokenizer=tokenizer,
        state_dict=state,
        weight_dtype=weight_dtype,
    )


__all__ = [
    "ARTIFACT_NAME",
    "ARTIFACT_VERSION",
    "InferenceArtifact",
    "InferenceArtifactError",
    "export_checkpoint",
    "load_inference_artifact",
]
