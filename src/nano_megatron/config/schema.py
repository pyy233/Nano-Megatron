"""Small, typed configuration blocks for Nano-Megatron.

The project intentionally uses separate standard-library dataclasses instead
of one Megatron-style catch-all arguments object.  Modules should receive only
the block they need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from nano_megatron.parallel.axes import (
    DEFAULT_RANK_ORDER,
    ParallelAxis,
    normalize_rank_order,
)


def _require_int(name: str, value: int, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")


def _require_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean, got {value!r}")


def _require_positive_float(name: str, value: float, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {value!r}")
    threshold_ok = value >= 0 if allow_zero else value > 0
    if not threshold_ok:
        operator = ">=" if allow_zero else ">"
        raise ValueError(f"{name} must be {operator} 0, got {value}")


def _coerce_enum(name: str, enum_type: type[StrEnum], value: StrEnum | str) -> StrEnum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        choices = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{name} must be one of: {choices}; got {value!r}") from error


class PrecisionDType(StrEnum):
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"


class KernelBackend(StrEnum):
    TORCH = "torch"
    TRANSFORMER_ENGINE = "transformer_engine"


class DataParallelMode(StrEnum):
    DDP = "ddp"
    ZERO1 = "zero1"
    ZERO2 = "zero2"
    ZERO3 = "zero3"


class PipelineSchedule(StrEnum):
    ONE_F_ONE_B = "1f1b"
    GPIPE = "gpipe"


class ContextParallelBackend(StrEnum):
    ALL_GATHER = "all_gather"
    RING = "ring"


class ActivationCheckpointMode(StrEnum):
    NONE = "none"
    FULL = "full"
    SELECTIVE = "selective"


@dataclass(frozen=True, slots=True)
class DistributedConfig:
    backend: str = "auto"
    timeout_minutes: int = 30
    device: str = "auto"
    init_method: str = "env://"

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("distributed.backend must be a non-empty string")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("distributed.device must be a non-empty string")
        if not isinstance(self.init_method, str) or not self.init_method.strip():
            raise ValueError("distributed.init_method must be a non-empty string")
        _require_int("distributed.timeout_minutes", self.timeout_minutes)
        object.__setattr__(self, "backend", self.backend.lower().strip())
        object.__setattr__(self, "device", self.device.lower().strip())
        object.__setattr__(self, "init_method", self.init_method.strip())


@dataclass(frozen=True, slots=True)
class ParallelConfig:
    tensor: int = 1
    pipeline: int = 1
    context: int = 1
    expert: int = 1
    data: int | None = None
    order: tuple[ParallelAxis, ...] = DEFAULT_RANK_ORDER
    sequence_parallel: bool = False

    def __post_init__(self) -> None:
        for name in ("tensor", "pipeline", "context", "expert"):
            _require_int(f"parallel.{name}", getattr(self, name))
        if self.data is not None:
            _require_int("parallel.data", self.data)
        _require_bool("parallel.sequence_parallel", self.sequence_parallel)
        object.__setattr__(self, "order", normalize_rank_order(self.order))

    @property
    def tensor_parallel_size(self) -> int:
        return self.tensor

    @property
    def pipeline_parallel_size(self) -> int:
        return self.pipeline

    @property
    def context_parallel_size(self) -> int:
        return self.context

    @property
    def expert_parallel_size(self) -> int:
        return self.expert

    @property
    def data_parallel_size(self) -> int:
        if self.data is None:
            raise ValueError("parallel.data has not been resolved against a world size")
        return self.data

    @property
    def model_parallel_size(self) -> int:
        return self.tensor * self.pipeline * self.context * self.expert

    @property
    def configured_world_size(self) -> int | None:
        return None if self.data is None else self.model_parallel_size * self.data

    def resolve_data_parallel_size(self, world_size: int | None) -> int:
        if world_size is not None:
            _require_int("world_size", world_size)
        if self.data is not None:
            expected = self.model_parallel_size * self.data
            if world_size is not None and expected != world_size:
                raise ValueError(
                    "world size does not match parallel topology: "
                    f"{world_size} != TP({self.tensor}) * PP({self.pipeline}) * "
                    f"CP({self.context}) * EP({self.expert}) * DP({self.data}) = {expected}"
                )
            return self.data
        if world_size is None:
            raise ValueError("world_size is required when parallel.data is null")
        if world_size % self.model_parallel_size != 0:
            raise ValueError(
                f"world size {world_size} is not divisible by TP*PP*CP*EP "
                f"({self.model_parallel_size})"
            )
        data = world_size // self.model_parallel_size
        if data < 1:
            raise ValueError(
                f"world size {world_size} is smaller than TP*PP*CP*EP ({self.model_parallel_size})"
            )
        return data

    def resolved(self, world_size: int) -> ParallelConfig:
        return ParallelConfig(
            tensor=self.tensor,
            pipeline=self.pipeline,
            context=self.context,
            expert=self.expert,
            data=self.resolve_data_parallel_size(world_size),
            order=self.order,
            sequence_parallel=self.sequence_parallel,
        )


@dataclass(frozen=True, slots=True)
class GPTConfig:
    layers: int = 12
    hidden_size: int = 768
    ffn_hidden_size: int = 2048
    heads: int = 12
    kv_heads: int | None = None
    seq_length: int = 2048
    vocab_size: int = 50304
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    norm_epsilon: float = 1.0e-5
    bias: bool = False

    def __post_init__(self) -> None:
        for name in (
            "layers",
            "hidden_size",
            "ffn_hidden_size",
            "heads",
            "seq_length",
            "vocab_size",
        ):
            _require_int(f"model.{name}", getattr(self, name))
        if self.kv_heads is not None:
            _require_int("model.kv_heads", self.kv_heads)
        _require_positive_float("model.rope_theta", self.rope_theta)
        _require_positive_float("model.norm_epsilon", self.norm_epsilon)
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not 0 <= self.dropout < 1
        ):
            raise ValueError(f"model.dropout must be in [0, 1), got {self.dropout}")
        _require_bool("model.tie_embeddings", self.tie_embeddings)
        _require_bool("model.bias", self.bias)
        if self.hidden_size % self.heads != 0:
            raise ValueError(
                f"model.hidden_size ({self.hidden_size}) must be divisible by "
                f"model.heads ({self.heads})"
            )
        if self.num_kv_heads > self.heads or self.heads % self.num_kv_heads != 0:
            raise ValueError(
                "model.kv_heads must divide model.heads and cannot exceed it; "
                f"got heads={self.heads}, kv_heads={self.num_kv_heads}"
            )

    @property
    def num_layers(self) -> int:
        return self.layers

    @property
    def num_attention_heads(self) -> int:
        return self.heads

    @property
    def num_kv_heads(self) -> int:
        return self.heads if self.kv_heads is None else self.kv_heads

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.heads

    def padded_vocab_size(self, tensor_parallel_size: int) -> int:
        _require_int("tensor_parallel_size", tensor_parallel_size)
        return (
            (self.vocab_size + tensor_parallel_size - 1) // tensor_parallel_size
        ) * tensor_parallel_size


@dataclass(frozen=True, slots=True)
class PrecisionConfig:
    params: PrecisionDType = PrecisionDType.BFLOAT16
    compute: PrecisionDType = PrecisionDType.BFLOAT16
    grad_reduce: PrecisionDType = PrecisionDType.FLOAT32

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "params", _coerce_enum("precision.params", PrecisionDType, self.params)
        )
        object.__setattr__(
            self,
            "compute",
            _coerce_enum("precision.compute", PrecisionDType, self.compute),
        )
        object.__setattr__(
            self,
            "grad_reduce",
            _coerce_enum("precision.grad_reduce", PrecisionDType, self.grad_reduce),
        )


@dataclass(frozen=True, slots=True)
class KernelConfig:
    backend: KernelBackend = KernelBackend.TORCH

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend",
            _coerce_enum("kernels.backend", KernelBackend, self.backend),
        )


@dataclass(frozen=True, slots=True)
class DataParallelConfig:
    mode: DataParallelMode = DataParallelMode.DDP
    bucket_bytes: int = 256 * 1024 * 1024
    overlap_grad_reduce: bool = False
    reshard_after_forward: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mode",
            _coerce_enum("data_parallel.mode", DataParallelMode, self.mode),
        )
        _require_int("data_parallel.bucket_bytes", self.bucket_bytes)
        _require_bool("data_parallel.overlap_grad_reduce", self.overlap_grad_reduce)
        _require_bool("data_parallel.reshard_after_forward", self.reshard_after_forward)


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    schedule: PipelineSchedule = PipelineSchedule.ONE_F_ONE_B
    activation_dtype: PrecisionDType | None = None
    overlap_p2p: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schedule",
            _coerce_enum("pipeline.schedule", PipelineSchedule, self.schedule),
        )
        if self.activation_dtype is not None:
            object.__setattr__(
                self,
                "activation_dtype",
                _coerce_enum(
                    "pipeline.activation_dtype",
                    PrecisionDType,
                    self.activation_dtype,
                ),
            )
        _require_bool("pipeline.overlap_p2p", self.overlap_p2p)


@dataclass(frozen=True, slots=True)
class ContextParallelConfig:
    backend: ContextParallelBackend = ContextParallelBackend.ALL_GATHER
    dropout: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend",
            _coerce_enum(
                "context_parallel.backend",
                ContextParallelBackend,
                self.backend,
            ),
        )
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise TypeError("context_parallel.dropout must be a number")
        if self.dropout != 0.0:
            raise ValueError(
                "context_parallel.dropout must be 0 in phase one; CP attention "
                "dropout is not implemented"
            )


@dataclass(frozen=True, slots=True)
class ActivationCheckpointConfig:
    mode: ActivationCheckpointMode = ActivationCheckpointMode.NONE
    block_interval: int = 1
    selective_ops: tuple[str, ...] = ("attention", "mlp")
    use_reentrant: bool = False
    offload_saved_tensors: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mode",
            _coerce_enum(
                "activation_checkpoint.mode",
                ActivationCheckpointMode,
                self.mode,
            ),
        )
        object.__setattr__(self, "selective_ops", tuple(self.selective_ops))
        _require_int("activation_checkpoint.block_interval", self.block_interval)
        if not self.selective_ops:
            raise ValueError("activation_checkpoint.selective_ops cannot be empty")
        allowed = {"attention", "mlp"}
        unknown = set(self.selective_ops).difference(allowed)
        if unknown:
            raise ValueError(
                "activation_checkpoint.selective_ops contains unsupported values: "
                + ", ".join(sorted(unknown))
            )
        _require_bool("activation_checkpoint.use_reentrant", self.use_reentrant)
        if self.use_reentrant:
            raise ValueError("Nano-Megatron only supports non-reentrant activation checkpointing")
        _require_bool(
            "activation_checkpoint.offload_saved_tensors",
            self.offload_saved_tensors,
        )


@dataclass(frozen=True, slots=True)
class OffloadConfig:
    optimizer_state: bool = False
    zero3_params_and_grads: bool = False
    activations: bool = False
    pin_memory: bool = True
    non_blocking: bool = True

    def __post_init__(self) -> None:
        for name in (
            "optimizer_state",
            "zero3_params_and_grads",
            "activations",
            "pin_memory",
            "non_blocking",
        ):
            _require_bool(f"offload.{name}", getattr(self, name))


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3.0e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1.0e-8
    weight_decay: float = 0.1
    clip_grad_norm: float | None = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("optimizer.name must be a string")
        object.__setattr__(self, "name", self.name.lower())
        object.__setattr__(self, "betas", tuple(self.betas))
        if self.name != "adamw":
            raise ValueError("optimizer.name must be 'adamw' in the first implementation phase")
        _require_positive_float("optimizer.lr", self.lr)
        _require_positive_float("optimizer.eps", self.eps)
        _require_positive_float("optimizer.weight_decay", self.weight_decay, allow_zero=True)
        if len(self.betas) != 2 or any(
            isinstance(beta, bool) or not isinstance(beta, (int, float)) or not 0 <= beta < 1
            for beta in self.betas
        ):
            raise ValueError("optimizer.betas must contain two values in [0, 1)")
        if self.clip_grad_norm is not None:
            _require_positive_float("optimizer.clip_grad_norm", self.clip_grad_norm)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_steps: int = 1000
    seed: int = 1234
    log_interval: int = 10

    def __post_init__(self) -> None:
        for name in (
            "micro_batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "log_interval",
        ):
            _require_int(f"training.{name}", getattr(self, name))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("training.seed must be a non-negative integer")

    def global_batch_size(self, parallel: ParallelConfig, *, world_size: int | None = None) -> int:
        data_size = parallel.resolve_data_parallel_size(world_size)
        return (
            self.micro_batch_size * self.gradient_accumulation_steps * data_size * parallel.expert
        )


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    directory: Path = Path("checkpoints/gpt")
    save_interval: int = 500
    async_save: bool = False
    keep_last: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        _require_int("checkpoint.save_interval", self.save_interval, minimum=0)
        _require_int("checkpoint.keep_last", self.keep_last)
        _require_bool("checkpoint.async_save", self.async_save)


@dataclass(frozen=True, slots=True)
class DataConfig:
    path: Path | None = None
    tokenizer: str | None = None
    num_workers: int = 0
    shuffle: bool = True
    packed_sequences: bool = False

    def __post_init__(self) -> None:
        if self.path is not None:
            object.__setattr__(self, "path", Path(self.path))
        _require_int("data.num_workers", self.num_workers, minimum=0)
        if self.tokenizer is not None and not isinstance(self.tokenizer, str):
            raise TypeError("data.tokenizer must be a string or null")
        _require_bool("data.shuffle", self.shuffle)
        _require_bool("data.packed_sequences", self.packed_sequences)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    model: GPTConfig = field(default_factory=GPTConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    kernels: KernelConfig = field(default_factory=KernelConfig)
    data_parallel: DataParallelConfig = field(default_factory=DataParallelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    context_parallel: ContextParallelConfig = field(default_factory=ContextParallelConfig)
    activation_checkpoint: ActivationCheckpointConfig = field(
        default_factory=ActivationCheckpointConfig
    )
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def global_batch_size(self, *, world_size: int | None = None) -> int:
        return self.training.global_batch_size(self.parallel, world_size=world_size)
