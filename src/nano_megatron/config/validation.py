"""Cross-block validation and derived configuration values."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import (
    ActivationCheckpointMode,
    DataParallelMode,
    PipelineSchedule,
    PrecisionDType,
    TrainConfig,
)


class ConfigValidationError(ValueError):
    """Raised when individually valid configuration blocks cannot be combined."""


@dataclass(frozen=True, slots=True)
class ValidationResult:
    world_size: int | None
    data_parallel_size: int | None
    padded_vocab_size: int
    global_batch_size: int | None
    warnings: tuple[str, ...] = ()


def validate_config(
    config: TrainConfig,
    *,
    world_size: int | None = None,
) -> ValidationResult:
    """Validate a complete training configuration.

    Validation is intentionally centralized here rather than spread across
    model and optimizer constructors.  Dataclass ``__post_init__`` methods only
    enforce invariants local to one block.
    """

    errors: list[str] = []
    notices: list[str] = []
    parallel = config.parallel
    model = config.model

    if (
        config.precision.compute is PrecisionDType.FLOAT32
        and config.precision.params is not PrecisionDType.FLOAT32
    ):
        errors.append(
            "precision.compute=float32 requires precision.params=float32; "
            "low-precision parameters cannot guarantee FP32 compute"
        )

    data_parallel_size: int | None = None
    try:
        if world_size is not None or parallel.data is not None:
            data_parallel_size = parallel.resolve_data_parallel_size(world_size)
    except (TypeError, ValueError) as error:
        errors.append(str(error))

    divisibility_checks = (
        (model.hidden_size, parallel.tensor, "model.hidden_size", "parallel.tensor"),
        (
            model.ffn_hidden_size,
            parallel.tensor,
            "model.ffn_hidden_size",
            "parallel.tensor",
        ),
        (model.heads, parallel.tensor, "model.heads", "parallel.tensor"),
        (model.num_kv_heads, parallel.tensor, "model.kv_heads", "parallel.tensor"),
        (model.seq_length, parallel.context, "model.seq_length", "parallel.context"),
    )
    for value, divisor, value_name, divisor_name in divisibility_checks:
        if value % divisor != 0:
            errors.append(f"{value_name} ({value}) must be divisible by {divisor_name} ({divisor})")

    if parallel.sequence_parallel:
        cp_local_sequence = model.seq_length // parallel.context
        if model.seq_length % parallel.context == 0 and cp_local_sequence % parallel.tensor != 0:
            errors.append(
                "sequence parallelism requires (model.seq_length / parallel.context) "
                f"to be divisible by parallel.tensor; got {cp_local_sequence} and {parallel.tensor}"
            )
        if parallel.tensor == 1:
            notices.append(
                "parallel.sequence_parallel is enabled with TP=1 and therefore "
                "has no sharding effect"
            )

    if model.layers < parallel.pipeline:
        errors.append(
            f"model.layers ({model.layers}) must be >= parallel.pipeline ({parallel.pipeline})"
        )

    if (
        config.pipeline.schedule is PipelineSchedule.ONE_F_ONE_B
        and config.training.gradient_accumulation_steps < parallel.pipeline
    ):
        notices.append(
            "1F1B pipeline utilization is low when gradient_accumulation_steps "
            f"({config.training.gradient_accumulation_steps}) < PP ({parallel.pipeline})"
        )

    if parallel.context > 1 and config.data.packed_sequences:
        errors.append("context parallelism does not support packed/document masks in phase one")
    if parallel.context > 1 and model.dropout > 0.0:
        errors.append(
            "model.dropout must be 0 when context parallelism is enabled in phase one; "
            "CP attention dropout is not implemented"
        )

    mode = config.data_parallel.mode
    if config.precision.params is PrecisionDType.FLOAT16 and mode in {
        DataParallelMode.DDP,
        DataParallelMode.ZERO3,
    }:
        errors.append(
            f"precision.params=float16 is not supported with data_parallel.mode={mode.value}; "
            "DDP and ZeRO-3 do not maintain FP32 master parameters. Use "
            "precision.params=float32 with precision.compute=float16, or use ZeRO-1/ZeRO-2 "
            "with their FP32 master shards"
        )
    if config.offload.optimizer_state:
        if mode is DataParallelMode.DDP:
            errors.append("optimizer-state offload requires data_parallel.mode zero1 or zero2")
        elif mode is DataParallelMode.ZERO3:
            errors.append(
                "optimizer_state offload is not a separate ZeRO-3 option; use "
                "offload.zero3_params_and_grads with the FSDP2 CPU offload policy"
            )
    if config.offload.zero3_params_and_grads and mode is not DataParallelMode.ZERO3:
        errors.append("offload.zero3_params_and_grads requires data_parallel.mode=zero3")
    if config.data_parallel.overlap_grad_reduce and mode is DataParallelMode.ZERO3:
        notices.append(
            "data_parallel.overlap_grad_reduce is managed internally by FSDP2 in zero3 mode"
        )
    elif config.data_parallel.overlap_grad_reduce:
        errors.append(
            "data_parallel.overlap_grad_reduce is not implemented for ddp/zero1/zero2 "
            "in phase one; set it to false"
        )

    if config.pipeline.overlap_p2p:
        errors.append(
            "pipeline.overlap_p2p is not implemented in phase one; set it to false"
        )
    if config.checkpoint.async_save:
        errors.append(
            "checkpoint.async_save is not implemented in phase one; set it to false"
        )

    if (
        config.activation_checkpoint.mode is ActivationCheckpointMode.SELECTIVE
        and not config.activation_checkpoint.selective_ops
    ):
        errors.append("selective activation checkpointing requires at least one selective op")
    if (
        config.activation_checkpoint.mode is ActivationCheckpointMode.NONE
        and config.activation_checkpoint.offload_saved_tensors
    ):
        errors.append(
            "activation_checkpoint.offload_saved_tensors requires activation "
            "checkpointing to be enabled"
        )

    if parallel.expert > 1:
        notices.append(
            "dense GPT with EP>1 treats EP ranks as additional dense batch replicas; "
            "MoE token dispatch is not implemented in phase one"
        )

    dense_replica_size: int | None = None
    if data_parallel_size is not None:
        dense_replica_size = data_parallel_size * parallel.expert * parallel.context
    if mode is DataParallelMode.ZERO3 and dense_replica_size == 1:
        notices.append("ZeRO-3 replica mesh has size 1 and will degenerate to local execution")
        if config.offload.zero3_params_and_grads:
            errors.append(
                "offload.zero3_params_and_grads requires an active ZeRO-3 replica mesh; "
                "the configured mesh has size 1"
            )
    if errors:
        formatted = "invalid Nano-Megatron configuration:\n- " + "\n- ".join(errors)
        raise ConfigValidationError(formatted)

    padded_vocab_size = model.padded_vocab_size(parallel.tensor)
    global_batch_size = None
    if data_parallel_size is not None:
        global_batch_size = (
            config.training.micro_batch_size
            * config.training.gradient_accumulation_steps
            * data_parallel_size
            * parallel.expert
        )
    return ValidationResult(
        world_size=world_size,
        data_parallel_size=data_parallel_size,
        padded_vocab_size=padded_vocab_size,
        global_batch_size=global_batch_size,
        warnings=tuple(notices),
    )


# The longer name reads naturally at assembly sites and remains an alias, not
# a second validation path.
validate_train_config = validate_config
