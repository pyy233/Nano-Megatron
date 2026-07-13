from __future__ import annotations

import itertools

import pytest

from nano_megatron.config import (
    ActivationCheckpointConfig,
    CheckpointConfig,
    ContextParallelBackend,
    ContextParallelConfig,
    DataParallelConfig,
    DataParallelMode,
    GPTConfig,
    OffloadConfig,
    ParallelConfig,
    PipelineConfig,
    PrecisionConfig,
    TrainConfig,
    TrainingConfig,
    validate_config,
)
from nano_megatron.config.validation import ConfigValidationError
from nano_megatron.parallel import ParallelAxis


def small_config(**overrides: object) -> TrainConfig:
    values = {
        "parallel": ParallelConfig(data=1),
        "model": GPTConfig(
            layers=2,
            hidden_size=16,
            ffn_hidden_size=32,
            heads=4,
            kv_heads=2,
            seq_length=8,
            vocab_size=17,
        ),
    }
    values.update(overrides)
    return TrainConfig(**values)


def test_parallel_config_normalizes_order_and_infers_data_axis() -> None:
    config = ParallelConfig(
        tensor=2,
        pipeline=2,
        context=2,
        expert=2,
        data=None,
        order=("pp", "dp", "ep", "cp", "tp"),
    )

    assert config.order == (
        ParallelAxis.PP,
        ParallelAxis.DP,
        ParallelAxis.EP,
        ParallelAxis.CP,
        ParallelAxis.TP,
    )
    assert config.resolve_data_parallel_size(48) == 3
    assert config.resolved(48).data == 3


@pytest.mark.parametrize(
    "order",
    [
        ("tp", "cp", "ep", "dp"),
        ("tp", "cp", "ep", "dp", "dp"),
        ("tp", "cp", "ep", "dp", "unknown"),
    ],
)
def test_parallel_config_rejects_invalid_rank_orders(order: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="rank order|parallel axis"):
        ParallelConfig(data=1, order=order)


def test_data_parallel_mode_is_coerced_when_constructed_directly() -> None:
    config = DataParallelConfig(mode="zero2")
    assert config.mode is DataParallelMode.ZERO2


def test_context_parallel_config_is_separate_from_topology_and_dropout_is_explicit() -> None:
    config = ContextParallelConfig(backend="ring")
    assert config.backend is ContextParallelBackend.RING

    with pytest.raises(ValueError, match="dropout must be 0"):
        ContextParallelConfig(dropout=0.1)


def test_gpt_config_exposes_readable_aliases_and_vocab_padding() -> None:
    config = GPTConfig(
        layers=3,
        hidden_size=24,
        ffn_hidden_size=48,
        heads=6,
        kv_heads=2,
        seq_length=16,
        vocab_size=101,
    )

    assert config.num_layers == 3
    assert config.num_attention_heads == 6
    assert config.num_kv_heads == 2
    assert config.head_dim == 4
    assert config.padded_vocab_size(8) == 104


def test_validation_derives_replica_batch_and_padded_vocab() -> None:
    config = small_config(
        parallel=ParallelConfig(tensor=2, context=2, expert=2, data=None),
        training=TrainingConfig(micro_batch_size=3, gradient_accumulation_steps=4),
    )

    result = validate_config(config, world_size=16)

    assert result.data_parallel_size == 2
    assert result.global_batch_size == 3 * 4 * 2 * 2
    assert result.padded_vocab_size == 18
    assert any("EP>1" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    ("parallel", "world_size", "message"),
    [
        (ParallelConfig(tensor=2, data=2), 2, "world size does not match"),
        (ParallelConfig(tensor=2, data=None), 3, "not divisible"),
    ],
)
def test_validation_rejects_world_size_mismatch(
    parallel: ParallelConfig,
    world_size: int,
    message: str,
) -> None:
    with pytest.raises(ConfigValidationError, match=message):
        validate_config(small_config(parallel=parallel), world_size=world_size)


@pytest.mark.parametrize("params", ["bfloat16", "float16"])
def test_validation_rejects_low_precision_params_with_float32_compute(
    params: str,
) -> None:
    config = small_config(
        precision=PrecisionConfig(params=params, compute="float32"),
    )

    with pytest.raises(
        ConfigValidationError,
        match=r"precision\.compute=float32 requires precision\.params=float32",
    ):
        validate_config(config, world_size=1)


@pytest.mark.parametrize("mode", ["ddp", "zero3"])
def test_validation_rejects_float16_params_without_fp32_master_parameters(
    mode: str,
) -> None:
    config = small_config(
        precision=PrecisionConfig(params="float16", compute="float16"),
        data_parallel=DataParallelConfig(mode=mode),
    )

    with pytest.raises(
        ConfigValidationError,
        match=r"precision\.params=float16.*do not maintain FP32 master parameters",
    ):
        validate_config(config, world_size=1)


@pytest.mark.parametrize("mode", ["zero1", "zero2"])
def test_validation_allows_float16_params_with_fp32_master_shards(mode: str) -> None:
    config = small_config(
        precision=PrecisionConfig(params="float16", compute="float16"),
        data_parallel=DataParallelConfig(mode=mode),
    )

    validate_config(config, world_size=1)


def test_validation_rejects_cross_block_invariants() -> None:
    config = small_config(
        parallel=ParallelConfig(tensor=4, pipeline=3, context=2, data=1),
        model=GPTConfig(
            layers=2,
            hidden_size=24,
            ffn_hidden_size=30,
            heads=6,
            kv_heads=2,
            seq_length=10,
            vocab_size=32,
        ),
    )

    with pytest.raises(ConfigValidationError) as captured:
        validate_config(config, world_size=24)

    message = str(captured.value)
    assert "ffn_hidden_size" in message
    assert "model.heads" in message
    assert "model.kv_heads" in message
    assert "model.layers" in message


def test_validation_rejects_offload_modes_that_do_not_own_the_lifecycle() -> None:
    config = small_config(
        data_parallel=DataParallelConfig(mode="ddp"),
        offload=OffloadConfig(optimizer_state=True, zero3_params_and_grads=True),
    )

    with pytest.raises(ConfigValidationError) as captured:
        validate_config(config, world_size=1)

    assert "optimizer-state offload" in str(captured.value)
    assert "zero3_params_and_grads" in str(captured.value)


def test_all_rank_order_permutations_are_accepted() -> None:
    axes = tuple(axis.value for axis in ParallelAxis)
    for order in itertools.permutations(axes):
        assert ParallelConfig(data=1, order=order).order[0].value == order[0]


def test_validation_rejects_declared_but_unimplemented_async_options() -> None:
    config = small_config(
        data_parallel=DataParallelConfig(overlap_grad_reduce=True),
        pipeline=PipelineConfig(overlap_p2p=True),
        checkpoint=CheckpointConfig(async_save=True),
    )

    with pytest.raises(ConfigValidationError) as captured:
        validate_config(config, world_size=1)

    message = str(captured.value)
    assert "overlap_grad_reduce" in message
    assert "overlap_p2p" in message
    assert "async_save" in message


def test_checkpoint_save_interval_zero_disables_periodic_saves() -> None:
    assert CheckpointConfig(save_interval=0).save_interval == 0


def test_validation_rejects_checkpoint_offload_when_checkpointing_is_disabled() -> None:
    config = small_config(
        activation_checkpoint=ActivationCheckpointConfig(
            mode="none",
            offload_saved_tensors=True,
        )
    )

    with pytest.raises(ConfigValidationError, match="requires activation checkpointing"):
        validate_config(config, world_size=1)


def test_validation_accepts_periodic_checkpointing_for_distributed_zero3() -> None:
    config = small_config(
        parallel=ParallelConfig(data=2),
        data_parallel=DataParallelConfig(mode="zero3"),
    )
    assert validate_config(config, world_size=2).data_parallel_size == 2


def test_validation_rejects_zero3_offload_on_a_size_one_replica_mesh() -> None:
    config = small_config(
        data_parallel=DataParallelConfig(mode="zero3"),
        offload=OffloadConfig(zero3_params_and_grads=True),
        checkpoint=CheckpointConfig(save_interval=0),
    )

    with pytest.raises(ConfigValidationError, match="active ZeRO-3 replica mesh"):
        validate_config(config, world_size=1)


def test_validation_rejects_model_dropout_with_context_parallelism() -> None:
    config = small_config(
        parallel=ParallelConfig(context=2, data=1),
        model=GPTConfig(
            layers=2,
            hidden_size=16,
            ffn_hidden_size=32,
            heads=4,
            kv_heads=2,
            seq_length=8,
            vocab_size=17,
            dropout=0.1,
        ),
    )

    with pytest.raises(ConfigValidationError, match="model.dropout must be 0"):
        validate_config(config, world_size=2)
