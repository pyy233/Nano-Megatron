from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from nano_megatron.training import Trainer, TrainerState  # noqa: E402
from nano_megatron.training.trainer import _number_microbatches  # noqa: E402


def test_trainer_state_round_trip() -> None:
    state = TrainerState(
        step=3,
        consumed_samples=48,
        consumed_tokens=96,
        data_fingerprint="corpus-v1",
        data_epoch=2,
        data_shuffle_seed=17,
        data_sample_offset=8,
        data_samples_per_epoch=40,
        lr_scheduler={"completed_steps": 3},
        wandb_run_id="run-123",
        metrics_history={"version": 1, "last_sequence": 4, "last_step": 3},
        best_validation_loss=2.5,
        best_validation_step=2,
        best_validation_checkpoint="checkpoints/run/step_00000002",
    )
    restored = TrainerState()
    restored.load_state_dict(state.state_dict())
    assert restored == state


def test_trainer_state_rejects_partial_or_invalid_best_validation_state() -> None:
    with pytest.raises(ValueError, match="requires best validation metrics"):
        TrainerState().load_state_dict(
            {"best_validation_checkpoint": "step_00000001"}
        )
    with pytest.raises(ValueError, match="best_validation_loss"):
        TrainerState().load_state_dict(
            {
                "best_validation_loss": float("nan"),
                "best_validation_step": 1,
            }
        )


def test_bind_data_iterator_rejects_checkpoint_data_identity_change() -> None:
    trainer = object.__new__(Trainer)
    trainer.state = TrainerState(data_fingerprint="original")
    trainer.data_iterator = None

    with pytest.raises(RuntimeError, match="does not match the checkpoint"):
        trainer.bind_data_iterator(iter([1]), data_fingerprint="replacement")

    with pytest.raises(RuntimeError, match="rebinding requires"):
        trainer.bind_data_iterator(iter([1]))

    trainer.bind_data_iterator(iter([2]), data_fingerprint="original")
    assert next(trainer.data_iterator) == 2


def test_fit_cannot_bypass_checkpoint_data_fingerprint_with_inline_data() -> None:
    trainer = object.__new__(Trainer)
    trainer.state = TrainerState(data_fingerprint="original")

    with pytest.raises(RuntimeError, match="bind_data_iterator"):
        trainer.fit(["unverified"], max_steps=1)


def test_restore_data_position_skips_completed_optimizer_steps() -> None:
    trainer = object.__new__(Trainer)
    trainer.state = TrainerState(step=2)
    trainer.data_iterator = iter(["step-0", "step-1", "step-2"])
    trainer._data_batches_consumed = 0
    trainer.batch_router = SimpleNamespace(is_source=True, data_replica_count=1)
    trainer.config = SimpleNamespace(
        training=SimpleNamespace(micro_batch_size=1, gradient_accumulation_steps=1)
    )

    trainer.restore_data_position()

    assert next(trainer.data_iterator) == "step-2"
    assert trainer._data_batches_consumed == 2


def test_restore_data_position_reports_exhausted_iterator() -> None:
    trainer = object.__new__(Trainer)
    trainer.state = TrainerState(step=2)
    trainer.data_iterator = iter(["only-one-batch"])
    trainer._data_batches_consumed = 0
    trainer.batch_router = SimpleNamespace(is_source=True, data_replica_count=1)
    trainer.config = SimpleNamespace(
        training=SimpleNamespace(micro_batch_size=1, gradient_accumulation_steps=1)
    )

    with pytest.raises(RuntimeError, match="ended before"):
        trainer.restore_data_position()


def test_restore_data_position_uses_consumed_samples_after_replica_reshard() -> None:
    trainer = object.__new__(Trainer)
    trainer.state = TrainerState(step=6, consumed_samples=24)
    trainer.data_iterator = iter(range(10))
    trainer._data_batches_consumed = 0
    trainer.batch_router = SimpleNamespace(is_source=True, data_replica_count=2)
    trainer.config = SimpleNamespace(
        training=SimpleNamespace(micro_batch_size=2, gradient_accumulation_steps=2)
    )

    trainer.restore_data_position()

    # 24 global samples / (2 current replicas * 4 local samples) = 3 batches.
    assert next(trainer.data_iterator) == 3


def test_microbatch_numbers_can_continue_across_optimizer_steps() -> None:
    numbered = _number_microbatches(
        ({"value": 1}, {"value": 2}),
        start_index=6,
    )
    assert [microbatch["_microbatch_index"] for microbatch in numbered] == [6, 7]


def test_runtime_sequence_validation_supports_dynamic_lengths() -> None:
    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(
        model=SimpleNamespace(seq_length=8),
        pipeline=SimpleNamespace(dynamic_activation_shapes=True),
        parallel=SimpleNamespace(sequence_parallel=False),
    )
    trainer.parallel = SimpleNamespace(
        cp=SimpleNamespace(size=2),
        tp=SimpleNamespace(size=1),
    )

    assert (
        trainer._validate_runtime_sequence({"input_ids": torch.zeros(2, 3, dtype=torch.long)}) == 6
    )


def test_runtime_sequence_validation_rejects_static_or_invalid_sp_shapes() -> None:
    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(
        model=SimpleNamespace(seq_length=8),
        pipeline=SimpleNamespace(dynamic_activation_shapes=False),
        parallel=SimpleNamespace(sequence_parallel=False),
    )
    trainer.parallel = SimpleNamespace(
        cp=SimpleNamespace(size=2),
        tp=SimpleNamespace(size=2),
    )
    batch = {"input_ids": torch.zeros(2, 3, dtype=torch.long)}
    with pytest.raises(ValueError, match="differs from model.seq_length"):
        trainer._validate_runtime_sequence(batch)

    trainer.config.pipeline.dynamic_activation_shapes = True
    trainer.config.parallel.sequence_parallel = True
    with pytest.raises(ValueError, match="divisible by TP"):
        trainer._validate_runtime_sequence(batch)
