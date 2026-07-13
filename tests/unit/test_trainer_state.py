from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from nano_megatron.training import Trainer, TrainerState  # noqa: E402
from nano_megatron.training.trainer import _number_microbatches  # noqa: E402


def test_trainer_state_round_trip() -> None:
    state = TrainerState(step=3, consumed_samples=48, consumed_tokens=96)
    restored = TrainerState()
    restored.load_state_dict(state.state_dict())
    assert restored == state


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
