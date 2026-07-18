from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from nano_megatron.cli import train as train_cli
from nano_megatron.cli.train import run
from nano_megatron.training import read_metric_history


class _PrimaryRuntime:
    is_primary = True

    @staticmethod
    def all_gather_object(value):
        return [value]


class _FollowerRuntime:
    is_primary = False

    def __init__(self, coordinator_directory: Path) -> None:
        self.coordinator_directory = coordinator_directory

    def all_gather_object(self, value):
        assert value is None
        return [(str(self.coordinator_directory), None), value]


def test_checkpoint_run_directory_is_unique_and_published_from_rank_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path / "checkpoints"
    monkeypatch.setattr(
        train_cli,
        "_checkpoint_run_timestamp",
        lambda: "20260718-010203-456789",
    )

    first = train_cli._allocate_checkpoint_run_directory(base, _PrimaryRuntime())
    second = train_cli._allocate_checkpoint_run_directory(base, _PrimaryRuntime())
    follower = train_cli._allocate_checkpoint_run_directory(
        base,
        _FollowerRuntime(first),
    )

    assert first == base / "20260718-010203-456789"
    assert second == base / "20260718-010203-456789-01"
    assert first.is_dir()
    assert second.is_dir()
    assert follower == first


def test_cli_resume_restores_state_and_continues_to_absolute_stop_step(
    tmp_path: Path,
) -> None:
    config_path = Path(__file__).parents[2] / "examples" / "configs" / "gpt_single_cpu.yaml"
    checkpoint_directory = tmp_path / "checkpoints"
    overrides = [
        "model.layers=1",
        "model.hidden_size=16",
        "model.ffn_hidden_size=32",
        "model.heads=4",
        "model.kv_heads=2",
        "model.seq_length=4",
        "model.vocab_size=32",
        "model.dropout=0.2",
        "training.micro_batch_size=1",
        "training.gradient_accumulation_steps=1",
        "training.max_steps=2",
        "data_parallel.mode=zero2",
        "activation_checkpoint.mode=full",
        "activation_checkpoint.block_interval=1",
        "activation_checkpoint.offload_saved_tensors=true",
        "offload.optimizer_state=true",
        "offload.activations=true",
        "checkpoint.save_interval=1",
        f"checkpoint.directory={checkpoint_directory}",
    ]

    first = run(
        config_path,
        overrides=overrides,
        resume=None,
        max_steps=1,
    )
    checkpoints = list(checkpoint_directory.glob("*/step_00000001"))
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    assert first.step == 1
    assert first.data_epoch == 0
    assert first.data_shuffle_seed == 1234
    assert first.data_sample_offset == 1
    assert first.data_samples_per_epoch == 1024
    assert first.lr_scheduler is not None
    assert first.lr_scheduler["completed_steps"] == 1
    assert first.lr_scheduler["learning_rate"] == pytest.approx(5.0e-4)
    assert checkpoint.is_dir()
    saved = torch.load(checkpoint / "trainer_state.pt", map_location="cpu", weights_only=True)
    assert saved["data_epoch"] == 0
    assert saved["data_shuffle_seed"] == 1234
    assert saved["data_sample_offset"] == 1
    assert saved["lr_scheduler"]["completed_steps"] == 1
    assert saved["metrics_history"] == {
        "version": 1,
        "last_sequence": 0,
        "last_step": 1,
    }

    resumed = run(
        config_path,
        overrides=overrides,
        resume=checkpoint,
        max_steps=2,
    )
    resumed_checkpoints = list(checkpoint_directory.glob("*/step_00000002"))
    assert len(resumed_checkpoints) == 1
    resumed_checkpoint = resumed_checkpoints[0]
    assert resumed.step == 2
    assert resumed.consumed_samples == 2
    assert resumed.consumed_tokens == 8
    assert resumed.data_epoch == 0
    assert resumed.data_shuffle_seed == 1234
    assert resumed.data_sample_offset == 2
    assert resumed.lr_scheduler is not None
    assert resumed.lr_scheduler["completed_steps"] == 2
    assert resumed.lr_scheduler["learning_rate"] == 0.0
    source_metrics = read_metric_history(checkpoint.parent / "metrics.jsonl")
    child_metrics = read_metric_history(resumed_checkpoint.parent / "metrics.jsonl")
    assert [(record.sequence, record.step) for record in source_metrics] == [(0, 1)]
    assert [(record.sequence, record.step) for record in child_metrics] == [
        (0, 1),
        (1, 2),
    ]
    resumed_saved = torch.load(
        resumed_checkpoint / "trainer_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert resumed_saved["metrics_history"] == {
        "version": 1,
        "last_sequence": 1,
        "last_step": 2,
    }
    assert checkpoint.parent != resumed_checkpoint.parent
    assert re.fullmatch(r"\d{8}-\d{6}-\d{6}", checkpoint.parent.name)
    assert re.fullmatch(r"\d{8}-\d{6}-\d{6}", resumed_checkpoint.parent.name)
