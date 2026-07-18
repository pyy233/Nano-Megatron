from __future__ import annotations

import json
from pathlib import Path

import torch

from nano_megatron.cli.train import run
from nano_megatron.training import read_metric_history


def test_cli_runs_periodic_validation_without_consuming_training_data(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = Path(__file__).parents[2] / "examples" / "configs" / "gpt_single_cpu.yaml"
    validation_tokens = tmp_path / "validation.pt"
    torch.save(torch.arange(65) % 32, validation_tokens)

    state = run(
        config_path,
        overrides=(
            "model.layers=1",
            "model.hidden_size=16",
            "model.ffn_hidden_size=32",
            "model.heads=4",
            "model.seq_length=4",
            "model.vocab_size=32",
            "training.micro_batch_size=1",
            "training.gradient_accumulation_steps=1",
            "training.max_steps=1",
            "training.log_interval=1",
            "validation.interval=1",
            "validation.batches=2",
            f"validation.data.path={validation_tokens}",
            "validation.data.shuffle=false",
            "checkpoint.save_interval=0",
            f"checkpoint.directory={tmp_path / 'checkpoints'}",
        ),
        resume=None,
        max_steps=1,
    )

    output = capsys.readouterr().out
    assert state.step == 1
    assert state.consumed_samples == 1
    assert "train step=1" in output
    assert "validation step=1" in output
    assert "perplexity=" in output
    metric_paths = list((tmp_path / "checkpoints").glob("*/metrics.jsonl"))
    assert len(metric_paths) == 1
    records = read_metric_history(metric_paths[0])
    assert [(record.step, record.prefix) for record in records] == [
        (1, "train"),
        (1, "validation"),
    ]


def test_cli_saves_non_interval_final_and_pins_best_validation_checkpoint(
    tmp_path: Path,
) -> None:
    config_path = Path(__file__).parents[2] / "examples" / "configs" / "gpt_single_cpu.yaml"
    validation_tokens = tmp_path / "validation.pt"
    torch.save(torch.arange(65) % 32, validation_tokens)
    checkpoint_base = tmp_path / "checkpoints"

    state = run(
        config_path,
        overrides=(
            "model.layers=1",
            "model.hidden_size=16",
            "model.ffn_hidden_size=32",
            "model.heads=4",
            "model.seq_length=4",
            "model.vocab_size=32",
            "training.micro_batch_size=1",
            "training.gradient_accumulation_steps=1",
            "training.max_steps=2",
            "training.log_interval=1",
            "validation.interval=1",
            "validation.batches=2",
            f"validation.data.path={validation_tokens}",
            "validation.data.shuffle=false",
            "checkpoint.save_interval=100",
            "checkpoint.keep_last=1",
            f"checkpoint.directory={checkpoint_base}",
        ),
        resume=None,
        max_steps=2,
    )

    run_directory = next(checkpoint_base.iterdir())
    pin = json.loads((run_directory / "best_validation.json").read_text(encoding="utf-8"))
    best_checkpoint = run_directory / pin["checkpoint"]
    final_checkpoint = run_directory / "step_00000002"
    assert state.step == 2
    assert state.best_validation_loss == pin["value"]
    assert state.best_validation_step == pin["step"]
    assert state.best_validation_checkpoint == str(best_checkpoint)
    assert (best_checkpoint / ".complete").is_file()
    assert (final_checkpoint / ".complete").is_file()
