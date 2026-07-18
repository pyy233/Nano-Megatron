from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from nano_megatron.cli.train import run
from nano_megatron.training import WandbLogger, read_metric_history


class _Run:
    def __init__(self, run_id: str) -> None:
        self.id = run_id
        self.summary: dict[str, object] = {}
        self.logs: list[dict[str, object]] = []
        self.finished = False

    def log(self, values) -> None:
        self.logs.append(dict(values))

    def finish(self) -> None:
        self.finished = True


class _Wandb:
    def __init__(self) -> None:
        self.init_calls: list[dict[str, object]] = []
        self.runs: list[_Run] = []
        generated_ids = iter(("initial-run-id", "forked-run-id"))
        self.util = SimpleNamespace(generate_id=lambda: next(generated_ids))

    def init(self, **options):
        self.init_calls.append(dict(options))
        created = _Run(str(options["id"]))
        self.runs.append(created)
        return created

    @staticmethod
    def define_metric(*args, **kwargs) -> None:
        del args, kwargs


def test_wandb_restore_replays_jsonl_into_logical_child_and_checkpoints_child_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_wandb = _Wandb()
    monkeypatch.setattr(
        WandbLogger,
        "_load_module",
        lambda self: fake_wandb,
    )
    config_path = Path(__file__).parents[2] / "examples" / "configs" / "gpt_single_cpu.yaml"
    checkpoints = tmp_path / "checkpoints"
    validation_tokens = tmp_path / "validation.pt"
    torch.save(torch.arange(65) % 32, validation_tokens)
    overrides = (
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
        "checkpoint.save_interval=1",
        f"checkpoint.directory={checkpoints}",
        "validation.interval=1",
        "validation.batches=2",
        f"validation.data.path={validation_tokens}",
        "validation.data.shuffle=false",
        "wandb.enabled=true",
        "wandb.mode=offline",
        "wandb.project=nano-megatron-tests",
    )

    first = run(config_path, overrides=overrides, resume=None, max_steps=1)
    checkpoint = next(checkpoints.glob("*/step_00000001"))
    resumed = run(config_path, overrides=overrides, resume=checkpoint, max_steps=2)
    resumed_checkpoint = next(checkpoints.glob("*/step_00000002"))

    assert first.wandb_run_id == "initial-run-id"
    assert resumed.wandb_run_id == "forked-run-id"
    assert [call["id"] for call in fake_wandb.init_calls] == [
        "initial-run-id",
        "forked-run-id",
    ]
    assert fake_wandb.init_calls[0]["resume"] == "allow"
    assert "fork_from" not in fake_wandb.init_calls[0]
    assert fake_wandb.init_calls[1]["resume"] == "never"
    assert "fork_from" not in fake_wandb.init_calls[1]
    assert all(run.finished for run in fake_wandb.runs)
    assert fake_wandb.runs[0].logs[0]["train/loss"] > 0.0
    assert fake_wandb.runs[0].logs[1]["validation/loss"] > 0.0
    assert fake_wandb.runs[0].logs[1]["validation/perplexity"] > 0.0
    assert [payload["trainer/step"] for payload in fake_wandb.runs[1].logs] == [
        1,
        1,
        2,
        2,
    ]
    assert fake_wandb.runs[1].logs[0]["train/loss"] > 0.0
    assert fake_wandb.runs[1].logs[1]["validation/loss"] > 0.0
    assert fake_wandb.runs[1].logs[2]["trainer/step"] == 2
    assert fake_wandb.runs[1].summary["wandb/logical_fork"] is True
    assert fake_wandb.runs[1].summary["wandb/forked_from_run_id"] == "initial-run-id"
    assert fake_wandb.runs[1].summary["wandb/forked_from_step"] == 1
    assert fake_wandb.runs[1].summary["wandb/history_replayed_records"] == 2
    saved = torch.load(checkpoint / "trainer_state.pt", map_location="cpu", weights_only=True)
    resumed_saved = torch.load(
        resumed_checkpoint / "trainer_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert saved["wandb_run_id"] == "initial-run-id"
    assert resumed_saved["wandb_run_id"] == "forked-run-id"
    assert saved["metrics_history"] == {
        "version": 1,
        "last_sequence": 1,
        "last_step": 1,
    }
    assert resumed_saved["metrics_history"] == {
        "version": 1,
        "last_sequence": 3,
        "last_step": 2,
    }
    source_records = read_metric_history(checkpoint.parent / "metrics.jsonl")
    child_records = read_metric_history(resumed_checkpoint.parent / "metrics.jsonl")
    assert [(record.step, record.prefix) for record in source_records] == [
        (1, "train"),
        (1, "validation"),
    ]
    assert [(record.step, record.prefix) for record in child_records] == [
        (1, "train"),
        (1, "validation"),
        (2, "train"),
        (2, "validation"),
    ]
    assert checkpoint.parent != resumed_checkpoint.parent
