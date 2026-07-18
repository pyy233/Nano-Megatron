from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from nano_megatron.config import TrainConfig, WandbConfig
from nano_megatron.training import JsonlMetricHistory, WandbLogger, read_metric_history


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
    util = SimpleNamespace(generate_id=lambda: "generated-run")

    def __init__(self) -> None:
        self.init_options: dict[str, object] | None = None
        self.metrics: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.run: _Run | None = None

    def init(self, **options):
        self.init_options = dict(options)
        self.run = _Run(str(options["id"]))
        return self.run

    def define_metric(self, *args, **kwargs) -> None:
        self.metrics.append((args, kwargs))


def test_disabled_or_non_primary_logger_never_initializes_wandb(
    tmp_path: Path,
) -> None:
    module = _Wandb()
    non_primary_path = tmp_path / "non-primary" / "metrics.jsonl"
    disabled = WandbLogger(
        WandbConfig(enabled=False),
        run_config=TrainConfig(),
        is_primary=True,
        wandb_module=module,
    )
    non_primary = WandbLogger(
        WandbConfig(enabled=True),
        run_config=TrainConfig(),
        is_primary=False,
        wandb_module=module,
        metrics_path=non_primary_path,
    )
    disabled_mode = WandbLogger(
        WandbConfig(enabled=True, mode="disabled"),
        run_config=TrainConfig(),
        is_primary=True,
        wandb_module=module,
    )

    assert disabled.start() is None
    assert non_primary.start() is None
    assert disabled_mode.start() is None
    non_primary.log(
        "train",
        {"loss": 2.0},
        state=SimpleNamespace(
            step=1,
            consumed_samples=4,
            consumed_tokens=128,
            data_epoch=0,
            data_sample_offset=4,
            data_shuffle_seed=1234,
        ),
    )
    non_primary.finish()
    assert module.init_options is None
    assert not non_primary_path.exists()


def test_logger_replays_local_history_into_logical_child_and_records_lineage(
    tmp_path: Path,
) -> None:
    module = _Wandb()
    source_path = tmp_path / "source" / "metrics.jsonl"
    source = JsonlMetricHistory(source_path)
    source.start(through_step=None)
    checkpoint_state = SimpleNamespace(
        step=7,
        consumed_samples=56,
        consumed_tokens=448,
        data_epoch=2,
        data_sample_offset=16,
        data_shuffle_seed=1234,
    )
    source.append("train", {"loss": 1.75}, state=checkpoint_state)
    source_state = source.state_dict()
    source.close()
    child_path = tmp_path / "child" / "metrics.jsonl"
    logger = WandbLogger(
        WandbConfig(
            enabled=True,
            project="tests",
            mode="offline",
            tags=("tiny", "cpu"),
        ),
        run_config=TrainConfig(),
        is_primary=True,
        wandb_module=module,
        metrics_path=child_path,
        resume_metrics_path=source_path,
    )
    model = SimpleNamespace(
        parameters=lambda: iter(
            [
                SimpleNamespace(numel=lambda: 10, requires_grad=True),
                SimpleNamespace(numel=lambda: 2, requires_grad=False),
            ]
        )
    )

    run_id = logger.start(
        checkpoint_run_id="checkpoint-run",
        checkpoint_step=7,
        checkpoint_metrics_state=source_state,
        model=model,
    )
    state = SimpleNamespace(
        step=8,
        consumed_samples=64,
        consumed_tokens=512,
        data_epoch=2,
        data_sample_offset=24,
        data_shuffle_seed=1234,
    )
    logger.log("train", {"loss": 1.5, "learning_rate": 2.0e-4}, state=state)
    logger.update_summary({"checkpoint/step": 7})
    assert module.run is not None
    run = module.run
    logger.finish()

    assert run_id == "generated-run"
    assert module.init_options is not None
    assert module.init_options["id"] == "generated-run"
    assert module.init_options["resume"] == "never"
    assert "fork_from" not in module.init_options
    assert module.init_options["mode"] == "offline"
    assert module.init_options["tags"] == ["tiny", "cpu"]
    assert run.summary["wandb/forked_from_run_id"] == "checkpoint-run"
    assert run.summary["wandb/forked_from_step"] == 7
    assert run.summary["wandb/logical_fork"] is True
    assert run.summary["wandb/history_replayed_records"] == 1
    assert run.summary["metrics/history_path"] == str(child_path)
    assert run.summary["model/parameters"] == 12
    assert run.summary["model/trainable_parameters"] == 10
    assert run.summary["checkpoint/step"] == 7
    replay_payload = run.logs[0]
    payload = run.logs[1]
    assert replay_payload["trainer/step"] == 7
    assert replay_payload["train/loss"] == 1.75
    assert payload["trainer/step"] == 8
    assert payload["trainer/consumed_tokens"] == 512
    assert payload["data/epoch"] == 2
    assert payload["train/loss"] == 1.5
    assert run.finished
    records = read_metric_history(child_path)
    assert [(record.sequence, record.step) for record in records] == [(0, 7), (1, 8)]


def test_logger_uses_configured_run_id_for_new_fork() -> None:
    module = _Wandb()
    logger = WandbLogger(
        WandbConfig(enabled=True, run_id="configured"),
        run_config=TrainConfig(),
        is_primary=True,
        wandb_module=module,
    )

    assert logger.start(checkpoint_run_id="checkpoint", checkpoint_step=3) == "configured"
    assert module.init_options is not None
    assert module.init_options["resume"] == "never"
    assert "fork_from" not in module.init_options


def test_logger_replaces_source_run_id_and_rejects_missing_step_for_fork() -> None:
    module = _Wandb()
    same_id = WandbLogger(
        WandbConfig(enabled=True, run_id="checkpoint"),
        run_config=TrainConfig(),
        is_primary=True,
        wandb_module=module,
    )
    missing_step = WandbLogger(
        WandbConfig(enabled=True),
        run_config=TrainConfig(),
        is_primary=True,
        wandb_module=_Wandb(),
    )

    assert same_id.start(checkpoint_run_id="checkpoint", checkpoint_step=3) == "generated-run"
    assert module.init_options is not None
    assert module.init_options["resume"] == "never"
    assert "fork_from" not in module.init_options
    with pytest.raises(ValueError, match="checkpoint_step"):
        missing_step.start(checkpoint_run_id="checkpoint")


def test_fresh_logger_retains_resume_allow_initialization() -> None:
    module = _Wandb()
    logger = WandbLogger(
        WandbConfig(enabled=True),
        run_config=TrainConfig(),
        is_primary=True,
        wandb_module=module,
    )

    assert logger.start() == "generated-run"
    assert module.init_options is not None
    assert module.init_options["resume"] == "allow"
    assert "fork_from" not in module.init_options


def test_disabled_wandb_still_writes_framework_metric_history(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    logger = WandbLogger(
        WandbConfig(enabled=False),
        run_config=TrainConfig(),
        is_primary=True,
        wandb_module=_Wandb(),
        metrics_path=path,
    )
    state = SimpleNamespace(
        step=1,
        consumed_samples=4,
        consumed_tokens=128,
        data_epoch=0,
        data_sample_offset=4,
        data_shuffle_seed=1234,
    )

    assert logger.start() is None
    logger.log("train", {"loss": 2.0}, state=state)
    assert logger.history_state() == {
        "version": 1,
        "last_sequence": 0,
        "last_step": 1,
    }
    logger.finish()

    records = read_metric_history(path)
    assert len(records) == 1
    assert records[0].metrics["loss"] == 2.0
