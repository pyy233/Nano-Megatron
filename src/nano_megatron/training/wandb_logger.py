"""Optional, rank-gated Weights & Biases telemetry."""

from __future__ import annotations

import importlib
import uuid
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .metric_history import JsonlMetricHistory


def _plain_config(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain_config(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain_config(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_config(item) for item in value]
    if isinstance(value, (Enum, Path)):
        return str(getattr(value, "value", value))
    return value


class WandbLogger:
    """Keep W&B out of the core training dependency and non-primary ranks."""

    def __init__(
        self,
        config: Any | None,
        *,
        run_config: Any,
        is_primary: bool,
        wandb_module: Any | None = None,
        metrics_path: Path | None = None,
        resume_metrics_path: Path | None = None,
    ) -> None:
        self.config = config
        self.run_config = run_config
        self.enabled = bool(
            config is not None
            and config.enabled
            and config.mode != "disabled"
            and is_primary
        )
        self._wandb = wandb_module
        self._run: Any | None = None
        self.run_id: str | None = None
        self.history = JsonlMetricHistory(
            metrics_path,
            source=resume_metrics_path,
            enabled=is_primary,
        )

    def start(
        self,
        *,
        checkpoint_run_id: str | None = None,
        checkpoint_step: int | None = None,
        checkpoint_metrics_state: Mapping[str, Any] | None = None,
        model: Any | None = None,
    ) -> str | None:
        """Start local history and, on restore, replay it into a normal new W&B run."""

        if self._run is not None:
            return self.run_id
        is_restore = checkpoint_step is not None
        if checkpoint_run_id is not None and not is_restore:
            raise ValueError("checkpoint_run_id requires checkpoint_step")
        if is_restore and (
            isinstance(checkpoint_step, bool)
            or not isinstance(checkpoint_step, int)
            or checkpoint_step < 0
        ):
            raise ValueError("checkpoint_step must be a non-negative integer on restore")
        replay_records = self.history.start(
            through_step=checkpoint_step,
            expected_state=checkpoint_metrics_state,
        )
        if not self.enabled:
            return None

        module = self._load_module()
        configured_id = self.config.run_id
        if not is_restore:
            run_id = configured_id or self._generate_id(module)
        else:
            run_id = (
                configured_id
                if configured_id is not None and configured_id != checkpoint_run_id
                else self._generate_id(module)
            )
            if run_id == checkpoint_run_id:
                run_id = uuid.uuid4().hex
        options = {
            "project": self.config.project,
            "entity": self.config.entity,
            "name": self.config.name,
            "group": self.config.group,
            "tags": list(self.config.tags),
            "mode": self.config.mode,
            "dir": None if self.config.directory is None else str(self.config.directory),
            "id": run_id,
            "config": _plain_config(self.run_config),
        }
        options["resume"] = "never" if is_restore else "allow"
        self._run = module.init(
            **{key: value for key, value in options.items() if value is not None}
        )
        if self._run is None:
            raise RuntimeError("wandb.init() returned no run")
        actual_id = getattr(self._run, "id", run_id)
        if not isinstance(actual_id, str) or not actual_id:
            raise RuntimeError("W&B run must expose a non-empty string id")
        self.run_id = actual_id
        define_metric = getattr(module, "define_metric", None)
        if callable(define_metric):
            define_metric("trainer/step")
            define_metric("train/*", step_metric="trainer/step")
            define_metric("validation/*", step_metric="trainer/step")
        for record in replay_records:
            self._run.log(record.wandb_payload())
        if is_restore:
            lineage: dict[str, Any] = {
                "wandb/logical_fork": True,
                "wandb/forked_from_step": checkpoint_step,
                "wandb/history_replayed_records": len(replay_records),
            }
            if checkpoint_run_id is not None:
                lineage["wandb/forked_from_run_id"] = checkpoint_run_id
            self.update_summary(lineage)
        if self.history.destination is not None:
            self.update_summary(
                {"metrics/history_path": str(self.history.destination)}
            )
        if model is not None:
            self.update_summary(
                {
                    "model/parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "model/trainable_parameters": sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ),
                }
            )
        return self.run_id

    def log(
        self,
        prefix: str,
        metrics: Mapping[str, float],
        *,
        state: Any,
    ) -> None:
        record = self.history.append(prefix, metrics, state=state)
        if self._run is None:
            return
        payload: dict[str, Any] = (
            record.wandb_payload()
            if record is not None
            else {
                "trainer/step": int(state.step),
                "trainer/consumed_samples": int(state.consumed_samples),
                "trainer/consumed_tokens": int(state.consumed_tokens),
                "data/epoch": int(state.data_epoch),
                "data/sample_offset": int(state.data_sample_offset),
                "data/shuffle_seed": state.data_shuffle_seed,
            }
        )
        if record is None:
            payload.update(
                {f"{prefix}/{name}": float(value) for name, value in metrics.items()}
            )
        self._run.log(payload)

    def history_state(self) -> dict[str, int | None] | None:
        return self.history.state_dict()

    def flush(self) -> None:
        self.history.flush()

    def update_summary(self, values: Mapping[str, Any]) -> None:
        if self._run is None:
            return
        summary = getattr(self._run, "summary", None)
        update = getattr(summary, "update", None)
        if callable(update):
            update(dict(values))

    def finish(self) -> None:
        try:
            if self._run is not None:
                finish = getattr(self._run, "finish", None)
                if callable(finish):
                    finish()
        finally:
            self._run = None
            self.history.close()

    def _load_module(self) -> Any:
        if self._wandb is not None:
            return self._wandb
        try:
            self._wandb = importlib.import_module("wandb")
        except ImportError as error:
            raise ImportError(
                "wandb.enabled=true requires the optional tracking dependency; "
                "install nano-megatron[tracking]"
            ) from error
        return self._wandb

    @staticmethod
    def _generate_id(module: Any) -> str:
        utility = getattr(module, "util", None)
        generate_id = getattr(utility, "generate_id", None)
        if callable(generate_id):
            value = generate_id()
            if isinstance(value, str) and value:
                return value
        return uuid.uuid4().hex


__all__ = ["WandbLogger"]
