"""Versioned, replayable JSONL metric history owned by Nano-Megatron."""

from __future__ import annotations

import json
import math
import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

METRIC_HISTORY_VERSION = 1
_NONFINITE_VALUES = {
    "NaN": float("nan"),
    "Infinity": float("inf"),
    "-Infinity": float("-inf"),
}


def _non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _optional_non_negative_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    return _non_negative_int(name, value)


def _metric_value(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"metric {name!r} must be numeric")
    return float(value)


def _encode_metric(value: float) -> float | str:
    if math.isnan(value):
        return "NaN"
    if value == float("inf"):
        return "Infinity"
    if value == float("-inf"):
        return "-Infinity"
    return value


def _decode_metric(name: str, value: Any) -> float:
    if isinstance(value, str):
        if value not in _NONFINITE_VALUES:
            raise ValueError(f"metric {name!r} has an invalid string value {value!r}")
        return _NONFINITE_VALUES[value]
    return _metric_value(name, value)


@dataclass(frozen=True, slots=True)
class MetricHistoryRecord:
    """One train or validation event in the local metric timeline."""

    sequence: int
    step: int
    recorded_at: float
    prefix: str
    metrics: dict[str, float]
    consumed_samples: int
    consumed_tokens: int
    data_epoch: int
    data_sample_offset: int
    data_shuffle_seed: int | None

    @classmethod
    def capture(
        cls,
        *,
        sequence: int,
        prefix: str,
        metrics: Mapping[str, float],
        state: Any,
    ) -> MetricHistoryRecord:
        if not isinstance(prefix, str) or not prefix.strip():
            raise ValueError("metric prefix must be a non-empty string")
        normalized_metrics = {
            str(name): _metric_value(str(name), value) for name, value in metrics.items()
        }
        return cls(
            sequence=_non_negative_int("metric sequence", sequence),
            step=_non_negative_int("metric step", int(state.step)),
            recorded_at=time.time(),
            prefix=prefix.strip(),
            metrics=normalized_metrics,
            consumed_samples=_non_negative_int(
                "consumed_samples",
                int(state.consumed_samples),
            ),
            consumed_tokens=_non_negative_int(
                "consumed_tokens",
                int(state.consumed_tokens),
            ),
            data_epoch=_non_negative_int("data_epoch", int(state.data_epoch)),
            data_sample_offset=_non_negative_int(
                "data_sample_offset",
                int(state.data_sample_offset),
            ),
            data_shuffle_seed=_optional_non_negative_int(
                "data_shuffle_seed",
                state.data_shuffle_seed,
            ),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MetricHistoryRecord:
        version = value.get("version")
        if version != METRIC_HISTORY_VERSION:
            raise ValueError(
                f"unsupported metric history version {version!r}; "
                f"expected {METRIC_HISTORY_VERSION}"
            )
        prefix = value.get("prefix")
        if not isinstance(prefix, str) or not prefix.strip():
            raise ValueError("metric history prefix must be a non-empty string")
        raw_metrics = value.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            raise TypeError("metric history metrics must be a mapping")
        trainer = value.get("trainer")
        data = value.get("data")
        if not isinstance(trainer, Mapping) or not isinstance(data, Mapping):
            raise TypeError("metric history trainer and data state must be mappings")
        recorded_at = value.get("recorded_at")
        if (
            isinstance(recorded_at, bool)
            or not isinstance(recorded_at, (int, float))
            or not math.isfinite(float(recorded_at))
            or float(recorded_at) < 0.0
        ):
            raise ValueError("metric history recorded_at must be a finite non-negative number")
        return cls(
            sequence=_non_negative_int("metric sequence", value.get("sequence")),
            step=_non_negative_int("metric step", value.get("step")),
            recorded_at=float(recorded_at),
            prefix=prefix.strip(),
            metrics={
                str(name): _decode_metric(str(name), metric)
                for name, metric in raw_metrics.items()
            },
            consumed_samples=_non_negative_int(
                "consumed_samples",
                trainer.get("consumed_samples"),
            ),
            consumed_tokens=_non_negative_int(
                "consumed_tokens",
                trainer.get("consumed_tokens"),
            ),
            data_epoch=_non_negative_int("data_epoch", data.get("epoch")),
            data_sample_offset=_non_negative_int(
                "data_sample_offset",
                data.get("sample_offset"),
            ),
            data_shuffle_seed=_optional_non_negative_int(
                "data_shuffle_seed",
                data.get("shuffle_seed"),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": METRIC_HISTORY_VERSION,
            "sequence": self.sequence,
            "step": self.step,
            "recorded_at": self.recorded_at,
            "prefix": self.prefix,
            "metrics": {
                name: _encode_metric(metric) for name, metric in self.metrics.items()
            },
            "trainer": {
                "consumed_samples": self.consumed_samples,
                "consumed_tokens": self.consumed_tokens,
            },
            "data": {
                "epoch": self.data_epoch,
                "sample_offset": self.data_sample_offset,
                "shuffle_seed": self.data_shuffle_seed,
            },
        }

    def to_json_line(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )

    def wandb_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "trainer/step": self.step,
            "trainer/consumed_samples": self.consumed_samples,
            "trainer/consumed_tokens": self.consumed_tokens,
            "data/epoch": self.data_epoch,
            "data/sample_offset": self.data_sample_offset,
            "data/shuffle_seed": self.data_shuffle_seed,
        }
        payload.update(
            {f"{self.prefix}/{name}": value for name, value in self.metrics.items()}
        )
        return payload


def read_metric_history(
    path: Path,
    *,
    through_step: int | None = None,
) -> tuple[MetricHistoryRecord, ...]:
    """Read and validate a metric timeline, optionally selecting a checkpoint prefix."""

    if through_step is not None:
        _non_negative_int("metric history cutoff step", through_step)
    selected: list[MetricHistoryRecord] = []
    previous_sequence = -1
    previous_step = -1
    with path.open("r", encoding="utf-8") as stream:
        line_number = 0
        while True:
            line = stream.readline()
            if not line:
                break
            line_number += 1
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                remainder = stream.read()
                if not line.endswith("\n") and not remainder:
                    warnings.warn(
                        f"ignoring truncated final metric history line {line_number} in {path}",
                        stacklevel=2,
                    )
                    break
                raise ValueError(
                    f"invalid metric history JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, Mapping):
                raise TypeError(f"metric history line {path}:{line_number} must be an object")
            record = MetricHistoryRecord.from_dict(value)
            if record.sequence <= previous_sequence:
                raise ValueError(
                    f"metric history sequences must increase at {path}:{line_number}"
                )
            if record.step < previous_step:
                raise ValueError(f"metric history steps must not decrease at {path}:{line_number}")
            previous_sequence = record.sequence
            previous_step = record.step
            if through_step is None or record.step <= through_step:
                selected.append(record)
    return tuple(selected)


class JsonlMetricHistory:
    """Copy a checkpoint prefix, append new events, and expose records for replay."""

    def __init__(
        self,
        destination: Path | None,
        *,
        source: Path | None = None,
        enabled: bool = True,
    ) -> None:
        self.destination = None if destination is None else Path(destination)
        self.source = None if source is None else Path(source)
        self.enabled = bool(enabled and self.destination is not None)
        self._stream: TextIO | None = None
        self._next_sequence = 0
        self._last_record: MetricHistoryRecord | None = None
        self.replay_records: tuple[MetricHistoryRecord, ...] = ()
        self.source_missing = False

    def start(
        self,
        *,
        through_step: int | None,
        expected_state: Mapping[str, Any] | None = None,
    ) -> tuple[MetricHistoryRecord, ...]:
        if not self.enabled:
            return ()
        if self._stream is not None:
            return self.replay_records
        assert self.destination is not None
        if self.source is not None and through_step is None:
            raise ValueError("resume metric history requires a checkpoint step")
        replay: tuple[MetricHistoryRecord, ...] = ()
        if self.source is not None:
            if self.source.exists():
                if self.source.resolve() == self.destination.resolve():
                    raise ValueError("source and destination metric histories must be distinct")
                replay = read_metric_history(self.source, through_step=through_step)
            else:
                self.source_missing = True
                warnings.warn(
                    f"resume metric history is missing: {self.source}; "
                    "the logical W&B child will start without replayed metrics",
                    stacklevel=2,
                )
        self._validate_expected_state(replay, expected_state)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.destination.open("x", encoding="utf-8", newline="\n")
        for record in replay:
            self._stream.write(record.to_json_line() + "\n")
        self._stream.flush()
        self.replay_records = replay
        if replay:
            self._last_record = replay[-1]
            self._next_sequence = replay[-1].sequence + 1
        return replay

    def append(
        self,
        prefix: str,
        metrics: Mapping[str, float],
        *,
        state: Any,
    ) -> MetricHistoryRecord | None:
        if not self.enabled:
            return None
        if self._stream is None:
            raise RuntimeError("metric history must be started before logging")
        record = MetricHistoryRecord.capture(
            sequence=self._next_sequence,
            prefix=prefix,
            metrics=metrics,
            state=state,
        )
        self._stream.write(record.to_json_line() + "\n")
        self._stream.flush()
        self._next_sequence += 1
        self._last_record = record
        return record

    def state_dict(self) -> dict[str, int | None] | None:
        if not self.enabled:
            return None
        return {
            "version": METRIC_HISTORY_VERSION,
            "last_sequence": (
                None if self._last_record is None else self._last_record.sequence
            ),
            "last_step": None if self._last_record is None else self._last_record.step,
        }

    def flush(self) -> None:
        if self._stream is not None:
            self._stream.flush()

    def close(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.flush()
        finally:
            self._stream.close()
            self._stream = None

    def _validate_expected_state(
        self,
        replay: tuple[MetricHistoryRecord, ...],
        expected_state: Mapping[str, Any] | None,
    ) -> None:
        if expected_state is None:
            return
        version = expected_state.get("version")
        if version != METRIC_HISTORY_VERSION:
            raise ValueError(
                f"checkpoint metric history version {version!r} is unsupported; "
                f"expected {METRIC_HISTORY_VERSION}"
            )
        actual_sequence = None if not replay else replay[-1].sequence
        actual_step = None if not replay else replay[-1].step
        expected_sequence = _optional_non_negative_int(
            "checkpoint metric history last_sequence",
            expected_state.get("last_sequence"),
        )
        expected_step = _optional_non_negative_int(
            "checkpoint metric history last_step",
            expected_state.get("last_step"),
        )
        if self.source_missing:
            return
        if (actual_sequence, actual_step) != (expected_sequence, expected_step):
            raise RuntimeError(
                "metric history does not match the checkpoint: "
                f"expected sequence/step {(expected_sequence, expected_step)}, "
                f"got {(actual_sequence, actual_step)}"
            )


__all__ = [
    "JsonlMetricHistory",
    "METRIC_HISTORY_VERSION",
    "MetricHistoryRecord",
    "read_metric_history",
]
