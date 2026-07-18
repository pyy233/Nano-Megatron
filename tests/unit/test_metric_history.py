from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from nano_megatron.training import JsonlMetricHistory, read_metric_history


def _state(step: int) -> SimpleNamespace:
    return SimpleNamespace(
        step=step,
        consumed_samples=step * 4,
        consumed_tokens=step * 512,
        data_epoch=step // 10,
        data_sample_offset=step * 4,
        data_shuffle_seed=1234,
    )


def test_metric_history_round_trips_standard_json_and_nonfinite_metrics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    history = JsonlMetricHistory(path)
    history.start(through_step=None)
    history.append(
        "train",
        {"loss": 1.5, "nan_metric": float("nan"), "inf_metric": float("inf")},
        state=_state(1),
    )
    state = history.state_dict()
    history.close()

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["metrics"]["nan_metric"] == "NaN"
    assert raw["metrics"]["inf_metric"] == "Infinity"
    records = read_metric_history(path)
    assert len(records) == 1
    assert records[0].wandb_payload()["train/loss"] == 1.5
    assert math.isnan(records[0].metrics["nan_metric"])
    assert records[0].metrics["inf_metric"] == float("inf")
    assert state == {"version": 1, "last_sequence": 0, "last_step": 1}


def test_resume_history_copies_only_checkpoint_prefix_and_continues_sequence(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source" / "metrics.jsonl"
    source = JsonlMetricHistory(source_path)
    source.start(through_step=None)
    source.append("train", {"loss": 3.0}, state=_state(1))
    checkpoint_state = source.state_dict()
    source.append("train", {"loss": 2.0}, state=_state(2))
    source.append("validation", {"loss": 2.5}, state=_state(2))
    source.close()

    child_path = tmp_path / "child" / "metrics.jsonl"
    child = JsonlMetricHistory(child_path, source=source_path)
    replay = child.start(through_step=1, expected_state=checkpoint_state)
    child.append("train", {"loss": 1.0}, state=_state(2))
    child_state = child.state_dict()
    child.close()

    assert [(record.sequence, record.step, record.prefix) for record in replay] == [
        (0, 1, "train")
    ]
    records = read_metric_history(child_path)
    assert [(record.sequence, record.step, record.prefix) for record in records] == [
        (0, 1, "train"),
        (1, 2, "train"),
    ]
    assert child_state == {"version": 1, "last_sequence": 1, "last_step": 2}


def test_missing_resume_history_warns_and_starts_empty_child(tmp_path: Path) -> None:
    child = JsonlMetricHistory(
        tmp_path / "child" / "metrics.jsonl",
        source=tmp_path / "missing" / "metrics.jsonl",
    )

    with pytest.warns(UserWarning, match="resume metric history is missing"):
        assert child.start(
            through_step=5,
            expected_state={"version": 1, "last_sequence": 4, "last_step": 5},
        ) == ()
    child.append("train", {"loss": 1.0}, state=_state(6))
    child.close()

    records = read_metric_history(tmp_path / "child" / "metrics.jsonl")
    assert [(record.sequence, record.step) for record in records] == [(0, 6)]


def test_missing_resume_history_still_validates_checkpoint_state(tmp_path: Path) -> None:
    child = JsonlMetricHistory(
        tmp_path / "child" / "metrics.jsonl",
        source=tmp_path / "missing" / "metrics.jsonl",
    )

    with (
        pytest.warns(UserWarning, match="resume metric history is missing"),
        pytest.raises(ValueError, match="version"),
    ):
        child.start(
            through_step=5,
            expected_state={"version": 999, "last_sequence": 4, "last_step": 5},
        )


def test_resume_history_rejects_checkpoint_mismatch(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    source = JsonlMetricHistory(source_path)
    source.start(through_step=None)
    source.append("train", {"loss": 1.0}, state=_state(1))
    source.close()
    child = JsonlMetricHistory(tmp_path / "child.jsonl", source=source_path)

    with pytest.raises(RuntimeError, match="does not match the checkpoint"):
        child.start(
            through_step=1,
            expected_state={"version": 1, "last_sequence": 9, "last_step": 1},
        )


def test_reader_ignores_only_a_truncated_final_line(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    history = JsonlMetricHistory(path)
    history.start(through_step=None)
    history.append("train", {"loss": 1.0}, state=_state(1))
    history.close()
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"version":1')

    with pytest.warns(UserWarning, match="truncated final metric history"):
        records = read_metric_history(path)
    assert len(records) == 1


def test_reader_rejects_invalid_complete_final_line(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    history = JsonlMetricHistory(path)
    history.start(through_step=None)
    history.append("train", {"loss": 1.0}, state=_state(1))
    history.close()
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"version":1\n')

    with pytest.raises(ValueError, match="invalid metric history JSON"):
        read_metric_history(path)
