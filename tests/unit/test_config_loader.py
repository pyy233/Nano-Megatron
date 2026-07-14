from __future__ import annotations

from pathlib import Path

import pytest

from nano_megatron.config import (
    ConfigLoadError,
    DataParallelMode,
    PrecisionDType,
    config_from_dict,
    load_config,
)


def test_config_from_dict_is_recursive_strict_and_typed() -> None:
    config = config_from_dict(
        {
            "parallel": {
                "tensor": 2,
                "data": 1,
                "order": ["tp", "cp", "ep", "dp", "pp"],
                "sequence_parallel": True,
            },
            "model": {
                "layers": 2,
                "hidden_size": 16,
                "ffn_hidden_size": 32,
                "heads": 4,
                "seq_length": 8,
                "vocab_size": 31,
            },
            "precision": {
                "params": "float32",
                "compute": "bfloat16",
                "grad_reduce": "float32",
            },
            "data_parallel": {"mode": "zero2"},
            "context_parallel": {"backend": "ring", "dropout": 0.0},
            "optimizer": {"betas": [0.8, 0.9]},
            "checkpoint": {"directory": "tmp/checkpoints"},
            "data": {
                "text_path": "data/train.jsonl",
                "text_key": "story",
                "tokenizer": {
                    "path": "artifacts/tokenizer",
                    "append_eos": False,
                },
            },
        }
    )

    assert config.parallel.tensor == 2
    assert config.parallel.sequence_parallel
    assert config.precision.compute is PrecisionDType.BFLOAT16
    assert config.data_parallel.mode is DataParallelMode.ZERO2
    assert config.context_parallel.backend.value == "ring"
    assert config.optimizer.betas == (0.8, 0.9)
    assert config.checkpoint.directory == Path("tmp/checkpoints")
    assert config.data.text_path == Path("data/train.jsonl")
    assert config.data.text_key == "story"
    assert config.data.tokenizer is not None
    assert config.data.tokenizer.path == Path("artifacts/tokenizer")
    assert not config.data.tokenizer.append_eos


@pytest.mark.parametrize(
    "payload",
    [
        {"quantization": {"int8": True}},
        {"model": {"fp8": True}},
        {"parallel": {"tensor": 1, "mystery_axis": 2}},
        {"data": {"tokenizer": {"path": "tokenizer", "backend": "bpe"}}},
    ],
)
def test_unknown_or_quantization_fields_are_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ConfigLoadError, match="unsupported field"):
        config_from_dict(payload)


def test_loader_applies_dotted_overrides_before_validation(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
parallel:
  tensor: 1
  data: 1
model:
  layers: 2
  hidden_size: 16
  ffn_hidden_size: 32
  heads: 4
  seq_length: 8
  vocab_size: 31
training:
  micro_batch_size: 2
""".strip()
    )

    config = load_config(
        path,
        overrides=(
            "parallel.tensor=2",
            "parallel.sequence_parallel=true",
            "training.gradient_accumulation_steps=3",
        ),
        world_size=2,
    )

    assert config.parallel.tensor == 2
    assert config.parallel.sequence_parallel
    assert config.training.gradient_accumulation_steps == 3


def test_loader_rejects_invalid_override_path(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("parallel:\n  data: 1\n")

    with pytest.raises(ConfigLoadError, match="unsupported field"):
        load_config(path, overrides=("parallel.quantization=int8",), world_size=1)
