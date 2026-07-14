from __future__ import annotations

from pathlib import Path

import pytest

from nano_megatron.config import (
    ConfigValidationError,
    DataConfig,
    TokenizerConfig,
    TrainConfig,
    load_config,
    validate_config,
)


def test_yaml_loader_constructs_typed_mmap_path(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
parallel:
  data: 1
data:
  mmap_path: data/processed/stories.mmap
  tokenizer:
    path: data/tokenizers/stories
    append_eos: true
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path, world_size=1)

    assert config.data.mmap_path == Path("data/processed/stories.mmap")
    assert config.data.tokenizer == TokenizerConfig(
        path=Path("data/tokenizers/stories"),
        append_eos=True,
    )


@pytest.mark.parametrize(
    "data",
    [
        DataConfig(path="tokens.pt", mmap_path="tokens.mmap"),
        DataConfig(path="tokens.pt", text_path="stories.jsonl"),
        DataConfig(mmap_path="tokens.mmap", text_path="stories.jsonl"),
        DataConfig(
            path="tokens.pt",
            mmap_path="tokens.mmap",
            text_path="stories.jsonl",
        ),
    ],
)
def test_validation_rejects_multiple_data_sources(data: DataConfig) -> None:
    with pytest.raises(ConfigValidationError, match="at most one"):
        validate_config(TrainConfig(data=data), world_size=1)


def test_tokenizer_can_reference_an_mmap_data_source() -> None:
    config = TrainConfig(
        data=DataConfig(
            mmap_path="tokens.mmap",
            tokenizer=TokenizerConfig(path="tokenizer"),
        )
    )

    validate_config(config, world_size=1)


@pytest.mark.parametrize("value", ["", "   "])
def test_mmap_path_rejects_empty_strings(value: str) -> None:
    with pytest.raises(ValueError, match="data.mmap_path"):
        DataConfig(mmap_path=value)


def test_mmap_path_rejects_non_path_values() -> None:
    with pytest.raises(TypeError, match="data.mmap_path"):
        DataConfig(mmap_path=object())  # type: ignore[arg-type]
