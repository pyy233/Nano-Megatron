from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from nano_megatron.cli.tokenizer import run_preprocess, run_train
from nano_megatron.cli.train import run
from nano_megatron.data import TokenCorpus


def _write_documents(path: Path) -> None:
    documents = (
        "Once upon a time, a tiny fox learned to share a bright lantern.",
        "小猫坐在窗边听雨，然后对朋友说：明天见。",
        "Byte-level tokenizers preserve emoji 🐍🚀 and newlines.\nSecond paragraph.",
        "A red kite flew over the green hill while three children laughed.",
    ) * 16
    path.write_text(
        "".join(json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in documents),
        encoding="utf-8",
    )


def _write_train_config(
    path: Path,
    *,
    vocab_size: int,
    tokenizer_path: Path,
    text_path: Path | None = None,
    token_path: Path | None = None,
    max_steps: int = 1,
    save_interval: int = 0,
) -> None:
    payload = {
        "distributed": {"backend": "gloo", "device": "cpu"},
        "parallel": {"data": 1},
        "model": {
            "layers": 1,
            "hidden_size": 16,
            "ffn_hidden_size": 32,
            "heads": 4,
            "kv_heads": 2,
            "seq_length": 8,
            "vocab_size": vocab_size,
            "dropout": 0.0,
        },
        "precision": {
            "params": "float32",
            "compute": "float32",
            "grad_reduce": "float32",
        },
        "pipeline": {"schedule": "gpipe"},
        "optimizer": {"lr": 0.001, "weight_decay": 0.0},
        "training": {
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "max_steps": max_steps,
            "seed": 7,
        },
        "checkpoint": {
            "directory": str(path.parent / "checkpoints"),
            "save_interval": save_interval,
        },
        "data": {
            "path": None if token_path is None else str(token_path),
            "text_path": None if text_path is None else str(text_path),
            "text_key": "text",
            "tokenizer": {
                "path": str(tokenizer_path),
                "append_eos": True,
            },
            "num_workers": 0,
            "shuffle": False,
            "packed_sequences": False,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


@pytest.mark.parametrize("offline_preprocess", [False, True])
def test_real_text_pipeline_reaches_one_gpt_optimizer_step(
    tmp_path: Path,
    offline_preprocess: bool,
) -> None:
    documents = tmp_path / "stories.jsonl"
    artifact = tmp_path / "tokenizer"
    tokens = tmp_path / "stories.tokens.pt"
    config_path = tmp_path / "train.yaml"
    _write_documents(documents)
    trained = run_train(
        (documents,),
        artifact,
        vocab_size=300,
        min_frequency=1,
    )

    if offline_preprocess:
        run_preprocess(documents, tokens, artifact)
    _write_train_config(
        config_path,
        vocab_size=trained["vocab_size"],
        tokenizer_path=artifact,
        text_path=None if offline_preprocess else documents,
        token_path=tokens if offline_preprocess else None,
    )

    state = run(config_path, overrides=(), resume=None, max_steps=1)

    assert state.step == 1
    assert state.consumed_samples == 4
    assert state.consumed_tokens == 32


def test_train_cli_rejects_tokenizer_model_vocab_mismatch_before_setup(
    tmp_path: Path,
) -> None:
    documents = tmp_path / "stories.jsonl"
    artifact = tmp_path / "tokenizer"
    config_path = tmp_path / "train.yaml"
    _write_documents(documents)
    trained = run_train(
        (documents,),
        artifact,
        vocab_size=300,
        min_frequency=1,
    )
    _write_train_config(
        config_path,
        vocab_size=trained["vocab_size"] + 1,
        tokenizer_path=artifact,
        text_path=documents,
    )

    with pytest.raises(ValueError, match="vocabulary sizes must match exactly"):
        run(config_path, overrides=(), resume=None, max_steps=1)


def test_tokenized_data_checkpoint_resume_restores_next_batch_position(
    tmp_path: Path,
) -> None:
    documents = tmp_path / "stories.jsonl"
    artifact = tmp_path / "tokenizer"
    tokens = tmp_path / "stories.tokens.pt"
    config_path = tmp_path / "train.yaml"
    _write_documents(documents)
    trained = run_train(
        (documents,),
        artifact,
        vocab_size=300,
        min_frequency=1,
    )
    run_preprocess(documents, tokens, artifact)
    _write_train_config(
        config_path,
        vocab_size=trained["vocab_size"],
        tokenizer_path=artifact,
        token_path=tokens,
        max_steps=2,
        save_interval=1,
    )

    first = run(config_path, overrides=(), resume=None, max_steps=1)
    checkpoint = next((tmp_path / "checkpoints").glob("*/step_00000001"))
    resumed = run(config_path, overrides=(), resume=checkpoint, max_steps=2)

    assert checkpoint.is_dir()
    assert first.step == 1
    assert resumed.step == 2
    assert resumed.consumed_samples == 8
    assert resumed.consumed_tokens == 64
    assert first.data_fingerprint
    assert resumed.data_fingerprint == first.data_fingerprint


def test_tokenized_resume_rejects_replaced_corpus_at_the_same_path(
    tmp_path: Path,
) -> None:
    documents = tmp_path / "stories.jsonl"
    artifact = tmp_path / "tokenizer"
    tokens_path = tmp_path / "stories.tokens.pt"
    config_path = tmp_path / "train.yaml"
    _write_documents(documents)
    trained = run_train(
        (documents,),
        artifact,
        vocab_size=300,
        min_frequency=1,
    )
    run_preprocess(documents, tokens_path, artifact)
    _write_train_config(
        config_path,
        vocab_size=trained["vocab_size"],
        tokenizer_path=artifact,
        token_path=tokens_path,
        max_steps=2,
        save_interval=1,
    )
    run(config_path, overrides=(), resume=None, max_steps=1)

    corpus = TokenCorpus.load(tokens_path)
    changed_tokens = corpus.tokens.clone()
    changed_tokens[0] = (changed_tokens[0] + 1) % corpus.vocab_size
    replacement = replace(corpus, tokens=changed_tokens)
    tokens_path.unlink()
    replacement.save(tokens_path)

    checkpoint = next((tmp_path / "checkpoints").glob("*/step_00000001"))
    with pytest.raises(RuntimeError, match="data fingerprint does not match"):
        run(config_path, overrides=(), resume=checkpoint, max_steps=2)
