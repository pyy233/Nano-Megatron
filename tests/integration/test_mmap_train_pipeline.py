from __future__ import annotations

import json
from pathlib import Path

import yaml

from nano_megatron.cli.tokenizer import run_train
from nano_megatron.cli.train import run
from nano_megatron.data import TokenCorpus, preprocess_jsonl, preprocess_jsonl_mmap
from nano_megatron.tokenizer import ByteLevelBPETokenizer


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


def _write_config(
    path: Path,
    *,
    vocab_size: int,
    tokenizer_path: Path,
    checkpoint_path: Path,
    token_path: Path | None = None,
    mmap_path: Path | None = None,
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
            "max_steps": 2,
            "seed": 7,
        },
        "checkpoint": {
            "directory": str(checkpoint_path),
            "save_interval": 1,
        },
        "data": {
            "path": None if token_path is None else str(token_path),
            "mmap_path": None if mmap_path is None else str(mmap_path),
            "text_path": None,
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


def test_mmap_reaches_gpt_step_and_resumes_checkpoint_created_from_pt(
    tmp_path: Path,
) -> None:
    documents = tmp_path / "stories.jsonl"
    tokenizer_path = tmp_path / "tokenizer"
    token_path = tmp_path / "stories.tokens.pt"
    mmap_path = tmp_path / "stories.mmap"
    pt_config = tmp_path / "pt.yaml"
    mmap_config = tmp_path / "mmap.yaml"
    checkpoints = tmp_path / "checkpoints"
    _write_documents(documents)
    trained = run_train(
        (documents,),
        tokenizer_path,
        vocab_size=300,
        min_frequency=1,
    )
    tokenizer = ByteLevelBPETokenizer.load(tokenizer_path)
    pt_corpus = preprocess_jsonl(documents, token_path, tokenizer)
    mmap_corpus = preprocess_jsonl_mmap(documents, mmap_path, tokenizer)
    _write_config(
        pt_config,
        vocab_size=trained["vocab_size"],
        tokenizer_path=tokenizer_path,
        checkpoint_path=checkpoints,
        token_path=token_path,
    )
    _write_config(
        mmap_config,
        vocab_size=trained["vocab_size"],
        tokenizer_path=tokenizer_path,
        checkpoint_path=checkpoints,
        mmap_path=mmap_path,
    )

    assert mmap_corpus.fingerprint == pt_corpus.fingerprint
    assert mmap_corpus.token_count == pt_corpus.tokens.numel()
    mmap_corpus.close()
    first = run(pt_config, overrides=(), resume=None, max_steps=1)
    checkpoint = checkpoints / "step_00000001"
    resumed = run(mmap_config, overrides=(), resume=checkpoint, max_steps=2)

    assert checkpoint.is_dir()
    assert first.step == 1
    assert resumed.step == 2
    assert resumed.consumed_samples == 8
    assert resumed.consumed_tokens == 64
    assert first.data_fingerprint
    assert resumed.data_fingerprint == first.data_fingerprint


def test_mmap_artifact_is_not_a_torch_serialized_container(tmp_path: Path) -> None:
    documents = tmp_path / "stories.jsonl"
    tokenizer_path = tmp_path / "tokenizer"
    mmap_path = tmp_path / "stories.mmap"
    _write_documents(documents)
    run_train((documents,), tokenizer_path, vocab_size=300, min_frequency=1)
    tokenizer = ByteLevelBPETokenizer.load(tokenizer_path)

    corpus = preprocess_jsonl_mmap(documents, mmap_path, tokenizer)
    portable = TokenCorpus.from_jsonl(documents, tokenizer)

    assert corpus.fingerprint == portable.fingerprint
    assert (mmap_path / "tokens.bin").read_bytes()[:2] != b"PK"
    assert not (mmap_path / "data.pkl").exists()
