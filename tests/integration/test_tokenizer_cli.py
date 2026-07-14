from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nano_megatron.cli.tokenizer import (
    main,
    run_inspect,
    run_inspect_mmap,
    run_preprocess,
    run_train,
)
from nano_megatron.data import MMapTokenCorpus, TokenCorpus
from nano_megatron.tokenizer import ByteLevelBPETokenizer

_DOCUMENTS = (
    "Once upon a time, a small fox found a bright red lantern.\nIt glowed all night.",
    "你好，世界。小猫坐在窗边，认真地看着夏天的雨。",
    "Emoji remain exact bytes too: 🐍🚀✨ — even beside punctuation!",
    "Whitespace survives round trips:\n\nsecond paragraph\twith a tab.",
) * 12


def _write_jsonl(path: Path, *, text_key: str = "text") -> None:
    with path.open("w", encoding="utf-8") as stream:
        for text in _DOCUMENTS:
            stream.write(json.dumps({text_key: text}, ensure_ascii=False) + "\n")


def _assert_corpus_round_trip(
    corpus: TokenCorpus,
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    assert corpus.documents == len(_DOCUMENTS)
    assert corpus.document_offsets.numel() == len(_DOCUMENTS) + 1
    assert corpus.vocab_size == tokenizer.vocab_size
    assert corpus.eos_id == tokenizer.eos_token_id
    assert corpus.tokenizer_fingerprint == tokenizer.fingerprint
    assert corpus.append_eos

    offsets = corpus.document_offsets.tolist()
    for index, expected in enumerate(_DOCUMENTS):
        token_ids = corpus.tokens[offsets[index] : offsets[index + 1]].tolist()
        assert token_ids[-1] == tokenizer.eos_token_id
        assert tokenizer.decode(token_ids) == expected


def _read_stable_json(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert captured.out == json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    return payload


def test_direct_cli_functions_train_inspect_and_preprocess(tmp_path: Path) -> None:
    input_path = tmp_path / "stories.jsonl"
    artifact_path = tmp_path / "tokenizer"
    corpus_path = tmp_path / "stories.tokens.pt"
    _write_jsonl(input_path, text_key="story")

    trained = run_train(
        (input_path,),
        artifact_path,
        vocab_size=300,
        min_frequency=1,
        text_key="story",
    )

    assert trained == {
        "command": "train",
        "documents": len(_DOCUMENTS),
        "eos_token_id": 2,
        "fingerprint": trained["fingerprint"],
        "output": str(artifact_path),
        "vocab_size": 300,
    }
    assert len(trained["fingerprint"]) == 64
    assert (artifact_path / "tokenizer.json").is_file()
    assert (artifact_path / "metadata.json").is_file()

    sample = "Exact round trip:\n中文 and emoji 🐍🚀"
    inspected = run_inspect(artifact_path, sample, add_bos=True, add_eos=True)
    assert inspected == run_inspect(
        artifact_path,
        sample,
        add_bos=True,
        add_eos=True,
    )
    assert inspected["decoded"] == sample
    assert inspected["ids"][0] == inspected["bos_token_id"]
    assert inspected["ids"][-1] == inspected["eos_token_id"]
    assert len(inspected["ids"]) == len(inspected["tokens"])
    assert inspected["fingerprint"] == trained["fingerprint"]

    preprocessed = run_preprocess(
        input_path,
        corpus_path,
        artifact_path,
        text_key="story",
        append_eos=True,
    )
    tokenizer = ByteLevelBPETokenizer.load(artifact_path)
    corpus = TokenCorpus.load(corpus_path)

    assert preprocessed == {
        "append_eos": True,
        "command": "preprocess",
        "corpus_fingerprint": corpus.fingerprint,
        "documents": len(_DOCUMENTS),
        "eos_token_id": tokenizer.eos_token_id,
        "output": str(corpus_path),
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "tokens": int(corpus.tokens.numel()),
        "vocab_size": tokenizer.vocab_size,
    }
    _assert_corpus_round_trip(corpus, tokenizer)


def test_main_subcommands_emit_stable_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "stories.jsonl"
    artifact_path = tmp_path / "tokenizer"
    corpus_path = tmp_path / "stories.tokens.pt"
    _write_jsonl(input_path)

    main(
        [
            "train",
            "--input",
            str(input_path),
            "--output",
            str(artifact_path),
            "--vocab-size",
            "300",
            "--min-frequency",
            "1",
        ]
    )
    trained = _read_stable_json(capsys)
    assert trained["command"] == "train"
    assert trained["documents"] == len(_DOCUMENTS)
    assert trained["vocab_size"] == 300

    sample = "CLI preserves 中文, emoji 🚀, and\nnewlines."
    main(
        [
            "inspect",
            "--tokenizer",
            str(artifact_path),
            "--text",
            sample,
            "--add-bos",
            "--add-eos",
        ]
    )
    inspected = _read_stable_json(capsys)
    assert inspected["decoded"] == sample
    assert inspected["ids"][0] == inspected["bos_token_id"]
    assert inspected["ids"][-1] == inspected["eos_token_id"]
    assert inspected["fingerprint"] == trained["fingerprint"]

    main(
        [
            "preprocess",
            "--input",
            str(input_path),
            "--output",
            str(corpus_path),
            "--tokenizer",
            str(artifact_path),
        ]
    )
    preprocessed = _read_stable_json(capsys)
    corpus = TokenCorpus.load(corpus_path)
    tokenizer = ByteLevelBPETokenizer.load(artifact_path)

    assert preprocessed["command"] == "preprocess"
    assert preprocessed["documents"] == len(_DOCUMENTS)
    assert preprocessed["tokens"] == corpus.tokens.numel()
    assert preprocessed["tokenizer_fingerprint"] == trained["fingerprint"]
    _assert_corpus_round_trip(corpus, tokenizer)


def test_train_cli_checks_output_collision_before_reading_corpus(tmp_path: Path) -> None:
    output = tmp_path / "tokenizer"
    output.mkdir()
    (output / "user-file").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_train((tmp_path / "missing.jsonl",), output, vocab_size=300)

    assert (output / "user-file").read_text(encoding="utf-8") == "keep"


def test_mmap_preprocess_and_inspect_cli_emit_stable_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "stories.jsonl"
    tokenizer_path = tmp_path / "tokenizer"
    direct_path = tmp_path / "direct.mmap"
    cli_path = tmp_path / "cli.mmap"
    _write_jsonl(input_path)
    trained = run_train(
        (input_path,),
        tokenizer_path,
        vocab_size=300,
        min_frequency=1,
    )

    direct = run_preprocess(
        input_path,
        direct_path,
        tokenizer_path,
        output_format="mmap",
    )
    corpus = MMapTokenCorpus.load(direct_path)

    assert direct == {
        "append_eos": True,
        "command": "preprocess",
        "corpus_fingerprint": corpus.fingerprint,
        "documents": len(_DOCUMENTS),
        "eos_token_id": corpus.eos_id,
        "format": "mmap",
        "output": str(direct_path),
        "token_dtype": "uint16",
        "tokenizer_fingerprint": trained["fingerprint"],
        "tokens": corpus.token_count,
        "vocab_size": 300,
    }
    assert run_inspect_mmap(direct_path) == {
        "append_eos": True,
        "corpus_fingerprint": corpus.fingerprint,
        "documents": len(_DOCUMENTS),
        "eos_token_id": corpus.eos_id,
        "format": "mmap",
        "path": str(direct_path.resolve()),
        "token_dtype": "uint16",
        "tokenizer_fingerprint": trained["fingerprint"],
        "tokens": corpus.token_count,
        "vocab_size": 300,
    }

    main(
        [
            "preprocess",
            "--input",
            str(input_path),
            "--output",
            str(cli_path),
            "--tokenizer",
            str(tokenizer_path),
            "--format",
            "mmap",
        ]
    )
    preprocessed = _read_stable_json(capsys)
    assert preprocessed["format"] == "mmap"
    assert preprocessed["corpus_fingerprint"] == corpus.fingerprint

    main(["inspect", "--mmap", str(cli_path)])
    inspected = _read_stable_json(capsys)
    assert inspected["format"] == "mmap"
    assert inspected["corpus_fingerprint"] == corpus.fingerprint
    assert inspected["tokens"] == corpus.token_count


def test_preprocess_function_rejects_unknown_output_format(tmp_path: Path) -> None:
    input_path = tmp_path / "stories.jsonl"
    tokenizer_path = tmp_path / "tokenizer"
    _write_jsonl(input_path)
    run_train((input_path,), tokenizer_path, vocab_size=300, min_frequency=1)

    with pytest.raises(ValueError, match="output_format"):
        run_preprocess(
            input_path,
            tmp_path / "output",
            tokenizer_path,
            output_format="sharded",
        )
