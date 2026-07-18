from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import nano_megatron.data.tinystories_corpus as corpus_module
from nano_megatron.data import (
    TinyStoriesCorpusError,
    build_tinystories_corpus,
    validate_tinystories_corpus,
)
from nano_megatron.data.tinystories import TINYSTORIES_DELIMITER, FileVerification
from nano_megatron.tokenizer import ByteBPETrainingConfig, ByteLevelBPETokenizer


def _write_source(path: Path, stories: tuple[str, ...]) -> FileVerification:
    payload = TINYSTORIES_DELIMITER.join(text.encode("utf-8") for text in stories)
    payload += TINYSTORIES_DELIMITER
    path.write_bytes(payload)
    return FileVerification(
        path=path.resolve(),
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _tokenizer(path: Path, stories: tuple[str, ...]) -> ByteLevelBPETokenizer:
    tokenizer = ByteLevelBPETokenizer.train(
        stories * 24,
        ByteBPETrainingConfig(vocab_size=300, min_frequency=1),
    )
    tokenizer.save(path)
    return tokenizer


def test_full_tinystories_corpus_builds_validates_and_reuses(tmp_path: Path) -> None:
    train_stories = (
        "Once upon a time, a fox found a lantern.",
        "A blue bird crossed the quiet green hill.",
        "The child shared a warm meal with a friend.",
    )
    validation_stories = (
        "A tiny dragon learned to tell the truth.",
        "The moon helped a lost rabbit find home.",
    )
    train = tmp_path / "train.txt"
    validation = tmp_path / "validation.txt"
    train_verification = _write_source(train, train_stories)
    validation_verification = _write_source(validation, validation_stories)
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer = _tokenizer(tokenizer_path, train_stories + validation_stories)
    output = tmp_path / "full-corpus"

    built = build_tinystories_corpus(
        train,
        validation,
        tokenizer_path,
        output,
        train_verification=train_verification,
        validation_verification=validation_verification,
        expected_train_stories=len(train_stories),
        expected_validation_stories=len(validation_stories),
        expected_vocab_size=tokenizer.vocab_size,
    )
    reused = build_tinystories_corpus(
        train,
        validation,
        tokenizer_path,
        output,
        train_verification=train_verification,
        validation_verification=validation_verification,
        expected_train_stories=len(train_stories),
        expected_validation_stories=len(validation_stories),
        expected_vocab_size=tokenizer.vocab_size,
    )
    manifest = validate_tinystories_corpus(output, tokenizer_path=tokenizer_path)

    assert not built["reused"]
    assert reused["reused"]
    assert manifest["corpora"]["train"]["documents"] == len(train_stories)
    assert manifest["corpora"]["validation"]["documents"] == len(validation_stories)
    assert manifest["tokenizer"]["fingerprint"] == tokenizer.fingerprint
    assert (output / ".complete").read_text(encoding="utf-8") == "complete\n"


def test_prepare_reuses_artifact_without_redownloading_deleted_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_stories = ("one complete story", "another complete story")
    validation_stories = ("one held out story",)
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    train = data_dir / "TinyStories-train.txt"
    validation = data_dir / "TinyStories-valid.txt"
    train_verification = _write_source(train, train_stories)
    validation_verification = _write_source(validation, validation_stories)
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer = _tokenizer(tokenizer_path, train_stories + validation_stories)
    output = tmp_path / "full-corpus"

    build_tinystories_corpus(
        train,
        validation,
        tokenizer_path,
        output,
        train_verification=train_verification,
        validation_verification=validation_verification,
        expected_train_stories=len(train_stories),
        expected_validation_stories=len(validation_stories),
        expected_vocab_size=tokenizer.vocab_size,
    )
    monkeypatch.setattr(
        corpus_module,
        "TINYSTORIES_TRAIN_BYTES",
        train_verification.size_bytes,
    )
    monkeypatch.setattr(
        corpus_module,
        "TINYSTORIES_TRAIN_SHA256",
        train_verification.sha256,
    )
    monkeypatch.setattr(
        corpus_module,
        "TINYSTORIES_TRAIN_STORIES",
        len(train_stories),
    )
    monkeypatch.setattr(
        corpus_module,
        "TINYSTORIES_VALIDATION_BYTES",
        validation_verification.size_bytes,
    )
    monkeypatch.setattr(
        corpus_module,
        "TINYSTORIES_VALIDATION_SHA256",
        validation_verification.sha256,
    )
    monkeypatch.setattr(
        corpus_module,
        "TINYSTORIES_VALIDATION_STORIES",
        len(validation_stories),
    )
    monkeypatch.setattr(
        corpus_module,
        "_EXPECTED_VOCAB_SIZE",
        tokenizer.vocab_size,
    )

    def _unexpected_download(*args: object, **kwargs: object) -> FileVerification:
        raise AssertionError("existing artifact must be reused before downloading sources")

    monkeypatch.setattr(corpus_module, "download_pinned_file", _unexpected_download)
    train.unlink()
    validation.unlink()

    reused = corpus_module.prepare_tinystories_corpus(
        data_dir,
        tokenizer_path,
        output,
    )

    assert reused["reused"]
    assert reused["manifest"]["corpora"]["train"]["documents"] == len(
        train_stories
    )


def test_full_tinystories_corpus_rejects_source_or_payload_mismatch(tmp_path: Path) -> None:
    train_stories = ("one story", "another story")
    validation_stories = ("held out story",)
    train = tmp_path / "train.txt"
    validation = tmp_path / "validation.txt"
    train_verification = _write_source(train, train_stories)
    validation_verification = _write_source(validation, validation_stories)
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer = _tokenizer(tokenizer_path, train_stories + validation_stories)
    output = tmp_path / "full-corpus"

    with pytest.raises(ValueError, match="must be distinct"):
        build_tinystories_corpus(
            train,
            train,
            tokenizer_path,
            output,
            train_verification=train_verification,
            validation_verification=train_verification,
            expected_train_stories=len(train_stories),
            expected_validation_stories=len(train_stories),
            expected_vocab_size=tokenizer.vocab_size,
        )

    build_tinystories_corpus(
        train,
        validation,
        tokenizer_path,
        output,
        train_verification=train_verification,
        validation_verification=validation_verification,
        expected_train_stories=len(train_stories),
        expected_validation_stories=len(validation_stories),
        expected_vocab_size=tokenizer.vocab_size,
    )
    with (output / "validation.mmap" / "tokens.bin").open("r+b") as stream:
        first = stream.read(1)
        stream.seek(0)
        stream.write(bytes([first[0] ^ 1]))

    with pytest.raises(TinyStoriesCorpusError, match="SHA256"):
        validate_tinystories_corpus(output, tokenizer_path=tokenizer_path)
