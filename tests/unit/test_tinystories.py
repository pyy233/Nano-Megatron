from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from nano_megatron.data import iter_jsonl_text
from nano_megatron.data.tinystories import (
    TINYSTORIES_DELIMITER,
    FileVerification,
    create_exact_sample,
    download_pinned_file,
    iter_delimited_stories,
    select_story_indices,
    validate_sample_artifact,
)
from nano_megatron.tokenizer.tinystories import train_tinystories_tokenizer


def _write_source(path: Path, stories: list[str], *, trailing_delimiter: bool = True) -> None:
    payload = TINYSTORIES_DELIMITER.join(story.encode("utf-8") for story in stories)
    if trailing_delimiter:
        payload += TINYSTORIES_DELIMITER
    path.write_bytes(payload)


def _verification(path: Path) -> FileVerification:
    payload = path.read_bytes()
    return FileVerification(
        path=path.resolve(),
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, *, status: int, headers: dict[str, str]) -> None:
        super().__init__(payload)
        self.status = status
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def test_iter_delimited_stories_handles_chunk_boundaries_unicode_and_final_record(
    tmp_path: Path,
) -> None:
    source = tmp_path / "stories.txt"
    stories = [
        "  Once upon a time.\n",
        "你好，世界。🐍",
        "Final story without a trailing delimiter.  ",
    ]
    _write_source(source, stories, trailing_delimiter=False)

    assert list(iter_delimited_stories(source, chunk_size=5)) == [
        "Once upon a time.",
        "你好，世界。🐍",
        "Final story without a trailing delimiter.",
    ]


def test_download_resumes_a_partial_file_and_verifies_before_publish(tmp_path: Path) -> None:
    payload = b"TinyStories fixture bytes\n" * 7
    destination = tmp_path / "TinyStories-train.txt"
    partial = destination.with_name(f"{destination.name}.part")
    partial.write_bytes(payload[:23])
    requests = []

    def opener(request):
        requests.append(request)
        assert request.headers["Range"] == "bytes=23-"
        return _Response(
            payload[23:],
            status=206,
            headers={"Content-Range": f"bytes 23-{len(payload) - 1}/{len(payload)}"},
        )

    result = download_pinned_file(
        destination,
        urls=("https://example.invalid/TinyStories-train.txt",),
        expected_size=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        chunk_size=11,
        retries=1,
        opener=opener,
    )

    assert len(requests) == 1
    assert result.path == destination.resolve()
    assert destination.read_bytes() == payload
    assert not partial.exists()


def test_download_restarts_partial_when_server_ignores_range(tmp_path: Path) -> None:
    payload = b"complete source"
    destination = tmp_path / "source.txt"
    destination.with_name(f"{destination.name}.part").write_bytes(payload[:4])

    def opener(request):
        assert request.headers["Range"] == "bytes=4-"
        return _Response(payload, status=200, headers={})

    download_pinned_file(
        destination,
        urls=("https://example.invalid/source.txt",),
        expected_size=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        chunk_size=3,
        retries=1,
        opener=opener,
    )

    assert destination.read_bytes() == payload


def test_exact_sample_is_deterministic_exact_and_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    first_sample = tmp_path / "sample-a.jsonl"
    first_metadata = tmp_path / "sample-a.metadata.json"
    second_sample = tmp_path / "sample-b.jsonl"
    second_metadata = tmp_path / "sample-b.metadata.json"
    stories = [f"Story {index}: value {index * 17}." for index in range(40)]
    _write_source(source, stories)
    verification = _verification(source)

    first = create_exact_sample(
        source,
        first_sample,
        first_metadata,
        source_verification=verification,
        sample_size=13,
        seed=77,
        expected_source_records=len(stories),
    )
    second = create_exact_sample(
        source,
        second_sample,
        second_metadata,
        source_verification=verification,
        sample_size=13,
        seed=77,
        expected_source_records=len(stories),
    )
    reused = create_exact_sample(
        source,
        first_sample,
        first_metadata,
        source_verification=verification,
        sample_size=13,
        seed=77,
        expected_source_records=len(stories),
    )

    assert len(list(iter_jsonl_text(first_sample))) == 13
    assert first_sample.read_bytes() == second_sample.read_bytes()
    assert first["artifact"]["sha256"] == second["artifact"]["sha256"]
    assert not first["reused"]
    assert reused["reused"]
    assert validate_sample_artifact(
        first_sample,
        first_metadata,
        sample_size=13,
        seed=77,
        expected_source_records=len(stories),
        expected_source_sha256=verification.sha256,
    )["artifact"]["records"] == 13


def test_exact_sample_rejects_unexpected_upstream_record_count(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    _write_source(source, ["one", "two", "three"])

    with pytest.raises(ValueError, match="source record count changed"):
        select_story_indices(
            source,
            sample_size=2,
            seed=1,
            expected_source_records=4,
        )


def test_verified_tokenizer_training_publishes_exact_vocab_and_reuses(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    sample = tmp_path / "train.jsonl"
    sample_metadata = tmp_path / "train.metadata.json"
    tokenizer_path = tmp_path / "tokenizer"
    stories = [
        f"Once upon a time, child {index} found a color {index % 11} lantern."
        for index in range(80)
    ]
    _write_source(source, stories)
    verification = _verification(source)
    create_exact_sample(
        source,
        sample,
        sample_metadata,
        source_verification=verification,
        sample_size=64,
        seed=9,
        expected_source_records=len(stories),
    )

    trained = train_tinystories_tokenizer(
        sample,
        sample_metadata,
        tokenizer_path,
        expected_documents=64,
        vocab_size=300,
        min_frequency=1,
        sample_seed=9,
        expected_source_records=len(stories),
        expected_source_sha256=verification.sha256,
        show_progress=False,
    )
    reused = train_tinystories_tokenizer(
        sample,
        sample_metadata,
        tokenizer_path,
        expected_documents=64,
        vocab_size=300,
        min_frequency=1,
        sample_seed=9,
        expected_source_records=len(stories),
        expected_source_sha256=verification.sha256,
        show_progress=False,
    )

    assert trained["vocab_size"] == 300
    assert trained["documents"] == 64
    assert not trained["reused"]
    assert reused["reused"]
    assert (tokenizer_path / "tokenizer.json").is_file()
    assert (tokenizer_path / "metadata.json").is_file()
    assert (tokenizer_path / "training.json").is_file()


def test_verified_tokenizer_does_not_publish_an_underfilled_vocab(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    sample = tmp_path / "train.jsonl"
    sample_metadata = tmp_path / "train.metadata.json"
    tokenizer_path = tmp_path / "underfilled-tokenizer"
    stories = ["A tiny repeated story." for _ in range(12)]
    _write_source(source, stories)
    verification = _verification(source)
    create_exact_sample(
        source,
        sample,
        sample_metadata,
        source_verification=verification,
        sample_size=10,
        seed=2,
        expected_source_records=len(stories),
    )

    with pytest.raises(RuntimeError, match="exact vocabulary size"):
        train_tinystories_tokenizer(
            sample,
            sample_metadata,
            tokenizer_path,
            expected_documents=10,
            vocab_size=5000,
            min_frequency=2,
            sample_seed=2,
            expected_source_records=len(stories),
            expected_source_sha256=verification.sha256,
            show_progress=False,
        )

    assert not tokenizer_path.exists()
    assert not list(tmp_path.glob(".underfilled-tokenizer.tmp-*"))
