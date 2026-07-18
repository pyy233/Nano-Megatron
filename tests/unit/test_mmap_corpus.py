from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch

from nano_megatron.data.corpus import TokenCorpus
from nano_megatron.data.mmap_corpus import (
    MMapCorpusError,
    MMapTokenCorpus,
    preprocess_jsonl_mmap,
    preprocess_texts_mmap,
)


class _Tokenizer:
    fingerprint = "mmap-unit-test-tokenizer"

    def __init__(self, vocab_size: int = 257) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = vocab_size - 1

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        assert not add_bos
        assert not add_eos
        usable = max(1, self.vocab_size - 1)
        return [ord(character) % usable for character in text]


def _write_jsonl(path: Path) -> tuple[str, ...]:
    documents = (
        "A small fox found a lantern.",
        "中文、emoji 🐍 and a newline.\nSecond line.",
        "The final story is deliberately a little longer than the others.",
    )
    path.write_text(
        "".join(json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in documents),
        encoding="utf-8",
    )
    return documents


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_metadata(path: Path) -> dict[str, object]:
    loaded = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _write_metadata(path: Path, metadata: dict[str, object]) -> None:
    (path / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_mmap_corpus_matches_portable_corpus_and_has_exact_layout(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    documents = _write_jsonl(source)
    tokenizer = _Tokenizer()
    portable = TokenCorpus.from_jsonl(source, tokenizer)
    artifact = tmp_path / "corpus.mmap"

    corpus = preprocess_jsonl_mmap(source, artifact, tokenizer)
    metadata = _read_metadata(artifact)

    assert {entry.name for entry in artifact.iterdir()} == {
        "tokens.bin",
        "document_offsets.bin",
        "metadata.json",
    }
    assert corpus.documents == len(documents)
    assert corpus.token_count == portable.tokens.numel()
    assert corpus.token_dtype == np.dtype("<u2")
    assert corpus.fingerprint == portable.fingerprint
    assert corpus.document_offsets.tolist() == portable.document_offsets.tolist()
    assert corpus.tokens.tolist() == portable.tokens.tolist()
    assert isinstance(corpus.tokens, np.memmap)
    assert not corpus.tokens.flags.writeable
    assert (artifact / "tokens.bin").stat().st_size == corpus.token_count * 2
    assert (artifact / "document_offsets.bin").stat().st_size == (len(documents) + 1) * 8
    assert metadata["tokens_sha256"] == _sha256(artifact / "tokens.bin")
    assert metadata["document_offsets_sha256"] == _sha256(artifact / "document_offsets.bin")
    assert metadata["fingerprint"] == portable.fingerprint
    corpus.close()
    assert corpus._tokens is None


@pytest.mark.parametrize(
    ("vocab_size", "expected_dtype", "itemsize"),
    [(1 << 16, "<u2", 2), ((1 << 16) + 1, "<u4", 4)],
)
def test_mmap_selects_smallest_unsigned_dtype_for_vocabulary(
    tmp_path: Path,
    vocab_size: int,
    expected_dtype: str,
    itemsize: int,
) -> None:
    source = tmp_path / "documents.jsonl"
    _write_jsonl(source)
    artifact = tmp_path / f"vocab-{vocab_size}"

    corpus = preprocess_jsonl_mmap(source, artifact, _Tokenizer(vocab_size))

    assert corpus.token_dtype == np.dtype(expected_dtype)
    assert (artifact / "tokens.bin").stat().st_size == corpus.token_count * itemsize
    assert int(corpus.tokens.max()) == vocab_size - 1


def test_mmap_rejects_vocabularies_larger_than_uint32(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    _write_jsonl(source)

    with pytest.raises(ValueError, match="vocab_size"):
        preprocess_jsonl_mmap(source, tmp_path / "too-wide", _Tokenizer((1 << 32) + 1))


def test_mmap_without_appended_eos_matches_portable_corpus(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    _write_jsonl(source)
    tokenizer = _Tokenizer()
    expected = TokenCorpus.from_jsonl(source, tokenizer, append_eos=False)

    actual = preprocess_jsonl_mmap(
        source,
        tmp_path / "without-eos",
        tokenizer,
        append_eos=False,
    )

    assert not actual.append_eos
    assert actual.tokens.tolist() == expected.tokens.tolist()
    assert actual.fingerprint == expected.fingerprint


def test_mmap_iterable_writer_matches_jsonl_writer(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    documents = _write_jsonl(source)
    tokenizer = _Tokenizer()
    expected = preprocess_jsonl_mmap(source, tmp_path / "jsonl", tokenizer)

    actual = preprocess_texts_mmap(iter(documents), tmp_path / "iterable", tokenizer)

    assert actual.documents == expected.documents
    assert actual.token_count == expected.token_count
    assert actual.fingerprint == expected.fingerprint
    assert actual.document_offsets.tolist() == expected.document_offsets.tolist()
    assert actual.tokens.tolist() == expected.tokens.tolist()


def test_mmap_iterable_writer_rejects_empty_or_invalid_documents(tmp_path: Path) -> None:
    tokenizer = _Tokenizer()

    with pytest.raises(ValueError, match="zero documents"):
        preprocess_texts_mmap((), tmp_path / "empty", tokenizer)
    assert not (tmp_path / "empty").exists()

    with pytest.raises(TypeError, match="must be a string"):
        preprocess_texts_mmap(["valid", 17], tmp_path / "invalid", tokenizer)  # type: ignore[list-item]
    assert not (tmp_path / "invalid").exists()


def test_mmap_iterable_writer_uses_bounded_batch_encoding(tmp_path: Path) -> None:
    class BatchTokenizer(_Tokenizer):
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes: list[int] = []

        def encode_batch(
            self,
            texts: list[str],
            *,
            add_bos: bool = False,
            add_eos: bool = False,
        ) -> list[list[int]]:
            assert not add_bos
            assert not add_eos
            self.batch_sizes.append(len(texts))
            return [self.encode(text) for text in texts]

    tokenizer = BatchTokenizer()
    documents = (f"document {index}" for index in range(10))

    corpus = preprocess_texts_mmap(
        documents,
        tmp_path / "batched",
        tokenizer,
        encoding_batch_size=4,
    )

    assert corpus.documents == 10
    assert tokenizer.batch_sizes == [4, 4, 2]


def test_mmap_iterable_writer_rejects_invalid_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="encoding_batch_size"):
        preprocess_texts_mmap(
            ["document"],
            tmp_path / "invalid-batch",
            _Tokenizer(),
            encoding_batch_size=0,
        )


def test_mmap_pickle_drops_open_mapping_and_reopens_lazily(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    _write_jsonl(source)
    corpus = preprocess_jsonl_mmap(source, tmp_path / "corpus", _Tokenizer())
    expected = corpus.read_tokens(1, 9).copy()
    assert corpus._tokens is not None

    restored = pickle.loads(pickle.dumps(corpus))

    assert restored._tokens is None
    assert not restored.document_offsets.flags.writeable
    np.testing.assert_array_equal(restored.read_tokens(1, 9), expected)
    assert restored._tokens is not None
    restored.close()


def test_mmap_close_keeps_escaped_views_safe_and_reopens_new_mapping(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    _write_jsonl(source)
    corpus = preprocess_jsonl_mmap(source, tmp_path / "corpus", _Tokenizer())
    view = corpus.read_tokens(0, 8)
    expected = view.copy()
    original_mapping = corpus.tokens

    corpus.close()

    np.testing.assert_array_equal(view, expected)
    assert corpus._tokens is None
    reopened = corpus.tokens
    assert reopened is not original_mapping
    np.testing.assert_array_equal(reopened[:8], expected)


def test_mmap_lazy_reopen_rejects_payload_changed_after_validation(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    _write_jsonl(source)
    artifact = tmp_path / "corpus"
    corpus = preprocess_jsonl_mmap(source, artifact, _Tokenizer())
    corpus.close()
    tokens_path = artifact / "tokens.bin"
    changed = bytearray(tokens_path.read_bytes())
    changed[0] ^= 1
    tokens_path.write_bytes(changed)

    with pytest.raises(MMapCorpusError, match="changed after.*validated"):
        corpus.read_tokens(0, 1)


def test_mmap_pickle_keeps_absolute_storage_identity_after_cwd_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "documents.jsonl"
    _write_jsonl(source)
    monkeypatch.chdir(tmp_path)
    corpus = preprocess_jsonl_mmap(source.name, "relative-corpus", _Tokenizer())
    expected = corpus.read_tokens(0, 8).copy()
    serialized = pickle.dumps(corpus)
    other_directory = tmp_path / "other"
    other_directory.mkdir()
    monkeypatch.chdir(other_directory)

    restored = pickle.loads(serialized)

    assert restored.path.is_absolute()
    np.testing.assert_array_equal(restored.read_tokens(0, 8), expected)


def test_mmap_preprocess_does_not_mutate_tokenizer_owned_lists(tmp_path: Path) -> None:
    class CachingTokenizer(_Tokenizer):
        def __init__(self) -> None:
            super().__init__()
            self.cached = [1, 2]

        def encode(
            self,
            text: str,
            *,
            add_bos: bool = False,
            add_eos: bool = False,
        ) -> list[int]:
            return self.cached

    source = tmp_path / "documents.jsonl"
    source.write_text('{"text":"first"}\n{"text":"second"}\n', encoding="utf-8")
    tokenizer = CachingTokenizer()

    corpus = preprocess_jsonl_mmap(source, tmp_path / "corpus", tokenizer)

    assert tokenizer.cached == [1, 2]
    assert corpus.tokens.tolist() == [1, 2, tokenizer.eos_token_id] * 2


@pytest.mark.parametrize(
    ("start", "stop", "error"),
    [(-1, 1, IndexError), (2, 1, IndexError), (0, 10_000, IndexError), (True, 1, TypeError)],
)
def test_mmap_range_reads_validate_bounds(
    tmp_path: Path,
    start: int,
    stop: int,
    error: type[Exception],
) -> None:
    source = tmp_path / "documents.jsonl"
    _write_jsonl(source)
    corpus = preprocess_jsonl_mmap(source, tmp_path / "corpus", _Tokenizer())

    with pytest.raises(error):
        corpus.read_tokens(start, stop)


def test_mmap_publish_accepts_empty_directory_but_never_overwrites_data(
    tmp_path: Path,
) -> None:
    source = tmp_path / "documents.jsonl"
    _write_jsonl(source)
    empty = tmp_path / "empty"
    empty.mkdir()

    written = preprocess_jsonl_mmap(source, empty, _Tokenizer())

    assert written.path == empty
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    user_file = occupied / "keep.txt"
    user_file.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        preprocess_jsonl_mmap(source, occupied, _Tokenizer())
    assert user_file.read_text(encoding="utf-8") == "keep"
    file_target = tmp_path / "file"
    file_target.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        preprocess_jsonl_mmap(source, file_target, _Tokenizer())


def test_mmap_failure_cleans_temporary_artifact_and_restores_empty_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "documents.jsonl"
    _write_jsonl(source)
    output = tmp_path / "corpus"
    output.mkdir()
    original_load = MMapTokenCorpus.load

    def fail_self_validation(path: str | Path) -> MMapTokenCorpus:
        if Path(path) != output:
            raise RuntimeError("injected self-validation failure")
        return original_load(path)

    monkeypatch.setattr(MMapTokenCorpus, "load", staticmethod(fail_self_validation))

    with pytest.raises(RuntimeError, match="injected"):
        preprocess_jsonl_mmap(source, output, _Tokenizer())

    assert output.is_dir()
    assert not any(output.iterdir())
    assert not list(tmp_path.glob(".corpus.tmp-*"))


def test_mmap_publish_failure_after_empty_target_removal_restores_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "documents.jsonl"
    _write_jsonl(source)
    output = tmp_path / "corpus"
    output.mkdir()

    def fail_publish(source_path: Path, destination_path: Path) -> None:
        raise OSError(f"injected rename failure: {source_path} -> {destination_path}")

    monkeypatch.setattr("nano_megatron.data.mmap_corpus.os.replace", fail_publish)

    with pytest.raises(OSError, match="injected rename failure"):
        preprocess_jsonl_mmap(source, output, _Tokenizer())

    assert output.is_dir()
    assert not any(output.iterdir())
    assert not list(tmp_path.glob(".corpus.tmp-*"))


def test_mmap_publish_refuses_to_replace_broken_symlink(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    _write_jsonl(source)
    output = tmp_path / "corpus"
    output.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    with pytest.raises(FileExistsError, match="symlink"):
        preprocess_jsonl_mmap(source, output, _Tokenizer())

    assert output.is_symlink()
    assert not (tmp_path / "missing-target").exists()


@pytest.mark.parametrize("filename", ["tokens.bin", "document_offsets.bin"])
def test_mmap_detects_truncated_payloads(tmp_path: Path, filename: str) -> None:
    source = tmp_path / "documents.jsonl"
    _write_jsonl(source)
    artifact = tmp_path / "corpus"
    preprocess_jsonl_mmap(source, artifact, _Tokenizer()).close()
    payload = artifact / filename
    payload.write_bytes(payload.read_bytes()[:-1])

    with pytest.raises(MMapCorpusError, match="size"):
        MMapTokenCorpus.load(artifact)


def test_mmap_detects_payload_checksum_and_logical_fingerprint_changes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "documents.jsonl"
    _write_jsonl(source)
    artifact = tmp_path / "corpus"
    preprocess_jsonl_mmap(source, artifact, _Tokenizer()).close()
    tokens_path = artifact / "tokens.bin"
    changed = bytearray(tokens_path.read_bytes())
    changed[0] ^= 1
    tokens_path.write_bytes(changed)

    with pytest.raises(MMapCorpusError, match="SHA256"):
        MMapTokenCorpus.load(artifact)

    metadata = _read_metadata(artifact)
    metadata["tokens_sha256"] = _sha256(tokens_path)
    _write_metadata(artifact, metadata)
    with pytest.raises(MMapCorpusError, match="logical fingerprint"):
        MMapTokenCorpus.load(artifact)


def test_mmap_detects_invalid_offsets_even_with_updated_checksum(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    _write_jsonl(source)
    artifact = tmp_path / "corpus"
    preprocess_jsonl_mmap(source, artifact, _Tokenizer()).close()
    offsets_path = artifact / "document_offsets.bin"
    offsets = np.fromfile(offsets_path, dtype="<u8")
    offsets[1] = offsets[0]
    offsets.tofile(offsets_path)
    metadata = _read_metadata(artifact)
    metadata["document_offsets_sha256"] = _sha256(offsets_path)
    _write_metadata(artifact, metadata)

    with pytest.raises(MMapCorpusError, match="empty documents"):
        MMapTokenCorpus.load(artifact)


def test_mmap_rejects_more_documents_than_tokens_before_loading_offsets(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    _write_jsonl(source)
    artifact = tmp_path / "corpus"
    corpus = preprocess_jsonl_mmap(source, artifact, _Tokenizer())
    corpus.close()
    metadata = _read_metadata(artifact)
    metadata["documents"] = int(metadata["token_count"]) + 1
    _write_metadata(artifact, metadata)

    with pytest.raises(MMapCorpusError, match="documents.*between"):
        MMapTokenCorpus.load(artifact)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda metadata: metadata.update(extra=True), "unknown keys"),
        (lambda metadata: metadata.pop("token_count"), "missing keys"),
        (lambda metadata: metadata.update(token_dtype="<u4"), "canonical dtype"),
        (lambda metadata: metadata.update(fingerprint="bad"), "SHA256"),
    ],
)
def test_mmap_rejects_malformed_metadata(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    source = tmp_path / "documents.jsonl"
    _write_jsonl(source)
    artifact = tmp_path / "corpus"
    preprocess_jsonl_mmap(source, artifact, _Tokenizer()).close()
    metadata = _read_metadata(artifact)
    mutation(metadata)
    _write_metadata(artifact, metadata)

    with pytest.raises(MMapCorpusError, match=message):
        MMapTokenCorpus.load(artifact)


def test_mmap_rejects_missing_or_extra_artifact_files(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    _write_jsonl(source)
    missing = tmp_path / "missing"
    preprocess_jsonl_mmap(source, missing, _Tokenizer()).close()
    (missing / "tokens.bin").unlink()
    with pytest.raises(MMapCorpusError, match="missing files"):
        MMapTokenCorpus.load(missing)

    extra = tmp_path / "extra"
    preprocess_jsonl_mmap(source, extra, _Tokenizer()).close()
    (extra / "unexpected").write_text("x", encoding="utf-8")
    with pytest.raises(MMapCorpusError, match="unknown files"):
        MMapTokenCorpus.load(extra)


def test_mmap_rejects_non_regular_or_symlinked_sidecars(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    _write_jsonl(source)
    artifact = tmp_path / "corpus"
    preprocess_jsonl_mmap(source, artifact, _Tokenizer()).close()
    metadata = artifact / "metadata.json"
    real_metadata = tmp_path / "metadata.real.json"
    metadata.rename(real_metadata)
    metadata.symlink_to(real_metadata)

    with pytest.raises(MMapCorpusError, match="regular file"):
        MMapTokenCorpus.load(artifact)


def test_mmap_terminal_eos_validation_is_independent_of_checksum(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    _write_jsonl(source)
    artifact = tmp_path / "corpus"
    corpus = preprocess_jsonl_mmap(source, artifact, _Tokenizer())
    last_index = int(corpus.document_offsets[1] - 1)
    corpus.close()
    tokens_path = artifact / "tokens.bin"
    tokens = np.memmap(tokens_path, dtype="<u2", mode="r+")
    tokens[last_index] = 1
    tokens.flush()
    tokens._mmap.close()
    metadata = _read_metadata(artifact)
    metadata["tokens_sha256"] = _sha256(tokens_path)
    _write_metadata(artifact, metadata)

    with pytest.raises(MMapCorpusError, match="end with eos_id"):
        MMapTokenCorpus.load(artifact)


def test_mmap_tokens_can_be_copied_to_torch_without_sharing_storage(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    _write_jsonl(source)
    corpus = preprocess_jsonl_mmap(source, tmp_path / "corpus", _Tokenizer())
    window = np.array(corpus.read_tokens(0, 8), dtype=np.int64, copy=True)
    tensor = torch.from_numpy(window)
    original = int(corpus.tokens[0])

    tensor[0] = -1

    assert int(corpus.tokens[0]) == original
