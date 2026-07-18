"""Atomic full TinyStories train/validation mmap preparation."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nano_megatron.tokenizer import ByteLevelBPETokenizer

from .mmap_corpus import MMapTokenCorpus, preprocess_texts_mmap
from .tinystories import (
    TINYSTORIES_CANONICAL_REVISION,
    TINYSTORIES_DELIMITER,
    TINYSTORIES_MIRROR_REVISION,
    TINYSTORIES_TRAIN_BYTES,
    TINYSTORIES_TRAIN_SHA256,
    TINYSTORIES_TRAIN_STORIES,
    TINYSTORIES_TRAIN_URLS,
    TINYSTORIES_VALIDATION_BYTES,
    TINYSTORIES_VALIDATION_SHA256,
    TINYSTORIES_VALIDATION_STORIES,
    TINYSTORIES_VALIDATION_URLS,
    FileVerification,
    ProgressCallback,
    download_pinned_file,
    iter_delimited_stories,
    verify_file,
)

_ARTIFACT_NAME = "nano_megatron.tinystories_corpus"
_ARTIFACT_VERSION = 1
_COMPLETE_FILE = ".complete"
_MANIFEST_FILE = "manifest.json"
_TRAIN_DIRECTORY = "train.mmap"
_VALIDATION_DIRECTORY = "validation.mmap"
_EXPECTED_VOCAB_SIZE = 8192
_MANIFEST_KEYS = {
    "artifact",
    "corpora",
    "format_version",
    "source_revision",
    "sources",
    "tokenizer",
}


class TinyStoriesCorpusError(ValueError):
    """Raised when a full TinyStories artifact violates its contract."""


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TinyStoriesCorpusError(
            f"could not read TinyStories corpus manifest: {error}"
        ) from error
    if not isinstance(value, dict):
        raise TinyStoriesCorpusError("TinyStories corpus manifest must contain a JSON object")
    return value


def _require_mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TinyStoriesCorpusError(f"{description} must be an object")
    return value


def _require_string(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise TinyStoriesCorpusError(f"{description} must be a non-empty string")
    return value


def _safe_directory(root: Path, value: Any, description: str) -> Path:
    relative = Path(_require_string(value, description))
    if relative.is_absolute() or ".." in relative.parts:
        raise TinyStoriesCorpusError(f"{description} must be a safe relative path")
    path = root / relative
    if not path.is_dir():
        raise TinyStoriesCorpusError(f"{description} does not exist: {path}")
    return path


def _corpus_manifest(directory: str, corpus: MMapTokenCorpus) -> dict[str, Any]:
    return {
        "append_eos": corpus.append_eos,
        "directory": directory,
        "document_offsets_sha256": corpus.document_offsets_sha256,
        "documents": corpus.documents,
        "fingerprint": corpus.fingerprint,
        "token_count": corpus.token_count,
        "token_dtype": corpus.token_dtype.str,
        "tokens_sha256": corpus.tokens_sha256,
    }


def _load_corpus(path: Path, description: str) -> MMapTokenCorpus:
    try:
        return MMapTokenCorpus.load(path)
    except (OSError, ValueError) as error:
        raise TinyStoriesCorpusError(f"could not validate {description}: {error}") from error


def _source_manifest(verification: FileVerification, stories: int) -> dict[str, Any]:
    return {
        "path": str(verification.path),
        "sha256": verification.sha256,
        "size_bytes": verification.size_bytes,
        "stories": stories,
        "story_delimiter": TINYSTORIES_DELIMITER.decode("ascii"),
    }


def _validate_corpus_entry(
    name: str,
    entry: Any,
    corpus: MMapTokenCorpus,
) -> None:
    value = _require_mapping(entry, f"corpora.{name}")
    expected = _corpus_manifest(str(value.get("directory")), corpus)
    if dict(value) != expected:
        raise TinyStoriesCorpusError(
            f"corpora.{name} metadata does not match its mmap artifact"
        )


def validate_tinystories_corpus(
    path: str | Path,
    *,
    tokenizer_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fully validate a published full TinyStories mmap artifact."""

    directory = Path(path).resolve()
    if not directory.is_dir() or not (directory / _COMPLETE_FILE).is_file():
        raise TinyStoriesCorpusError(f"TinyStories corpus is incomplete or missing: {directory}")
    actual_entries = {entry.name for entry in directory.iterdir()}
    expected_entries = {
        _COMPLETE_FILE,
        _MANIFEST_FILE,
        _TRAIN_DIRECTORY,
        _VALIDATION_DIRECTORY,
    }
    if actual_entries != expected_entries:
        raise TinyStoriesCorpusError(
            f"TinyStories corpus directory entries differ: {actual_entries} != {expected_entries}"
        )
    manifest = _read_json_object(directory / _MANIFEST_FILE)
    if set(manifest) != _MANIFEST_KEYS:
        raise TinyStoriesCorpusError("TinyStories corpus manifest has invalid top-level keys")
    if manifest.get("artifact") != _ARTIFACT_NAME:
        raise TinyStoriesCorpusError("TinyStories corpus artifact name is unsupported")
    if manifest.get("format_version") != _ARTIFACT_VERSION:
        raise TinyStoriesCorpusError("TinyStories corpus format version is unsupported")
    expected_revision = {
        "canonical": TINYSTORIES_CANONICAL_REVISION,
        "mirror": TINYSTORIES_MIRROR_REVISION,
    }
    if manifest.get("source_revision") != expected_revision:
        raise TinyStoriesCorpusError("TinyStories corpus source revision is invalid")

    tokenizer_info = _require_mapping(manifest.get("tokenizer"), "tokenizer")
    if set(tokenizer_info) != {"eos_token_id", "fingerprint", "path", "vocab_size"}:
        raise TinyStoriesCorpusError("tokenizer manifest fields are invalid")
    fingerprint = _require_string(tokenizer_info.get("fingerprint"), "tokenizer.fingerprint")
    _require_string(tokenizer_info.get("path"), "tokenizer.path")
    vocab_size = tokenizer_info.get("vocab_size")
    eos_id = tokenizer_info.get("eos_token_id")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 1:
        raise TinyStoriesCorpusError("tokenizer.vocab_size must be a positive integer")
    if isinstance(eos_id, bool) or not isinstance(eos_id, int) or not 0 <= eos_id < vocab_size:
        raise TinyStoriesCorpusError("tokenizer.eos_token_id is invalid")
    if tokenizer_path is not None:
        tokenizer = ByteLevelBPETokenizer.load(tokenizer_path)
        if (
            tokenizer.fingerprint != fingerprint
            or tokenizer.vocab_size != vocab_size
            or tokenizer.eos_token_id != eos_id
        ):
            raise TinyStoriesCorpusError(
                "TinyStories corpus tokenizer does not match the requested tokenizer artifact"
            )

    corpora = _require_mapping(manifest.get("corpora"), "corpora")
    if set(corpora) != {"train", "validation"}:
        raise TinyStoriesCorpusError("corpora must contain train and validation")
    train_entry = _require_mapping(corpora["train"], "corpora.train")
    validation_entry = _require_mapping(corpora["validation"], "corpora.validation")
    train = _load_corpus(
        _safe_directory(
            directory,
            train_entry.get("directory"),
            "corpora.train.directory",
        ),
        "train mmap",
    )
    validation = _load_corpus(
        _safe_directory(
            directory,
            validation_entry.get("directory"),
            "corpora.validation.directory",
        ),
        "validation mmap",
    )
    try:
        _validate_corpus_entry("train", train_entry, train)
        _validate_corpus_entry("validation", validation_entry, validation)
        for name, corpus in (("train", train), ("validation", validation)):
            if not corpus.append_eos:
                raise TinyStoriesCorpusError(f"{name} mmap must append EOS")
            if corpus.tokenizer_fingerprint != fingerprint:
                raise TinyStoriesCorpusError(f"{name} tokenizer fingerprint does not match")
            if corpus.vocab_size != vocab_size or corpus.eos_id != eos_id:
                raise TinyStoriesCorpusError(f"{name} tokenizer metadata does not match")
        if train.fingerprint == validation.fingerprint:
            raise TinyStoriesCorpusError("train and validation mmap fingerprints must differ")
    finally:
        train.close()
        validation.close()

    sources = _require_mapping(manifest.get("sources"), "sources")
    if set(sources) != {"train", "validation"}:
        raise TinyStoriesCorpusError("sources must contain train and validation")
    for name in ("train", "validation"):
        source = _require_mapping(sources[name], f"sources.{name}")
        if set(source) != {"path", "sha256", "size_bytes", "stories", "story_delimiter"}:
            raise TinyStoriesCorpusError(f"sources.{name} fields are invalid")
        _require_string(source.get("path"), f"sources.{name}.path")
        digest = _require_string(source.get("sha256"), f"sources.{name}.sha256")
        try:
            if len(digest) != 64:
                raise ValueError
            bytes.fromhex(digest)
        except ValueError as error:
            raise TinyStoriesCorpusError(
                f"sources.{name}.sha256 must be a SHA256 digest"
            ) from error
        for key in ("size_bytes", "stories"):
            value = source.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise TinyStoriesCorpusError(
                    f"sources.{name}.{key} must be a positive integer"
                )
        if source.get("story_delimiter") != TINYSTORIES_DELIMITER.decode("ascii"):
            raise TinyStoriesCorpusError(f"sources.{name}.story_delimiter is invalid")
    if sources["train"].get("sha256") == sources["validation"].get("sha256"):
        raise TinyStoriesCorpusError("train and validation source SHA256 must differ")
    return manifest


def _validate_output_target(directory: Path) -> bool:
    if directory.is_symlink():
        raise FileExistsError(f"refusing to replace TinyStories corpus symlink: {directory}")
    if not directory.exists():
        return False
    if not directory.is_dir():
        raise FileExistsError(f"TinyStories corpus output already exists: {directory}")
    return not any(directory.iterdir())


def _validated_source(
    path: Path,
    verification: FileVerification,
) -> FileVerification:
    if verification.path.resolve() != path.resolve():
        raise ValueError("source verification describes a different path")
    return verify_file(
        path,
        expected_size=verification.size_bytes,
        expected_sha256=verification.sha256,
    )


def build_tinystories_corpus(
    train_source: str | Path,
    validation_source: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
    *,
    train_verification: FileVerification,
    validation_verification: FileVerification,
    expected_train_stories: int,
    expected_validation_stories: int,
    expected_vocab_size: int = 8192,
) -> dict[str, Any]:
    """Build or strictly reuse one complete TinyStories mmap artifact."""

    train_path = Path(train_source).resolve()
    validation_path = Path(validation_source).resolve()
    if train_path == validation_path:
        raise ValueError("train and validation sources must be distinct")
    train_verified = _validated_source(train_path, train_verification)
    validation_verified = _validated_source(validation_path, validation_verification)
    if train_verified.sha256 == validation_verified.sha256:
        raise ValueError("train and validation source SHA256 must differ")
    tokenizer_source = Path(tokenizer_path).resolve()
    tokenizer = ByteLevelBPETokenizer.load(tokenizer_source)
    if tokenizer.vocab_size != expected_vocab_size:
        raise ValueError(
            f"TinyStories tokenizer vocab size {tokenizer.vocab_size} != {expected_vocab_size}"
        )
    destination = Path(output_path).resolve()
    destination_was_empty = _validate_output_target(destination)
    if destination.exists() and not destination_was_empty:
        manifest = validate_tinystories_corpus(
            destination,
            tokenizer_path=tokenizer_source,
        )
        sources = _require_mapping(manifest["sources"], "sources")
        corpora = _require_mapping(manifest["corpora"], "corpora")
        if (
            sources["train"] != _source_manifest(train_verified, expected_train_stories)
            or sources["validation"]
            != _source_manifest(validation_verified, expected_validation_stories)
            or corpora["train"].get("documents") != expected_train_stories
            or corpora["validation"].get("documents") != expected_validation_stories
        ):
            raise FileExistsError(
                "existing TinyStories corpus was built from different sources or counts"
            )
        return {
            "command": "prepare-tinystories-corpus",
            "manifest": manifest,
            "output": str(destination),
            "reused": True,
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    removed_empty_destination = False
    try:
        train = preprocess_texts_mmap(
            iter_delimited_stories(train_path),
            temporary / _TRAIN_DIRECTORY,
            tokenizer,
        )
        validation = preprocess_texts_mmap(
            iter_delimited_stories(validation_path),
            temporary / _VALIDATION_DIRECTORY,
            tokenizer,
        )
        if train.documents != expected_train_stories:
            raise TinyStoriesCorpusError(
                f"train source emitted {train.documents} stories, expected {expected_train_stories}"
            )
        if validation.documents != expected_validation_stories:
            raise TinyStoriesCorpusError(
                "validation source emitted "
                f"{validation.documents} stories, expected {expected_validation_stories}"
            )
        manifest = {
            "artifact": _ARTIFACT_NAME,
            "corpora": {
                "train": _corpus_manifest(_TRAIN_DIRECTORY, train),
                "validation": _corpus_manifest(_VALIDATION_DIRECTORY, validation),
            },
            "format_version": _ARTIFACT_VERSION,
            "source_revision": {
                "canonical": TINYSTORIES_CANONICAL_REVISION,
                "mirror": TINYSTORIES_MIRROR_REVISION,
            },
            "sources": {
                "train": _source_manifest(train_verified, train.documents),
                "validation": _source_manifest(validation_verified, validation.documents),
            },
            "tokenizer": {
                "eos_token_id": tokenizer.eos_token_id,
                "fingerprint": tokenizer.fingerprint,
                "path": str(tokenizer_source),
                "vocab_size": tokenizer.vocab_size,
            },
        }
        (temporary / _MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / _COMPLETE_FILE).write_text("complete\n", encoding="utf-8")
        validate_tinystories_corpus(temporary, tokenizer_path=tokenizer_source)

        if destination_was_empty:
            destination.rmdir()
            removed_empty_destination = True
        elif destination.exists() or destination.is_symlink():
            raise FileExistsError(f"TinyStories corpus output already exists: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        if removed_empty_destination and not destination.exists():
            destination.mkdir()
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    published = validate_tinystories_corpus(destination, tokenizer_path=tokenizer_source)
    return {
        "command": "prepare-tinystories-corpus",
        "manifest": published,
        "output": str(destination),
        "reused": False,
    }


def prepare_tinystories_corpus(
    data_dir: str | Path = Path("data/raw/tinystories"),
    tokenizer_path: str | Path = Path("data/tokenizers/tinystories-8k-500k"),
    output_path: str | Path = Path("data/processed/tinystories-full-8k"),
    *,
    train_urls: Sequence[str] = TINYSTORIES_TRAIN_URLS,
    validation_urls: Sequence[str] = TINYSTORIES_VALIDATION_URLS,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Download pinned train/validation sources and build their full mmap artifacts."""

    destination = Path(output_path).resolve()
    destination_was_empty = _validate_output_target(destination)
    if destination.exists() and not destination_was_empty:
        manifest = validate_tinystories_corpus(
            destination,
            tokenizer_path=tokenizer_path,
        )
        sources = _require_mapping(manifest["sources"], "sources")
        corpora = _require_mapping(manifest["corpora"], "corpora")
        tokenizer = _require_mapping(manifest["tokenizer"], "tokenizer")
        expected_sources = {
            "train": {
                "sha256": TINYSTORIES_TRAIN_SHA256,
                "size_bytes": TINYSTORIES_TRAIN_BYTES,
                "stories": TINYSTORIES_TRAIN_STORIES,
            },
            "validation": {
                "sha256": TINYSTORIES_VALIDATION_SHA256,
                "size_bytes": TINYSTORIES_VALIDATION_BYTES,
                "stories": TINYSTORIES_VALIDATION_STORIES,
            },
        }
        source_matches = all(
            all(source.get(key) == value for key, value in expected.items())
            for name, expected in expected_sources.items()
            for source in (_require_mapping(sources[name], f"sources.{name}"),)
        )
        corpus_counts_match = all(
            _require_mapping(corpora[name], f"corpora.{name}").get("documents")
            == expected["stories"]
            for name, expected in expected_sources.items()
        )
        if (
            not source_matches
            or not corpus_counts_match
            or tokenizer.get("vocab_size") != _EXPECTED_VOCAB_SIZE
        ):
            raise FileExistsError(
                "existing TinyStories corpus was built from different sources or counts"
            )
        return {
            "command": "prepare-tinystories-corpus",
            "manifest": manifest,
            "output": str(destination),
            "reused": True,
        }

    directory = Path(data_dir)
    train_source = directory / "TinyStories-train.txt"
    validation_source = directory / "TinyStories-valid.txt"
    train_verification = download_pinned_file(
        train_source,
        urls=train_urls,
        expected_size=TINYSTORIES_TRAIN_BYTES,
        expected_sha256=TINYSTORIES_TRAIN_SHA256,
        progress=progress,
    )
    validation_verification = download_pinned_file(
        validation_source,
        urls=validation_urls,
        expected_size=TINYSTORIES_VALIDATION_BYTES,
        expected_sha256=TINYSTORIES_VALIDATION_SHA256,
        progress=progress,
    )
    return build_tinystories_corpus(
        train_source,
        validation_source,
        tokenizer_path,
        output_path,
        train_verification=train_verification,
        validation_verification=validation_verification,
        expected_train_stories=TINYSTORIES_TRAIN_STORIES,
        expected_validation_stories=TINYSTORIES_VALIDATION_STORIES,
        expected_vocab_size=_EXPECTED_VOCAB_SIZE,
    )


__all__ = [
    "TinyStoriesCorpusError",
    "build_tinystories_corpus",
    "prepare_tinystories_corpus",
    "validate_tinystories_corpus",
]
