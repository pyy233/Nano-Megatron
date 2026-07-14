"""Stream JSONL into a compact, validated, read-only mmap token corpus."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from nano_megatron.tokenizer import TextTokenizer

from .corpus import (
    _compute_token_corpus_fingerprint,
    _validate_token_ids,
    _validate_tokenizer,
    iter_jsonl_text,
)

_ARTIFACT_NAME = "nano_megatron.mmap_token_corpus"
_ARTIFACT_VERSION = 1
_TOKENS_FILE = "tokens.bin"
_OFFSETS_FILE = "document_offsets.bin"
_METADATA_FILE = "metadata.json"
_OFFSET_DTYPE = np.dtype("<u8")
_UINT16_DTYPE = np.dtype("<u2")
_UINT32_DTYPE = np.dtype("<u4")
_MAX_UINT16_VOCAB = 1 << 16
_MAX_UINT32_VOCAB = 1 << 32
_TOKEN_SCAN_ITEMS = 1 << 20
_METADATA_KEYS = {
    "append_eos",
    "artifact",
    "document_offsets_dtype",
    "document_offsets_sha256",
    "documents",
    "eos_id",
    "fingerprint",
    "format_version",
    "token_count",
    "token_dtype",
    "tokenizer_fingerprint",
    "tokens_sha256",
    "vocab_size",
}


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    status = path.stat()
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)


class MMapCorpusError(ValueError):
    """Raised when an mmap corpus exists but violates its artifact contract."""


def _token_dtype_for_vocab(vocab_size: int) -> np.dtype[Any]:
    if vocab_size <= _MAX_UINT16_VOCAB:
        return _UINT16_DTYPE
    if vocab_size <= _MAX_UINT32_VOCAB:
        return _UINT32_DTYPE
    raise ValueError(
        f"mmap token corpora support vocab_size <= {_MAX_UINT32_VOCAB}, got {vocab_size}"
    )


def _require_integer(
    metadata: dict[str, Any],
    key: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise MMapCorpusError(f"mmap metadata field {key!r} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f">= {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        raise MMapCorpusError(f"mmap metadata field {key!r} must be {bounds}")
    return value


def _require_string(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MMapCorpusError(f"mmap metadata field {key!r} must be a non-empty string")
    return value


def _require_sha256(metadata: dict[str, Any], key: str) -> str:
    value = _require_string(metadata, key)
    if len(value) != 64:
        raise MMapCorpusError(f"mmap metadata field {key!r} must be a SHA256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise MMapCorpusError(f"mmap metadata field {key!r} must be a SHA256 hex digest") from error
    return value


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MMapCorpusError(f"could not read mmap corpus metadata {path}: {error}") from error
    if not isinstance(payload, dict):
        raise MMapCorpusError("mmap corpus metadata must contain a JSON object")
    keys = set(payload)
    if keys != _METADATA_KEYS:
        missing = sorted(_METADATA_KEYS - keys)
        unknown = sorted(keys - _METADATA_KEYS)
        details = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown keys: {', '.join(unknown)}")
        raise MMapCorpusError("invalid mmap corpus metadata (" + "; ".join(details) + ")")
    return payload


def _canonical_token_chunks(
    tokens: np.memmap[Any, Any],
) -> Iterator[bytes]:
    for start in range(0, int(tokens.size), _TOKEN_SCAN_ITEMS):
        chunk = tokens[start : start + _TOKEN_SCAN_ITEMS]
        yield np.asarray(chunk, dtype="<i8").tobytes(order="C")


def _logical_fingerprint(
    *,
    tokens: np.memmap[Any, Any],
    document_offsets: np.ndarray[Any, Any],
    append_eos: bool,
    documents: int,
    eos_id: int,
    tokenizer_fingerprint: str,
    vocab_size: int,
) -> str:
    offsets = memoryview(document_offsets).cast("B")
    return _compute_token_corpus_fingerprint(
        append_eos=append_eos,
        documents=documents,
        eos_id=eos_id,
        tokenizer_fingerprint=tokenizer_fingerprint,
        vocab_size=vocab_size,
        token_chunks=_canonical_token_chunks(tokens),
        document_offset_chunks=(offsets,),
    )


class MMapTokenCorpus:
    """Validated metadata plus a lazily opened read-only token mmap.

    ``document_offsets`` is intentionally loaded into ordinary memory: it is a
    small index, while ``tokens.bin`` is the single potentially large payload.
    Pickling never serializes or retains an open mmap, so spawned DataLoader
    workers reopen the file locally on first access.
    """

    def __init__(
        self,
        *,
        path: Path,
        token_dtype: np.dtype[Any],
        token_count: int,
        document_offsets: np.ndarray[Any, Any],
        documents: int,
        vocab_size: int,
        eos_id: int,
        tokenizer_fingerprint: str,
        append_eos: bool,
        tokens_sha256: str,
        document_offsets_sha256: str,
        fingerprint: str,
        token_file_identity: tuple[int, int, int, int],
    ) -> None:
        self.path = path
        self.token_dtype = token_dtype
        self.token_count = token_count
        document_offsets.flags.writeable = False
        self.document_offsets = document_offsets
        self.documents = documents
        self.vocab_size = vocab_size
        self.eos_id = eos_id
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self.append_eos = append_eos
        self.tokens_sha256 = tokens_sha256
        self.document_offsets_sha256 = document_offsets_sha256
        self.fingerprint = fingerprint
        self._token_file_identity = token_file_identity
        self._tokens: np.memmap[Any, Any] | None = None

    @property
    def is_legacy(self) -> bool:
        return False

    @property
    def tokens(self) -> np.memmap[Any, Any]:
        """Open and cache ``tokens.bin`` in read-only mode on first access."""

        if self._tokens is None:
            tokens_path = self.path / _TOKENS_FILE
            if _file_identity(tokens_path) != self._token_file_identity:
                raise MMapCorpusError(
                    "tokens.bin changed after the mmap corpus was validated; reload the artifact"
                )
            self._tokens = np.memmap(
                tokens_path,
                dtype=self.token_dtype,
                mode="r",
                shape=(self.token_count,),
            )
            self._tokens.flags.writeable = False
        return self._tokens

    def read_tokens(self, start: int, stop: int) -> np.ndarray[Any, Any]:
        """Return a read-only mmap view over ``[start, stop)``."""

        for name, value in (("start", start), ("stop", stop)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if start < 0 or stop < start or stop > self.token_count:
            raise IndexError(
                f"invalid token range [{start}, {stop}) for token_count={self.token_count}"
            )
        return self.tokens[start:stop]

    def close(self) -> None:
        """Release this object's mapping reference; future reads reopen it lazily.

        NumPy slices keep their base memmap alive.  Deliberately avoid closing
        its private ``_mmap`` handle here: doing so would invalidate an escaped
        view and can crash the interpreter instead of raising a Python error.
        """

        self._tokens = None

    def __len__(self) -> int:
        return self.token_count

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_tokens"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        offsets = self.document_offsets
        if not isinstance(offsets, np.ndarray):
            raise TypeError("pickled mmap document_offsets must be a NumPy array")
        offsets.flags.writeable = False
        self._tokens = None

    @classmethod
    def load(cls, path: str | Path) -> MMapTokenCorpus:
        """Load and fully validate one mmap artifact without retaining a mapping."""

        directory = Path(path).resolve()
        if not directory.exists():
            raise FileNotFoundError(f"mmap corpus directory does not exist: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"mmap corpus path is not a directory: {directory}")

        expected_files = {_TOKENS_FILE, _OFFSETS_FILE, _METADATA_FILE}
        actual_files = {entry.name for entry in directory.iterdir()}
        if actual_files != expected_files:
            missing = sorted(expected_files - actual_files)
            unknown = sorted(actual_files - expected_files)
            details = []
            if missing:
                details.append(f"missing files: {', '.join(missing)}")
            if unknown:
                details.append(f"unknown files: {', '.join(unknown)}")
            raise MMapCorpusError("invalid mmap corpus directory (" + "; ".join(details) + ")")
        for filename in sorted(expected_files):
            artifact_file = directory / filename
            if artifact_file.is_symlink() or not artifact_file.is_file():
                raise MMapCorpusError(
                    f"mmap corpus sidecar must be a regular file: {artifact_file}"
                )

        metadata = _read_metadata(directory / _METADATA_FILE)
        if metadata.get("artifact") != _ARTIFACT_NAME:
            raise MMapCorpusError(f"unsupported mmap corpus artifact: {metadata.get('artifact')!r}")
        version = _require_integer(metadata, "format_version", minimum=1)
        if version != _ARTIFACT_VERSION:
            raise MMapCorpusError(
                f"unsupported mmap corpus format_version {version}; expected {_ARTIFACT_VERSION}"
            )

        vocab_size = _require_integer(
            metadata,
            "vocab_size",
            minimum=1,
            maximum=_MAX_UINT32_VOCAB,
        )
        token_dtype_text = _require_string(metadata, "token_dtype")
        expected_dtype = _token_dtype_for_vocab(vocab_size)
        if token_dtype_text != expected_dtype.str:
            raise MMapCorpusError(
                "mmap token dtype is not the canonical dtype for vocab_size: "
                f"{token_dtype_text!r} != {expected_dtype.str!r}"
            )
        if metadata.get("document_offsets_dtype") != _OFFSET_DTYPE.str:
            raise MMapCorpusError(f"document_offsets_dtype must be {_OFFSET_DTYPE.str!r}")

        token_count = _require_integer(
            metadata,
            "token_count",
            minimum=1,
            maximum=np.iinfo(np.int64).max,
        )
        documents = _require_integer(
            metadata,
            "documents",
            minimum=1,
            maximum=token_count,
        )
        eos_id = _require_integer(metadata, "eos_id", minimum=0, maximum=vocab_size - 1)
        tokenizer_fingerprint = _require_string(metadata, "tokenizer_fingerprint")
        append_eos = metadata.get("append_eos")
        if not isinstance(append_eos, bool):
            raise MMapCorpusError("mmap metadata field 'append_eos' must be a boolean")
        expected_tokens_sha256 = _require_sha256(metadata, "tokens_sha256")
        expected_offsets_sha256 = _require_sha256(metadata, "document_offsets_sha256")
        expected_fingerprint = _require_sha256(metadata, "fingerprint")

        tokens_path = directory / _TOKENS_FILE
        offsets_path = directory / _OFFSETS_FILE
        token_file_identity = _file_identity(tokens_path)
        expected_token_bytes = token_count * expected_dtype.itemsize
        if tokens_path.stat().st_size != expected_token_bytes:
            raise MMapCorpusError(
                "tokens.bin size does not match token_count and token_dtype: "
                f"{tokens_path.stat().st_size} != {expected_token_bytes}"
            )
        expected_offset_bytes = (documents + 1) * _OFFSET_DTYPE.itemsize
        if offsets_path.stat().st_size != expected_offset_bytes:
            raise MMapCorpusError(
                "document_offsets.bin size does not match documents: "
                f"{offsets_path.stat().st_size} != {expected_offset_bytes}"
            )
        document_offsets = np.fromfile(offsets_path, dtype=_OFFSET_DTYPE)
        offsets_bytes = memoryview(document_offsets).cast("B")
        if hashlib.sha256(offsets_bytes).hexdigest() != expected_offsets_sha256:
            raise MMapCorpusError("document_offsets.bin SHA256 does not match mmap metadata")
        if int(document_offsets[0]) != 0:
            raise MMapCorpusError("document offsets must start at zero")
        if int(document_offsets[-1]) != token_count:
            raise MMapCorpusError("document offsets must end at token_count")
        if np.any(document_offsets[1:] <= document_offsets[:-1]):
            raise MMapCorpusError("mmap token corpus cannot contain empty documents")
        document_offsets.flags.writeable = False

        validation_tokens = np.memmap(
            tokens_path,
            dtype=expected_dtype,
            mode="r",
            shape=(token_count,),
        )
        try:
            maximum_token_id = 0
            tokens_digest = hashlib.sha256()

            def validated_token_chunks() -> Iterator[bytes]:
                nonlocal maximum_token_id
                for start in range(0, token_count, _TOKEN_SCAN_ITEMS):
                    chunk = validation_tokens[start : start + _TOKEN_SCAN_ITEMS]
                    tokens_digest.update(chunk.tobytes(order="C"))
                    maximum_token_id = max(maximum_token_id, int(chunk.max()))
                    yield np.asarray(chunk, dtype="<i8").tobytes(order="C")

            actual_fingerprint = _compute_token_corpus_fingerprint(
                append_eos=append_eos,
                documents=documents,
                eos_id=eos_id,
                tokenizer_fingerprint=tokenizer_fingerprint,
                vocab_size=vocab_size,
                token_chunks=validated_token_chunks(),
                document_offset_chunks=(offsets_bytes,),
            )
            if tokens_digest.hexdigest() != expected_tokens_sha256:
                raise MMapCorpusError("tokens.bin SHA256 does not match mmap metadata")
            if maximum_token_id >= vocab_size:
                raise MMapCorpusError(
                    f"tokens.bin contains token id {maximum_token_id} outside vocab_size"
                )
            if append_eos:
                ends = document_offsets[1:] - np.uint64(1)
                for start in range(0, documents, _TOKEN_SCAN_ITEMS):
                    indices = np.asarray(ends[start : start + _TOKEN_SCAN_ITEMS], dtype=np.int64)
                    if np.any(validation_tokens[indices] != eos_id):
                        raise MMapCorpusError(
                            "every document must end with eos_id when append_eos is true"
                        )
        finally:
            validation_tokens._mmap.close()
        if actual_fingerprint != expected_fingerprint:
            raise MMapCorpusError("mmap corpus logical fingerprint does not match its payload")
        if _file_identity(tokens_path) != token_file_identity:
            raise MMapCorpusError("tokens.bin changed while the mmap corpus was being validated")

        return cls(
            path=directory,
            token_dtype=expected_dtype,
            token_count=token_count,
            document_offsets=document_offsets,
            documents=documents,
            vocab_size=vocab_size,
            eos_id=eos_id,
            tokenizer_fingerprint=tokenizer_fingerprint,
            append_eos=append_eos,
            tokens_sha256=expected_tokens_sha256,
            document_offsets_sha256=expected_offsets_sha256,
            fingerprint=actual_fingerprint,
            token_file_identity=token_file_identity,
        )


def _validate_save_target(directory: Path) -> bool:
    """Validate the destination and return whether it is an existing empty directory."""

    if directory.is_symlink():
        raise FileExistsError(f"refusing to replace mmap corpus symlink: {directory}")
    if not directory.exists():
        return False
    if not directory.is_dir():
        raise FileExistsError(f"mmap corpus path already exists: {directory}")
    if any(directory.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty mmap corpus directory: {directory}")
    return True


def preprocess_jsonl_mmap(
    input_path: str | Path,
    output_path: str | Path,
    tokenizer: TextTokenizer,
    *,
    text_key: str = "text",
    append_eos: bool = True,
) -> MMapTokenCorpus:
    """Stream one JSONL corpus into an atomically published mmap artifact."""

    if not isinstance(append_eos, bool):
        raise TypeError("append_eos must be a boolean")
    vocab_size, eos_id, tokenizer_fingerprint = _validate_tokenizer(tokenizer)
    token_dtype = _token_dtype_for_vocab(vocab_size)
    destination = Path(output_path)
    destination_was_empty = _validate_save_target(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    removed_empty_destination = False
    validated: MMapTokenCorpus | None = None
    try:
        tokens_digest = hashlib.sha256()
        offsets_digest = hashlib.sha256()
        token_count = 0
        documents = 0
        tokens_path = temporary / _TOKENS_FILE
        offsets_path = temporary / _OFFSETS_FILE
        with tokens_path.open("xb") as token_stream, offsets_path.open("xb") as offset_stream:
            initial_offset = np.asarray([0], dtype=_OFFSET_DTYPE).tobytes(order="C")
            offset_stream.write(initial_offset)
            offsets_digest.update(initial_offset)
            for documents, text in enumerate(
                iter_jsonl_text(input_path, text_key=text_key),
                start=1,
            ):
                token_ids = _validate_token_ids(
                    tokenizer.encode(text, add_bos=False, add_eos=False),
                    document=documents,
                    vocab_size=vocab_size,
                )
                encoded = np.asarray(token_ids, dtype=token_dtype).tobytes(order="C")
                token_stream.write(encoded)
                tokens_digest.update(encoded)
                token_count += len(token_ids)
                if append_eos:
                    encoded_eos = np.asarray([eos_id], dtype=token_dtype).tobytes(order="C")
                    token_stream.write(encoded_eos)
                    tokens_digest.update(encoded_eos)
                    token_count += 1
                if token_count > np.iinfo(np.int64).max:
                    raise OverflowError("mmap corpus token_count exceeds signed int64 capacity")
                offset = np.asarray([token_count], dtype=_OFFSET_DTYPE).tobytes(order="C")
                offset_stream.write(offset)
                offsets_digest.update(offset)

        document_offsets = np.fromfile(offsets_path, dtype=_OFFSET_DTYPE)
        fingerprint_tokens = np.memmap(
            tokens_path,
            dtype=token_dtype,
            mode="r",
            shape=(token_count,),
        )
        try:
            fingerprint = _logical_fingerprint(
                tokens=fingerprint_tokens,
                document_offsets=document_offsets,
                append_eos=append_eos,
                documents=documents,
                eos_id=eos_id,
                tokenizer_fingerprint=tokenizer_fingerprint,
                vocab_size=vocab_size,
            )
        finally:
            fingerprint_tokens._mmap.close()

        metadata = {
            "append_eos": append_eos,
            "artifact": _ARTIFACT_NAME,
            "document_offsets_dtype": _OFFSET_DTYPE.str,
            "document_offsets_sha256": offsets_digest.hexdigest(),
            "documents": documents,
            "eos_id": eos_id,
            "fingerprint": fingerprint,
            "format_version": _ARTIFACT_VERSION,
            "token_count": token_count,
            "token_dtype": token_dtype.str,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "tokens_sha256": tokens_digest.hexdigest(),
            "vocab_size": vocab_size,
        }
        (temporary / _METADATA_FILE).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validated = MMapTokenCorpus.load(temporary)
        validated.close()
        if validated.fingerprint != fingerprint:
            raise MMapCorpusError("mmap corpus fingerprint changed during self-validation")

        if destination_was_empty:
            destination.rmdir()
            removed_empty_destination = True
        elif destination.exists() or destination.is_symlink():
            raise FileExistsError(f"mmap corpus path already exists: {destination}")
        os.replace(temporary, destination)
        validated.path = destination.resolve()
    except BaseException:
        if removed_empty_destination and not destination.exists():
            destination.mkdir()
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    if validated is None:  # pragma: no cover - every successful path self-validates first.
        raise AssertionError("mmap corpus publish completed without a validated artifact")
    return validated
