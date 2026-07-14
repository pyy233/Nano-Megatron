"""Strict text ingestion and portable token-corpus artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from nano_megatron.tokenizer import TextTokenizer

_ARTIFACT_NAME = "nano_megatron.token_corpus"
_ARTIFACT_VERSION = 2
_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}
_STRUCTURED_KEYS_V1 = {
    "artifact",
    "format_version",
    "tokens",
    "document_offsets",
    "documents",
    "vocab_size",
    "eos_id",
    "tokenizer_fingerprint",
    "append_eos",
}
_STRUCTURED_KEYS_V2 = _STRUCTURED_KEYS_V1 | {"fingerprint"}


def _compute_token_corpus_fingerprint(
    *,
    append_eos: bool,
    documents: int,
    eos_id: int | None,
    tokenizer_fingerprint: str | None,
    vocab_size: int,
    token_chunks: Iterable[bytes | bytearray | memoryview],
    document_offset_chunks: Iterable[bytes | bytearray | memoryview],
) -> str:
    """Hash one logical corpus independently of its storage container.

    Both the portable ``.pt`` artifact and the mmap backend feed canonical
    little-endian int64 chunks into this function.  A checkpoint can therefore
    identify the data itself instead of treating a storage-format change as a
    different corpus.
    """

    metadata = json.dumps(
        {
            "append_eos": append_eos,
            "artifact": _ARTIFACT_NAME,
            "documents": documents,
            "eos_id": eos_id,
            "format_version": _ARTIFACT_VERSION,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "vocab_size": vocab_size,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(metadata)
    for chunk in token_chunks:
        digest.update(chunk)
    for chunk in document_offset_chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _validate_text_key(text_key: str) -> str:
    if not isinstance(text_key, str):
        raise TypeError("text_key must be a string")
    if not text_key.strip():
        raise ValueError("text_key must not be empty")
    return text_key


def _validate_document_text(text: Any, *, location: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"{location}: text must be a string")
    if not text.strip():
        raise ValueError(f"{location}: text must not be empty")
    return text


def iter_jsonl_text(
    path: str | Path,
    *,
    text_key: str = "text",
) -> Iterator[str]:
    """Yield non-empty strings from a JSONL file with precise error locations."""

    source = Path(path)
    key = _validate_text_key(text_key)
    records = 0
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            records += 1
            location = f"{source}:{line_number}"
            if not line.strip():
                raise ValueError(f"{location}: JSONL record must not be empty")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{location}: invalid JSON ({error.msg} at column {error.colno})"
                ) from error
            if not isinstance(record, dict):
                raise TypeError(f"{location}: JSONL record must be an object")
            if key not in record:
                raise ValueError(f"{location}: missing text key {key!r}")
            yield _validate_document_text(record[key], location=f"{location}:{key}")
    if records == 0:
        raise ValueError(f"{source}:1: JSONL file must contain at least one record")


def _validate_tokenizer(tokenizer: TextTokenizer) -> tuple[int, int, str]:
    vocab_size = tokenizer.vocab_size
    eos_id = tokenizer.eos_token_id
    fingerprint = tokenizer.fingerprint
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 1:
        raise ValueError("tokenizer.vocab_size must be a positive integer")
    if isinstance(eos_id, bool) or not isinstance(eos_id, int):
        raise TypeError("tokenizer.eos_token_id must be an integer")
    if not 0 <= eos_id < vocab_size:
        raise ValueError("tokenizer.eos_token_id must be inside the tokenizer vocabulary")
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise ValueError("tokenizer.fingerprint must be a non-empty string")
    return vocab_size, eos_id, fingerprint


def _validate_token_ids(ids: Any, *, document: int, vocab_size: int) -> list[int]:
    if not isinstance(ids, list):
        raise TypeError(f"document {document}: tokenizer.encode() must return list[int]")
    if not ids:
        raise ValueError(f"document {document}: tokenizer produced no token ids")
    for offset, token_id in enumerate(ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(f"document {document}: token id at offset {offset} must be an integer")
        if not 0 <= token_id < vocab_size:
            raise ValueError(
                f"document {document}: token id {token_id} at offset {offset} "
                f"is outside vocab_size={vocab_size}"
            )
    return ids


@dataclass(frozen=True, slots=True, eq=False)
class TokenCorpus:
    """A flat token stream plus enough metadata to validate document boundaries."""

    tokens: Tensor
    document_offsets: Tensor
    documents: int
    vocab_size: int
    eos_id: int | None
    tokenizer_fingerprint: str | None
    append_eos: bool

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TokenCorpus):
            return NotImplemented
        return (
            torch.equal(self.tokens, other.tokens)
            and torch.equal(self.document_offsets, other.document_offsets)
            and self.documents == other.documents
            and self.vocab_size == other.vocab_size
            and self.eos_id == other.eos_id
            and self.tokenizer_fingerprint == other.tokenizer_fingerprint
            and self.append_eos == other.append_eos
        )

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, Tensor):
            raise TypeError("tokens must be a torch.Tensor")
        if self.tokens.ndim != 1:
            raise ValueError("tokens must be one-dimensional")
        if self.tokens.dtype not in _INTEGER_DTYPES:
            raise TypeError("tokens must use an integer dtype")
        if self.tokens.device.type != "cpu":
            raise ValueError("tokens must reside on CPU")
        if not isinstance(self.document_offsets, Tensor):
            raise TypeError("document_offsets must be a torch.Tensor")
        if self.document_offsets.ndim != 1:
            raise ValueError("document_offsets must be one-dimensional")
        if self.document_offsets.dtype not in _INTEGER_DTYPES:
            raise TypeError("document_offsets must use an integer dtype")
        if self.document_offsets.device.type != "cpu":
            raise ValueError("document_offsets must reside on CPU")

        object.__setattr__(self, "tokens", self.tokens.to(dtype=torch.long).contiguous())
        object.__setattr__(
            self,
            "document_offsets",
            self.document_offsets.to(dtype=torch.long).contiguous(),
        )

        if isinstance(self.documents, bool) or not isinstance(self.documents, int):
            raise TypeError("documents must be an integer")
        if self.documents < 1:
            raise ValueError("documents must be positive")
        if self.document_offsets.numel() != self.documents + 1:
            raise ValueError("document_offsets must contain documents + 1 entries")
        if self.document_offsets[0].item() != 0:
            raise ValueError("document_offsets must start at zero")
        if self.document_offsets[-1].item() != self.tokens.numel():
            raise ValueError("document_offsets must end at the token count")
        if not isinstance(self.append_eos, bool):
            raise TypeError("append_eos must be a boolean")

        differences = self.document_offsets[1:] - self.document_offsets[:-1]
        legacy = self.is_legacy
        if legacy:
            if self.documents != 1 or self.append_eos:
                raise ValueError("legacy token corpora must be one document without appended EOS")
            if torch.any(differences < 0):
                raise ValueError("legacy document_offsets must be non-decreasing")
        elif torch.any(differences <= 0):
            raise ValueError("structured token corpora cannot contain empty documents")

        if isinstance(self.vocab_size, bool) or not isinstance(self.vocab_size, int):
            raise TypeError("vocab_size must be an integer")
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.tokens.numel():
            minimum = int(self.tokens.min().item())
            maximum = int(self.tokens.max().item())
            if minimum < 0 or maximum >= self.vocab_size:
                raise ValueError("tokens contain ids outside vocab_size")

        if legacy:
            if self.eos_id is not None or self.tokenizer_fingerprint is not None:
                raise ValueError("legacy metadata must leave eos_id and fingerprint unset")
            return

        if isinstance(self.eos_id, bool) or not isinstance(self.eos_id, int):
            raise TypeError("eos_id must be an integer")
        if not 0 <= self.eos_id < self.vocab_size:
            raise ValueError("eos_id must be inside vocab_size")
        if (
            not isinstance(self.tokenizer_fingerprint, str)
            or not self.tokenizer_fingerprint.strip()
        ):
            raise ValueError("tokenizer_fingerprint must be a non-empty string")
        if self.append_eos:
            document_ends = self.document_offsets[1:] - 1
            if not torch.all(self.tokens[document_ends] == self.eos_id):
                raise ValueError("every document must end with eos_id when append_eos is true")

    @property
    def is_legacy(self) -> bool:
        """Whether metadata was inferred from the old tensor-only representation."""

        return self.eos_id is None and self.tokenizer_fingerprint is None

    @property
    def fingerprint(self) -> str:
        """Content identity independent of the surrounding ``torch.save`` container."""

        token_bytes = self.tokens.numpy().astype("<i8", copy=False).tobytes(order="C")
        offset_bytes = self.document_offsets.numpy().astype("<i8", copy=False).tobytes(order="C")
        return _compute_token_corpus_fingerprint(
            append_eos=self.append_eos,
            documents=self.documents,
            eos_id=self.eos_id,
            tokenizer_fingerprint=self.tokenizer_fingerprint,
            vocab_size=self.vocab_size,
            token_chunks=(token_bytes,),
            document_offset_chunks=(offset_bytes,),
        )

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        tokenizer: TextTokenizer,
        *,
        append_eos: bool = True,
    ) -> TokenCorpus:
        """Encode documents without implicit BOS/EOS and optionally append one EOS each."""

        if not isinstance(append_eos, bool):
            raise TypeError("append_eos must be a boolean")
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be an iterable of documents, not one string")
        vocab_size, eos_id, fingerprint = _validate_tokenizer(tokenizer)
        flat_tokens: list[int] = []
        offsets = [0]
        documents = 0
        for documents, text in enumerate(texts, start=1):
            text = _validate_document_text(text, location=f"document {documents}")
            ids = _validate_token_ids(
                tokenizer.encode(text, add_bos=False, add_eos=False),
                document=documents,
                vocab_size=vocab_size,
            )
            flat_tokens.extend(ids)
            if append_eos:
                flat_tokens.append(eos_id)
            offsets.append(len(flat_tokens))
        if documents == 0:
            raise ValueError("at least one document is required")
        return cls(
            tokens=torch.tensor(flat_tokens, dtype=torch.long),
            document_offsets=torch.tensor(offsets, dtype=torch.long),
            documents=documents,
            vocab_size=vocab_size,
            eos_id=eos_id,
            tokenizer_fingerprint=fingerprint,
            append_eos=append_eos,
        )

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        tokenizer: TextTokenizer,
        *,
        text_key: str = "text",
        append_eos: bool = True,
    ) -> TokenCorpus:
        return cls.from_texts(
            iter_jsonl_text(path, text_key=text_key),
            tokenizer,
            append_eos=append_eos,
        )

    def save(self, path: str | Path) -> None:
        """Save a versioned artifact that remains loadable with ``weights_only=True``."""

        if self.is_legacy:
            raise ValueError("legacy token tensors cannot be saved as structured corpora")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact": _ARTIFACT_NAME,
            "format_version": _ARTIFACT_VERSION,
            "tokens": self.tokens,
            "document_offsets": self.document_offsets,
            "documents": self.documents,
            "vocab_size": self.vocab_size,
            "eos_id": self.eos_id,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "append_eos": self.append_eos,
            "fingerprint": self.fingerprint,
        }
        try:
            stream = target.open("xb")
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite token corpus: {target}") from error
        try:
            with stream:
                torch.save(payload, stream)
        except BaseException:
            target.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str | Path) -> TokenCorpus:
        """Load a structured corpus or adapt the two legacy tensor-only formats."""

        source = Path(path)
        loaded = torch.load(source, map_location="cpu", weights_only=True)
        if isinstance(loaded, Tensor):
            return cls._from_legacy_tokens(loaded)
        if not isinstance(loaded, dict):
            raise TypeError(
                "token file must contain TokenCorpus data, a Tensor, or {'tokens': Tensor}"
            )
        if set(loaded) == {"tokens"}:
            tokens = loaded["tokens"]
            if not isinstance(tokens, Tensor):
                raise TypeError("legacy {'tokens': ...} value must be a Tensor")
            return cls._from_legacy_tokens(tokens)
        artifact = loaded.get("artifact")
        if not isinstance(artifact, str) or artifact != _ARTIFACT_NAME:
            raise ValueError(f"unsupported token artifact: {artifact!r}")
        version = loaded.get("format_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError(f"unsupported TokenCorpus format_version: {version!r}")
        expected_keys = {
            1: _STRUCTURED_KEYS_V1,
            _ARTIFACT_VERSION: _STRUCTURED_KEYS_V2,
        }.get(version)
        if expected_keys is None:
            raise ValueError(f"unsupported TokenCorpus format_version: {version!r}")
        keys = set(loaded)
        if keys != expected_keys:
            missing = sorted(expected_keys - keys)
            unknown = sorted(keys - expected_keys)
            details = []
            if missing:
                details.append(f"missing keys: {', '.join(missing)}")
            if unknown:
                details.append(f"unknown keys: {', '.join(unknown)}")
            raise ValueError("invalid TokenCorpus artifact (" + "; ".join(details) + ")")
        corpus = cls(
            tokens=loaded["tokens"],
            document_offsets=loaded["document_offsets"],
            documents=loaded["documents"],
            vocab_size=loaded["vocab_size"],
            eos_id=loaded["eos_id"],
            tokenizer_fingerprint=loaded["tokenizer_fingerprint"],
            append_eos=loaded["append_eos"],
        )
        if version == _ARTIFACT_VERSION:
            expected_fingerprint = loaded["fingerprint"]
            if not isinstance(expected_fingerprint, str) or len(expected_fingerprint) != 64:
                raise ValueError("TokenCorpus fingerprint must be a SHA256 hex digest")
            try:
                bytes.fromhex(expected_fingerprint)
            except ValueError as error:
                raise ValueError("TokenCorpus fingerprint must be a SHA256 hex digest") from error
            if corpus.fingerprint != expected_fingerprint:
                raise ValueError("TokenCorpus content fingerprint does not match its payload")
        return corpus

    @classmethod
    def _from_legacy_tokens(cls, tokens: Tensor) -> TokenCorpus:
        if not isinstance(tokens, Tensor):
            raise TypeError("legacy tokens must be a Tensor")
        if tokens.ndim != 1:
            raise ValueError("legacy tokens must be one-dimensional")
        if tokens.dtype not in _INTEGER_DTYPES:
            raise TypeError("legacy tokens must use an integer dtype")
        tokens = tokens.to(dtype=torch.long, device="cpu").contiguous()
        if tokens.numel() and int(tokens.min().item()) < 0:
            raise ValueError("legacy tokens must be non-negative")
        vocab_size = int(tokens.max().item()) + 1 if tokens.numel() else 1
        return cls(
            tokens=tokens,
            document_offsets=torch.tensor([0, tokens.numel()], dtype=torch.long),
            documents=1,
            vocab_size=vocab_size,
            eos_id=None,
            tokenizer_fingerprint=None,
            append_eos=False,
        )


def build_token_corpus(
    texts: Iterable[str],
    tokenizer: TextTokenizer,
    *,
    append_eos: bool = True,
) -> TokenCorpus:
    """Encode an iterable of documents into one validated token corpus."""

    return TokenCorpus.from_texts(texts, tokenizer, append_eos=append_eos)


def preprocess_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    tokenizer: TextTokenizer,
    *,
    text_key: str = "text",
    append_eos: bool = True,
) -> TokenCorpus:
    """Encode a JSONL file, save its structured artifact, and return summary metadata."""

    corpus = TokenCorpus.from_jsonl(
        input_path,
        tokenizer,
        text_key=text_key,
        append_eos=append_eos,
    )
    corpus.save(output_path)
    return corpus
