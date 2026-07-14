"""A standalone Hugging Face byte-level BPE tokenizer.

Artifacts are directories rather than bare JSON files so the executable
tokenizer and the small amount of validation metadata travel together.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .base import ByteBPETrainingConfig, SpecialTokens

_ARTIFACT_FORMAT = "nano-megatron-byte-level-bpe"
_ARTIFACT_VERSION = 1
_TOKENIZER_FILE = "tokenizer.json"
_METADATA_FILE = "metadata.json"
_SPECIAL_ROLES = ("unk", "bos", "eos", "pad")


class TokenizerArtifactError(ValueError):
    """Raised when an artifact exists but violates the tokenizer contract."""


def _require_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def _checked_training_texts(texts: Iterable[str]) -> Iterator[str]:
    if isinstance(texts, (str, bytes)):
        raise TypeError("training texts must be an iterable of strings, not one string")
    try:
        iterator = iter(texts)
    except TypeError as error:
        raise TypeError("training texts must be an iterable of strings") from error

    try:
        first = next(iterator)
    except StopIteration as error:
        raise ValueError("training texts cannot be empty") from error

    def validate(text: Any, index: int) -> str:
        if not isinstance(text, str):
            raise TypeError(
                f"training text at index {index} must be a string, got {type(text).__name__}"
            )
        return text

    yield validate(first, 0)
    for index, text in enumerate(iterator, start=1):
        yield validate(text, index)


def _backend_payload(backend: Tokenizer) -> dict[str, Any]:
    try:
        payload = json.loads(backend.to_str())
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise TokenizerArtifactError("tokenizer backend did not produce valid JSON") from error
    if not isinstance(payload, dict):
        raise TokenizerArtifactError("tokenizer backend JSON must contain an object")
    return payload


def _fingerprint(backend: Tokenizer) -> str:
    canonical = json.dumps(
        _backend_payload(backend),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TokenizerArtifactError(
            f"could not read valid {description} from {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise TokenizerArtifactError(f"{description} in {path} must be a JSON object")
    return payload


def _metadata_mapping(metadata: dict[str, Any], key: str) -> dict[str, Any]:
    value = metadata.get(key)
    if not isinstance(value, dict):
        raise TokenizerArtifactError(f"metadata field {key!r} must be an object")
    missing = set(_SPECIAL_ROLES).difference(value)
    extra = set(value).difference(_SPECIAL_ROLES)
    if missing or extra:
        raise TokenizerArtifactError(
            f"metadata field {key!r} must contain exactly {', '.join(_SPECIAL_ROLES)}"
        )
    return value


class ByteLevelBPETokenizer:
    """Byte-level BPE with explicit special-token and artifact semantics."""

    def __init__(self, backend: Tokenizer, special_tokens: SpecialTokens) -> None:
        self._backend = backend
        # Treat strings such as ``<|endoftext|>`` in ordinary documents as text bytes.
        # BOS/EOS are introduced only by the explicit flags below, which guarantees that
        # preprocessing appends exactly one semantic EOD even when a document contains a
        # special-token-looking substring.
        self._backend.encode_special_tokens = True
        self._special_tokens = special_tokens
        self._validate_backend_contract()
        self._special_token_ids = self._resolve_special_token_ids()
        self._validate_special_token_text_isolation()
        self._fingerprint = _fingerprint(backend)

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        config: ByteBPETrainingConfig,
    ) -> ByteLevelBPETokenizer:
        """Train from a streaming iterable without normalizing the input text."""

        if not isinstance(config, ByteBPETrainingConfig):
            raise TypeError("config must be a ByteBPETrainingConfig instance")

        backend = Tokenizer(models.BPE(unk_token=config.special_tokens.unk))
        backend.encode_special_tokens = True
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=config.vocab_size,
            min_frequency=config.min_frequency,
            show_progress=config.show_progress,
            special_tokens=list(config.special_tokens.ordered()),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        try:
            backend.train_from_iterator(_checked_training_texts(texts), trainer=trainer)
        except (TypeError, ValueError):
            raise
        except Exception as error:
            raise RuntimeError(f"byte-level BPE training failed: {error}") from error
        return cls(backend, config.special_tokens)

    @classmethod
    def load(cls, artifact_dir: str | Path) -> ByteLevelBPETokenizer:
        """Load and validate both files in a tokenizer artifact directory."""

        directory = Path(artifact_dir)
        if not directory.exists():
            raise FileNotFoundError(f"tokenizer artifact directory does not exist: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"tokenizer artifact path is not a directory: {directory}")

        tokenizer_path = directory / _TOKENIZER_FILE
        metadata_path = directory / _METADATA_FILE
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"tokenizer artifact is missing {tokenizer_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"tokenizer artifact is missing {metadata_path}")

        metadata = _read_json_object(metadata_path, "tokenizer metadata")
        cls._validate_metadata_header(metadata)
        special_tokens = cls._special_tokens_from_metadata(metadata)
        expected_ids = cls._special_ids_from_metadata(metadata)

        try:
            backend = Tokenizer.from_file(str(tokenizer_path))
        except Exception as error:
            raise TokenizerArtifactError(
                f"could not load tokenizer JSON from {tokenizer_path}: {error}"
            ) from error
        tokenizer = cls(backend, special_tokens)

        expected_vocab_size = metadata.get("vocab_size")
        if (
            isinstance(expected_vocab_size, bool)
            or not isinstance(expected_vocab_size, int)
            or expected_vocab_size < 1
        ):
            raise TokenizerArtifactError("metadata field 'vocab_size' must be a positive integer")
        if tokenizer.vocab_size != expected_vocab_size:
            raise TokenizerArtifactError(
                "tokenizer vocabulary size does not match metadata: "
                f"{tokenizer.vocab_size} != {expected_vocab_size}"
            )

        actual_ids = tokenizer._special_token_ids
        if actual_ids != expected_ids:
            raise TokenizerArtifactError(
                f"tokenizer special-token ids do not match metadata: {actual_ids} != {expected_ids}"
            )

        expected_fingerprint = metadata.get("fingerprint")
        if not isinstance(expected_fingerprint, str) or len(expected_fingerprint) != 64:
            raise TokenizerArtifactError(
                "metadata field 'fingerprint' must be a 64-character SHA256 hex digest"
            )
        try:
            bytes.fromhex(expected_fingerprint)
        except ValueError as error:
            raise TokenizerArtifactError(
                "metadata field 'fingerprint' must be a SHA256 hex digest"
            ) from error
        if tokenizer.fingerprint != expected_fingerprint:
            raise TokenizerArtifactError(
                "tokenizer fingerprint does not match metadata: "
                f"{tokenizer.fingerprint} != {expected_fingerprint}"
            )
        return tokenizer

    def save(self, artifact_dir: str | Path) -> None:
        """Atomically save without overwriting an existing non-empty directory."""

        directory = Path(artifact_dir)
        self.validate_save_target(directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        destination_was_empty = directory.is_dir()
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{directory.name}.tmp-",
                dir=directory.parent,
            )
        )
        removed_empty_destination = False
        try:
            tokenizer_path = temporary / _TOKENIZER_FILE
            metadata_path = temporary / _METADATA_FILE
            self._backend.save(str(tokenizer_path), pretty=True)
            metadata_path.write_text(
                json.dumps(self._metadata(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            validated = type(self).load(temporary)
            if validated.fingerprint != self.fingerprint:
                raise TokenizerArtifactError(
                    "saved tokenizer fingerprint changed during artifact validation"
                )

            if destination_was_empty:
                # rmdir is intentionally strict: a concurrent writer makes it fail rather than
                # allowing us to overwrite newly-created user data.
                directory.rmdir()
                removed_empty_destination = True
            elif directory.exists():
                raise FileExistsError(f"tokenizer artifact path already exists: {directory}")
            os.replace(temporary, directory)
        except BaseException:
            if removed_empty_destination and not directory.exists():
                directory.mkdir()
            raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    @staticmethod
    def validate_save_target(artifact_dir: str | Path) -> None:
        """Fail fast if saving would overwrite user data."""

        directory = Path(artifact_dir)
        if not directory.exists():
            return
        if not directory.is_dir():
            raise FileExistsError(f"tokenizer artifact path already exists: {directory}")
        if any(directory.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite non-empty tokenizer artifact directory: {directory}"
            )

    @property
    def special_tokens(self) -> SpecialTokens:
        return self._special_tokens

    @property
    def vocab_size(self) -> int:
        return int(self._backend.get_vocab_size(with_added_tokens=True))

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def unk_token_id(self) -> int:
        return self._special_token_ids["unk"]

    @property
    def bos_token_id(self) -> int:
        return self._special_token_ids["bos"]

    @property
    def eos_token_id(self) -> int:
        return self._special_token_ids["eos"]

    @property
    def pad_token_id(self) -> int:
        return self._special_token_ids["pad"]

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        _require_bool("add_bos", add_bos)
        _require_bool("add_eos", add_eos)
        token_ids = list(self._backend.encode(text, add_special_tokens=False).ids)
        if add_bos:
            token_ids.insert(0, self.bos_token_id)
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[list[int]]:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence of strings")
        _require_bool("add_bos", add_bos)
        _require_bool("add_eos", add_eos)
        materialized = list(texts)
        for index, text in enumerate(materialized):
            if not isinstance(text, str):
                raise TypeError(
                    f"text at index {index} must be a string, got {type(text).__name__}"
                )
        encodings = self._backend.encode_batch(materialized, add_special_tokens=False)
        batches = [list(encoding.ids) for encoding in encodings]
        if add_bos:
            batches = [[self.bos_token_id, *token_ids] for token_ids in batches]
        if add_eos:
            batches = [[*token_ids, self.eos_token_id] for token_ids in batches]
        return batches

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence):
            raise TypeError("token_ids must be a sequence of integers")
        _require_bool("skip_special_tokens", skip_special_tokens)
        ids = list(token_ids)
        for index, token_id in enumerate(ids):
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError(f"token id at index {index} must be an integer")
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(
                    f"token id at index {index} is outside [0, {self.vocab_size}): {token_id}"
                )
        return self._backend.decode(ids, skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> int | None:
        if not isinstance(token, str):
            raise TypeError("token must be a string")
        return self._backend.token_to_id(token)

    def id_to_token(self, token_id: int) -> str | None:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("token_id must be an integer")
        if not 0 <= token_id < self.vocab_size:
            return None
        return self._backend.id_to_token(token_id)

    def _resolve_special_token_ids(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for role, token in self.special_tokens.as_dict().items():
            token_id = self._backend.token_to_id(token)
            if token_id is None:
                raise TokenizerArtifactError(
                    f"tokenizer vocabulary is missing the {role!r} special token {token!r}"
                )
            result[role] = int(token_id)
        if len(set(result.values())) != len(result):
            raise TokenizerArtifactError("special tokens must map to unique token ids")
        return result

    def _validate_special_token_text_isolation(self) -> None:
        """Ensure ordinary text cannot silently become a semantic control token.

        A custom multi-character special token can also be learned as one BPE merge.  In that
        case ``encode_special_tokens=True`` still emits the added special id for ordinary text,
        and the default decoder removes it.  Reject such artifacts after training and on load.
        """

        special_ids = set(self._special_token_ids.values())
        for role, token in self.special_tokens.as_dict().items():
            ordinary_ids = self._backend.encode(token, add_special_tokens=False).ids
            collisions = special_ids.intersection(ordinary_ids)
            if collisions:
                raise TokenizerArtifactError(
                    f"{role} special token {token!r} collides with ordinary text encoding "
                    f"through token id(s) {sorted(collisions)}; choose a control string that "
                    "the byte-level BPE model does not learn as ordinary text"
                )

    def _validate_backend_contract(self) -> None:
        payload = _backend_payload(self._backend)
        model = payload.get("model")
        if not isinstance(model, dict) or model.get("type") != "BPE":
            raise TokenizerArtifactError("tokenizer model must be BPE")
        if model.get("unk_token") != self.special_tokens.unk:
            raise TokenizerArtifactError(
                "BPE unk_token does not match the configured unk special token"
            )
        pre_tokenizer = payload.get("pre_tokenizer")
        if not isinstance(pre_tokenizer, dict) or pre_tokenizer.get("type") != "ByteLevel":
            raise TokenizerArtifactError("tokenizer pre-tokenizer must be ByteLevel")
        decoder = payload.get("decoder")
        if not isinstance(decoder, dict) or decoder.get("type") != "ByteLevel":
            raise TokenizerArtifactError("tokenizer decoder must be ByteLevel")
        if payload.get("normalizer") is not None:
            raise TokenizerArtifactError("byte-level tokenizer must not normalize input text")
        if payload.get("post_processor") is not None:
            raise TokenizerArtifactError(
                "byte-level tokenizer uses explicit BOS/EOS flags and must not have "
                "a post-processor"
            )
        if payload.get("truncation") is not None or payload.get("padding") is not None:
            raise TokenizerArtifactError("tokenizer artifact must not enable truncation or padding")

        added_tokens = payload.get("added_tokens")
        if not isinstance(added_tokens, list):
            raise TokenizerArtifactError("tokenizer JSON is missing its added special tokens")
        by_content = {
            item.get("content"): item
            for item in added_tokens
            if isinstance(item, dict) and isinstance(item.get("content"), str)
        }
        for role, token in self.special_tokens.as_dict().items():
            entry = by_content.get(token)
            if entry is None or entry.get("special") is not True:
                raise TokenizerArtifactError(
                    f"tokenizer JSON does not mark {role!r} token {token!r} as special"
                )

    def _metadata(self) -> dict[str, Any]:
        return {
            "format": _ARTIFACT_FORMAT,
            "version": _ARTIFACT_VERSION,
            "encode_special_tokens_as_text": True,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens.as_dict(),
            "special_token_ids": dict(self._special_token_ids),
            "fingerprint": self.fingerprint,
        }

    @staticmethod
    def _validate_metadata_header(metadata: dict[str, Any]) -> None:
        if metadata.get("format") != _ARTIFACT_FORMAT:
            raise TokenizerArtifactError(
                f"unsupported tokenizer artifact format: {metadata.get('format')!r}"
            )
        version = metadata.get("version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise TokenizerArtifactError("metadata field 'version' must be an integer")
        if version != _ARTIFACT_VERSION:
            raise TokenizerArtifactError(
                f"unsupported tokenizer artifact version {version}; expected {_ARTIFACT_VERSION}"
            )
        if metadata.get("encode_special_tokens_as_text") is not True:
            raise TokenizerArtifactError(
                "tokenizer metadata must enable encode_special_tokens_as_text"
            )

    @staticmethod
    def _special_tokens_from_metadata(metadata: dict[str, Any]) -> SpecialTokens:
        raw = _metadata_mapping(metadata, "special_tokens")
        try:
            return SpecialTokens(**raw)
        except (TypeError, ValueError) as error:
            raise TokenizerArtifactError(f"invalid special tokens in metadata: {error}") from error

    @staticmethod
    def _special_ids_from_metadata(metadata: dict[str, Any]) -> dict[str, int]:
        raw = _metadata_mapping(metadata, "special_token_ids")
        result: dict[str, int] = {}
        for role in _SPECIAL_ROLES:
            value = raw[role]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TokenizerArtifactError(
                    f"metadata special-token id {role!r} must be a non-negative integer"
                )
            result[role] = value
        if len(set(result.values())) != len(result):
            raise TokenizerArtifactError("metadata special-token ids must be unique")
        return result
