"""Verified TinyStories tokenizer-training workflow."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from nano_megatron.data.tinystories import (
    TINYSTORIES_SAMPLE_SEED,
    TINYSTORIES_SAMPLE_STORIES,
    TINYSTORIES_TRAIN_SHA256,
    TINYSTORIES_TRAIN_STORIES,
    validate_sample_artifact,
)

TINYSTORIES_TOKENIZER_VOCAB_SIZE = 8192
TINYSTORIES_TOKENIZER_MIN_FREQUENCY = 2

_PROVENANCE_FILE = "training.json"
_PROVENANCE_FORMAT = "nano-megatron-tokenizer-training"
_PROVENANCE_VERSION = 1


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read tokenizer training metadata {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"tokenizer training metadata must contain a JSON object: {path}")
    return value


def _expected_provenance(
    *,
    input_path: Path,
    input_sha256: str,
    documents: int,
    vocab_size: int,
    min_frequency: int,
    tokenizer_fingerprint: str,
) -> dict[str, Any]:
    return {
        "format": _PROVENANCE_FORMAT,
        "input": {
            "documents": documents,
            "path": str(input_path.resolve()),
            "sha256": input_sha256,
            "text_key": "text",
        },
        "tokenizer": {
            "fingerprint": tokenizer_fingerprint,
            "vocab_size": vocab_size,
        },
        "training": {
            "min_frequency": min_frequency,
            "target_vocab_size": vocab_size,
        },
        "version": _PROVENANCE_VERSION,
    }


def _reuse_existing_tokenizer(
    output: Path,
    *,
    input_path: Path,
    input_sha256: str,
    documents: int,
    vocab_size: int,
    min_frequency: int,
) -> dict[str, Any]:
    from nano_megatron.tokenizer import ByteLevelBPETokenizer

    if not output.is_dir():
        raise FileExistsError(f"tokenizer output exists and is not a directory: {output}")
    provenance_path = output / _PROVENANCE_FILE
    if not provenance_path.is_file():
        raise FileExistsError(
            "refusing to reuse or overwrite a tokenizer without matching training metadata: "
            f"{output}"
        )
    tokenizer = ByteLevelBPETokenizer.load(output)
    expected = _expected_provenance(
        input_path=input_path,
        input_sha256=input_sha256,
        documents=documents,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        tokenizer_fingerprint=tokenizer.fingerprint,
    )
    actual = _read_json_object(provenance_path)
    if actual != expected:
        raise FileExistsError(
            "existing tokenizer was trained with different corpus/configuration metadata: "
            f"{output}"
        )
    if tokenizer.vocab_size != vocab_size:
        raise ValueError(
            f"existing tokenizer vocabulary is not exactly {vocab_size}: {tokenizer.vocab_size}"
        )
    return {
        "command": "train-tinystories-tokenizer",
        "documents": documents,
        "fingerprint": tokenizer.fingerprint,
        "min_frequency": min_frequency,
        "output": str(output.resolve()),
        "reused": True,
        "vocab_size": tokenizer.vocab_size,
    }


def train_tinystories_tokenizer(
    input_path: str | Path = Path("data/raw/tinystories/train_500k.jsonl"),
    sample_metadata_path: str | Path = Path(
        "data/raw/tinystories/train_500k.metadata.json"
    ),
    output_path: str | Path = Path("data/tokenizers/tinystories-8k-500k"),
    *,
    expected_documents: int = TINYSTORIES_SAMPLE_STORIES,
    vocab_size: int = TINYSTORIES_TOKENIZER_VOCAB_SIZE,
    min_frequency: int = TINYSTORIES_TOKENIZER_MIN_FREQUENCY,
    sample_seed: int = TINYSTORIES_SAMPLE_SEED,
    expected_source_records: int = TINYSTORIES_TRAIN_STORIES,
    expected_source_sha256: str = TINYSTORIES_TRAIN_SHA256,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Train and atomically publish an exact-size BPE from the verified 500k sample."""

    source = Path(input_path)
    metadata_file = Path(sample_metadata_path)
    output = Path(output_path)
    if isinstance(expected_documents, bool) or not isinstance(expected_documents, int):
        raise TypeError("expected_documents must be an integer")
    if expected_documents < 1:
        raise ValueError("expected_documents must be positive")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 1:
        raise ValueError("vocab_size must be a positive integer")
    if (
        isinstance(min_frequency, bool)
        or not isinstance(min_frequency, int)
        or min_frequency < 1
    ):
        raise ValueError("min_frequency must be a positive integer")

    sample_metadata = validate_sample_artifact(
        source,
        metadata_file,
        sample_size=expected_documents,
        seed=sample_seed,
        expected_source_records=expected_source_records,
        expected_source_sha256=expected_source_sha256,
    )
    artifact_metadata = sample_metadata["artifact"]
    input_sha256 = str(artifact_metadata["sha256"])

    if output.exists():
        return _reuse_existing_tokenizer(
            output,
            input_path=source,
            input_sha256=input_sha256,
            documents=expected_documents,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    try:
        from nano_megatron.cli.tokenizer import run_train
        from nano_megatron.tokenizer import ByteLevelBPETokenizer

        result = run_train(
            (source,),
            temporary,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            text_key="text",
            show_progress=show_progress,
        )
        if result["documents"] != expected_documents:
            raise RuntimeError(
                "tokenizer trainer did not consume the required corpus size: "
                f"{result['documents']} != {expected_documents}"
            )
        tokenizer = ByteLevelBPETokenizer.load(temporary)
        if tokenizer.vocab_size != vocab_size:
            raise RuntimeError(
                "BPE trainer did not reach the required exact vocabulary size: "
                f"{tokenizer.vocab_size} != {vocab_size}; use the verified 500k corpus "
                "or lower min_frequency explicitly"
            )
        provenance = _expected_provenance(
            input_path=source,
            input_sha256=input_sha256,
            documents=expected_documents,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            tokenizer_fingerprint=tokenizer.fingerprint,
        )
        (temporary / _PROVENANCE_FILE).write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return {
        "command": "train-tinystories-tokenizer",
        "documents": expected_documents,
        "fingerprint": tokenizer.fingerprint,
        "min_frequency": min_frequency,
        "output": str(output.resolve()),
        "reused": False,
        "vocab_size": tokenizer.vocab_size,
    }


__all__ = [
    "TINYSTORIES_TOKENIZER_MIN_FREQUENCY",
    "TINYSTORIES_TOKENIZER_VOCAB_SIZE",
    "train_tinystories_tokenizer",
]
