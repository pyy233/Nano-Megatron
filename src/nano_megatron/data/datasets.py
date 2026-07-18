"""Deterministic fixed-length GPT datasets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from nano_megatron.tokenizer import TextTokenizer

from .corpus import TokenCorpus
from .mmap_corpus import MMapTokenCorpus

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _config_tokenizer_append_eos(data_config: Any) -> bool:
    tokenizer_config = getattr(data_config, "tokenizer", None)
    if tokenizer_config is None:
        raise ValueError("data.text_path requires data.tokenizer configuration")
    append_eos = tokenizer_config.append_eos
    if not isinstance(append_eos, bool):
        raise TypeError("data.tokenizer.append_eos must be a boolean")
    return append_eos


def _validate_vocab_size(actual: int, expected: int, *, source: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{source} vocab_size ({actual}) does not match model.vocab_size ({expected})"
        )


def _validate_structured_corpus_for_training(
    corpus: Any,
    *,
    model_vocab_size: int,
    tokenizer: TextTokenizer | None,
    expected_append_eos: bool | None,
    source: str,
) -> None:
    _validate_vocab_size(corpus.vocab_size, model_vocab_size, source=source)
    if tokenizer is not None:
        _validate_vocab_size(tokenizer.vocab_size, model_vocab_size, source="tokenizer")
        if tokenizer.eos_token_id != corpus.eos_id:
            raise ValueError("tokenizer.eos_token_id does not match token corpus eos_id")
        if tokenizer.fingerprint != corpus.tokenizer_fingerprint:
            raise ValueError("tokenizer fingerprint does not match token corpus metadata")
    if expected_append_eos is not None and corpus.append_eos != expected_append_eos:
        raise ValueError("token corpus append_eos does not match data.tokenizer.append_eos")


def _validate_corpus_for_training(
    corpus: TokenCorpus,
    *,
    model_vocab_size: int,
    tokenizer: TextTokenizer | None,
    expected_append_eos: bool | None,
) -> None:
    if corpus.is_legacy:
        if tokenizer is not None:
            _validate_vocab_size(tokenizer.vocab_size, model_vocab_size, source="tokenizer")
        if corpus.tokens.numel() and int(corpus.tokens.max().item()) >= model_vocab_size:
            raise ValueError("legacy token file contains ids outside model.vocab_size")
        if expected_append_eos is not None:
            raise ValueError(
                "legacy token files do not record append_eos; omit data.tokenizer or "
                "preprocess into a structured TokenCorpus"
            )
        return
    _validate_structured_corpus_for_training(
        corpus,
        model_vocab_size=model_vocab_size,
        tokenizer=tokenizer,
        expected_append_eos=expected_append_eos,
        source="token corpus",
    )


def _fixed_length_fingerprint(corpus_fingerprint: str, sequence_length: int) -> str:
    if not isinstance(corpus_fingerprint, str) or not corpus_fingerprint:
        raise ValueError("corpus_fingerprint must be a non-empty string")
    return hashlib.sha256(
        json.dumps(
            {
                "algorithm": "fixed-length-causal-v1",
                "corpus": corpus_fingerprint,
                "sequence_length": sequence_length,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class FixedLengthTokenDataset(Dataset[dict[str, Tensor]]):
    """Turn one flat stream into ``S+1`` windows separated by a stride of ``S``."""

    def __init__(
        self,
        tokens: Tensor,
        sequence_length: int,
        *,
        corpus_fingerprint: str | None = None,
    ) -> None:
        if not isinstance(tokens, Tensor):
            raise TypeError("tokens must be a torch.Tensor")
        if tokens.ndim != 1:
            raise ValueError("tokens must be a one-dimensional tensor")
        if tokens.dtype not in _INTEGER_DTYPES:
            raise TypeError("tokens must use an integer dtype")
        if (
            isinstance(sequence_length, bool)
            or not isinstance(sequence_length, int)
            or sequence_length < 1
        ):
            raise ValueError("sequence_length must be positive")
        self.tokens = tokens.to(dtype=torch.long, device="cpu").contiguous()
        self.sequence_length = sequence_length
        self.window_size = sequence_length + 1
        self.num_samples = max(0, (self.tokens.numel() - 1) // sequence_length)
        if self.num_samples < 1:
            raise ValueError("token tensor is too short for one training sample")
        if corpus_fingerprint is None:
            digest = hashlib.sha256(b"nano-megatron-flat-token-stream-v1\0")
            digest.update(self.tokens.numpy().astype("<i8", copy=False).tobytes(order="C"))
            corpus_fingerprint = digest.hexdigest()
        self.fingerprint = _fixed_length_fingerprint(corpus_fingerprint, sequence_length)

    @classmethod
    def from_file(cls, path: str | Path, sequence_length: int) -> FixedLengthTokenDataset:
        corpus = TokenCorpus.load(path)
        return cls(
            corpus.tokens,
            sequence_length,
            corpus_fingerprint=corpus.fingerprint,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        start = index * self.sequence_length
        block = self.tokens[start : start + self.window_size]
        return {"input_ids": block[:-1], "labels": block[1:]}


class MMapTokenDataset(Dataset[dict[str, Tensor]]):
    """Read fixed-length windows without materializing the full token corpus.

    The storage object is reopened once per process.  This avoids serializing or
    inheriting a live mmap handle when a DataLoader uses spawn or fork workers.
    Each returned window is an owned CPU ``torch.long`` allocation, so model code
    never receives a writable view over read-only corpus storage.
    """

    def __init__(self, artifact_dir: str | Path, sequence_length: int) -> None:
        if (
            isinstance(sequence_length, bool)
            or not isinstance(sequence_length, int)
            or sequence_length < 1
        ):
            raise ValueError("sequence_length must be positive")
        corpus = MMapTokenCorpus.load(artifact_dir)
        self.artifact_dir = Path(corpus.path)
        self.sequence_length = sequence_length
        self.window_size = sequence_length + 1
        self.token_count = int(corpus.token_count)
        self.document_count = int(corpus.documents)
        self.documents = self.document_count
        self.vocab_size = int(corpus.vocab_size)
        self.eos_id = int(corpus.eos_id)
        self.tokenizer_fingerprint = str(corpus.tokenizer_fingerprint)
        self.append_eos = bool(corpus.append_eos)
        self.corpus_fingerprint = str(corpus.fingerprint)
        self.num_samples = max(0, (self.token_count - 1) // sequence_length)
        if self.num_samples < 1:
            raise ValueError("mmap token corpus is too short for one training sample")
        self.fingerprint = _fixed_length_fingerprint(
            self.corpus_fingerprint,
            sequence_length,
        )
        corpus.close()
        self._corpus = corpus
        self._owner_pid: int | None = os.getpid()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        start = index * self.sequence_length
        block = self._corpus_for_process().read_tokens(start, start + self.window_size)
        if not isinstance(block, np.ndarray):
            raise TypeError("MMapTokenCorpus.read_tokens() must return a NumPy array")
        if block.ndim != 1 or block.size != self.window_size:
            raise RuntimeError(
                "mmap token read returned an invalid window: "
                f"got shape {tuple(block.shape)}, expected ({self.window_size},)"
            )
        owned = torch.from_numpy(np.array(block, dtype=np.int64, copy=True))
        return {"input_ids": owned[:-1], "labels": owned[1:]}

    def _corpus_for_process(self) -> MMapTokenCorpus:
        process_id = os.getpid()
        if self._owner_pid != process_id:
            # A fork may inherit a mapping opened by the parent.  Close that
            # process-local handle and let the validated storage object reopen
            # the same path lazily in this worker.  Spawn already clears the
            # mapping through MMapTokenCorpus.__getstate__.
            self._corpus.close()
            self._owner_pid = process_id
        return self._corpus

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_owner_pid"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


class RandomTokenDataset(Dataset[dict[str, Tensor]]):
    """Deterministic mock data for smoke tests and examples."""

    def __init__(
        self,
        *,
        num_samples: int,
        sequence_length: int,
        vocab_size: int,
        seed: int = 1234,
    ) -> None:
        if min(num_samples, sequence_length, vocab_size) < 1:
            raise ValueError("dataset sizes must be positive")
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.seed = seed
        self.fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "algorithm": "random-token-dataset-v1",
                    "num_samples": num_samples,
                    "seed": seed,
                    "sequence_length": sequence_length,
                    "vocab_size": vocab_size,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(self.seed + index)
        block = torch.randint(
            self.vocab_size,
            (self.sequence_length + 1,),
            generator=generator,
        )
        return {"input_ids": block[:-1], "labels": block[1:]}


def build_train_dataset(
    config: Any,
    tokenizer: TextTokenizer | None = None,
    *,
    data_config: Any | None = None,
) -> Dataset[dict[str, Tensor]]:
    """Build random, pre-tokenized, or small deterministic JSONL data."""

    sequence_length = int(config.model.seq_length)
    model_vocab_size = int(config.model.vocab_size)
    data = config.data if data_config is None else data_config
    text_path = getattr(data, "text_path", None)
    token_path = getattr(data, "path", None)
    mmap_path = getattr(data, "mmap_path", None)
    configured_sources = sum(path is not None for path in (text_path, token_path, mmap_path))
    if configured_sources > 1:
        raise ValueError("data.path, data.mmap_path and data.text_path are mutually exclusive")
    if text_path is not None:
        if tokenizer is None:
            raise ValueError("data.text_path requires an explicitly supplied tokenizer")
        _validate_vocab_size(tokenizer.vocab_size, model_vocab_size, source="tokenizer")
        corpus = TokenCorpus.from_jsonl(
            text_path,
            tokenizer,
            text_key=data.text_key,
            append_eos=_config_tokenizer_append_eos(data),
        )
        return FixedLengthTokenDataset(
            corpus.tokens,
            sequence_length,
            corpus_fingerprint=corpus.fingerprint,
        )
    if mmap_path is not None:
        dataset = MMapTokenDataset(mmap_path, sequence_length)
        tokenizer_config = getattr(data, "tokenizer", None)
        if tokenizer_config is not None and tokenizer is None:
            raise ValueError(
                "data.tokenizer configuration requires an explicitly supplied tokenizer"
            )
        _validate_structured_corpus_for_training(
            dataset,
            model_vocab_size=model_vocab_size,
            tokenizer=tokenizer,
            expected_append_eos=(
                None if tokenizer_config is None else _config_tokenizer_append_eos(data)
            ),
            source="mmap token corpus",
        )
        return dataset
    if token_path is not None:
        corpus = TokenCorpus.load(token_path)
        tokenizer_config = getattr(data, "tokenizer", None)
        if tokenizer_config is not None and tokenizer is None:
            raise ValueError(
                "data.tokenizer configuration requires an explicitly supplied tokenizer"
            )
        _validate_corpus_for_training(
            corpus,
            model_vocab_size=model_vocab_size,
            tokenizer=tokenizer,
            expected_append_eos=(
                None if tokenizer_config is None else _config_tokenizer_append_eos(data)
            ),
        )
        return FixedLengthTokenDataset(
            corpus.tokens,
            sequence_length,
            corpus_fingerprint=corpus.fingerprint,
        )

    local_batch_size = int(config.training.micro_batch_size) * int(
        config.training.gradient_accumulation_steps
    )
    configured_data_parallel = getattr(config.parallel, "data", None) or 1
    configured_replicas = int(configured_data_parallel) * int(config.parallel.expert)
    return RandomTokenDataset(
        num_samples=max(
            int(config.training.max_steps) * local_batch_size * configured_replicas,
            1024,
        ),
        sequence_length=sequence_length,
        vocab_size=model_vocab_size,
        seed=int(config.training.seed),
    )
