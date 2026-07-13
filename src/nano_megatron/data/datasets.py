"""Minimal fixed-length token datasets for the first GPT phase."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


class FixedLengthTokenDataset(Dataset[dict[str, Tensor]]):
    """Turn one flat token tensor into non-overlapping next-token blocks."""

    def __init__(self, tokens: Tensor, sequence_length: int) -> None:
        if tokens.ndim != 1:
            raise ValueError("tokens must be a one-dimensional tensor")
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        self.tokens = tokens.to(dtype=torch.long, device="cpu").contiguous()
        self.sequence_length = sequence_length
        self.block_size = sequence_length + 1
        self.num_samples = self.tokens.numel() // self.block_size
        if self.num_samples < 1:
            raise ValueError("token tensor is too short for one training sample")

    @classmethod
    def from_file(cls, path: str | Path, sequence_length: int) -> FixedLengthTokenDataset:
        loaded = torch.load(Path(path), map_location="cpu", weights_only=True)
        if isinstance(loaded, dict):
            loaded = loaded.get("tokens")
        if not isinstance(loaded, Tensor):
            raise TypeError("token file must contain a Tensor or {'tokens': Tensor}")
        return cls(loaded, sequence_length)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        start = index * self.block_size
        block = self.tokens[start : start + self.block_size]
        return {"input_ids": block[:-1], "labels": block[1:]}


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

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        block = torch.randint(
            self.vocab_size,
            (self.sequence_length + 1,),
            generator=generator,
        )
        return {"input_ids": block[:-1], "labels": block[1:]}
