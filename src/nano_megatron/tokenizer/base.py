"""Small, framework-independent tokenizer contracts.

The training stack only needs integer token ids.  Keeping this protocol free of
PyTorch makes tokenization usable in preprocessing jobs and ordinary Python
programs without constructing a model or a parallel context.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from tokenizers.pre_tokenizers import ByteLevel

_CONTROL_TOKEN_PATTERN = re.compile(r"^<\|[A-Za-z0-9_.:-]+\|>$")


@runtime_checkable
class TextTokenizer(Protocol):
    """The narrow text/token-id boundary used by Nano-Megatron."""

    @property
    def vocab_size(self) -> int: ...

    @property
    def fingerprint(self) -> str: ...

    @property
    def unk_token_id(self) -> int: ...

    @property
    def bos_token_id(self) -> int: ...

    @property
    def eos_token_id(self) -> int: ...

    @property
    def pad_token_id(self) -> int: ...

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]: ...

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[list[int]]: ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str: ...

    def token_to_id(self, token: str) -> int | None: ...

    def id_to_token(self, token_id: int) -> str | None: ...


@dataclass(frozen=True, slots=True)
class SpecialTokens:
    """Text forms of the four special tokens understood by the data pipeline."""

    unk: str = "<|unk|>"
    bos: str = "<|bos|>"
    eos: str = "<|endoftext|>"
    pad: str = "<|pad|>"

    def __post_init__(self) -> None:
        values = self.as_dict()
        for role, token in values.items():
            if not isinstance(token, str):
                raise TypeError(f"special token {role!r} must be a string")
            if token == "":
                raise ValueError(f"special token {role!r} cannot be empty")
            if _CONTROL_TOKEN_PATTERN.fullmatch(token) is None:
                raise ValueError(
                    f"special token {role!r} must use the reserved control form "
                    "'<|name|>' with an ASCII name"
                )
        if len(set(values.values())) != len(values):
            raise ValueError("unk, bos, eos and pad special tokens must be unique")
        byte_alphabet = set(ByteLevel.alphabet())
        overlapping = [role for role, token in values.items() if token in byte_alphabet]
        if overlapping:
            raise ValueError(
                "special tokens must not overlap single-symbol byte-level alphabet entries: "
                + ", ".join(overlapping)
            )

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-friendly role-to-text mapping in stable id order."""

        return {
            "unk": self.unk,
            "bos": self.bos,
            "eos": self.eos,
            "pad": self.pad,
        }

    def ordered(self) -> tuple[str, str, str, str]:
        """Return the order passed to the BPE trainer.

        Hugging Face assigns trainer special tokens before ordinary vocabulary
        entries, so this stable order also makes newly trained artifacts easy to
        inspect: unk, bos, eos and pad normally receive ids 0, 1, 2 and 3.
        """

        return self.unk, self.bos, self.eos, self.pad


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class ByteBPETrainingConfig:
    """Configuration for deterministic byte-level BPE training."""

    vocab_size: int
    min_frequency: int = 2
    special_tokens: SpecialTokens = field(default_factory=SpecialTokens)
    show_progress: bool = False

    def __post_init__(self) -> None:
        _positive_integer("vocab_size", self.vocab_size)
        _positive_integer("min_frequency", self.min_frequency)
        if not isinstance(self.special_tokens, SpecialTokens):
            raise TypeError("special_tokens must be a SpecialTokens instance")
        if not isinstance(self.show_progress, bool):
            raise TypeError("show_progress must be a boolean")

        # A byte-level tokenizer must retain all 256 byte symbols even if they
        # are absent from the tiny training corpus.  Special tokens consume
        # additional entries unless their text deliberately overlaps a byte
        # alphabet symbol.
        required_tokens = set(ByteLevel.alphabet())
        required_tokens.update(self.special_tokens.ordered())
        minimum = len(required_tokens)
        if self.vocab_size < minimum:
            raise ValueError(
                "vocab_size is too small for the byte-level alphabet and special "
                f"tokens: got {self.vocab_size}, need at least {minimum}"
            )
