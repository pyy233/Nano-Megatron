"""Cross-component checks kept outside both the tokenizer and GPT model."""

from __future__ import annotations

from typing import Any

from .base import TextTokenizer


def _vocab_size(owner: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TypeError(f"{owner}.vocab_size must be a positive integer, got {value!r}")
    return value


def validate_tokenizer_for_model(tokenizer: TextTokenizer, model_config: Any) -> None:
    """Ensure tokenizer ids fit the model's unpadded vocabulary exactly."""

    try:
        tokenizer_vocab_size = _vocab_size("tokenizer", tokenizer.vocab_size)
    except AttributeError as error:
        raise TypeError("tokenizer must expose vocab_size") from error
    try:
        model_vocab_size = _vocab_size("model_config", model_config.vocab_size)
    except AttributeError as error:
        raise TypeError("model_config must expose vocab_size") from error

    if tokenizer_vocab_size != model_vocab_size:
        raise ValueError(
            "tokenizer and model vocabulary sizes must match exactly: "
            f"tokenizer.vocab_size={tokenizer_vocab_size}, "
            f"model_config.vocab_size={model_vocab_size}"
        )

    names = ("unk_token_id", "bos_token_id", "eos_token_id", "pad_token_id")
    special_ids: dict[str, int] = {}
    for name in names:
        try:
            token_id = getattr(tokenizer, name)
        except AttributeError as error:
            raise TypeError(f"tokenizer must expose {name}") from error
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(f"tokenizer.{name} must be an integer, got {token_id!r}")
        if not 0 <= token_id < tokenizer_vocab_size:
            raise ValueError(
                f"tokenizer.{name}={token_id} is outside vocabulary range "
                f"[0, {tokenizer_vocab_size})"
            )
        special_ids[name] = token_id

    if len(set(special_ids.values())) != len(special_ids):
        raise ValueError(f"tokenizer special-token ids must be unique: {special_ids}")
