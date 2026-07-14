from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nano_megatron.tokenizer import (
    ByteBPETrainingConfig,
    ByteLevelBPETokenizer,
    SpecialTokens,
    TextTokenizer,
    TokenizerArtifactError,
    validate_tokenizer_for_model,
)

_CORPUS = (
    "Hello, byte-level BPE!  Spaces stay exact.\n",
    "你好，世界。中文不需要预先分词。",
    "Emoji also round-trip: 🐍🚀✨",
    "newlines\n\nand\ttabs are bytes too",
) * 8


@pytest.fixture(scope="module")
def tokenizer() -> ByteLevelBPETokenizer:
    return ByteLevelBPETokenizer.train(
        _CORPUS,
        ByteBPETrainingConfig(vocab_size=320, min_frequency=1),
    )


@pytest.mark.parametrize(
    "text",
    [
        "plain ASCII",
        "  leading and trailing spaces  ",
        "第一行中文\n第二行中文",
        "emoji: 🐍🚀✨",
        "tabs\tand\nblank lines\n\nend",
        "",
    ],
)
def test_byte_level_round_trip_preserves_text(
    tokenizer: ByteLevelBPETokenizer,
    text: str,
) -> None:
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_tokens_have_explicit_unique_ids_and_manual_bos_eos(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    special_ids = {
        tokenizer.unk_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    }
    assert len(special_ids) == 4
    assert tokenizer.token_to_id(tokenizer.special_tokens.unk) == tokenizer.unk_token_id
    assert tokenizer.token_to_id(tokenizer.special_tokens.bos) == tokenizer.bos_token_id
    assert tokenizer.token_to_id(tokenizer.special_tokens.eos) == tokenizer.eos_token_id
    assert tokenizer.token_to_id(tokenizer.special_tokens.pad) == tokenizer.pad_token_id

    text = "hello 世界"
    plain = tokenizer.encode(text)
    wrapped = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert wrapped == [tokenizer.bos_token_id, *plain, tokenizer.eos_token_id]
    assert tokenizer.decode(wrapped) == text
    assert tokenizer.id_to_token(tokenizer.eos_token_id) == tokenizer.special_tokens.eos


def test_special_token_looking_text_is_not_implicitly_control_syntax(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    text = f"literal marker: {tokenizer.special_tokens.eos}"

    encoded = tokenizer.encode(text)

    assert tokenizer.eos_token_id not in encoded
    assert tokenizer.decode(encoded) == text
    assert tokenizer.encode(text, add_eos=True) == [*encoded, tokenizer.eos_token_id]


def test_encode_batch_matches_individual_encoding(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    texts = ["one", "第二", "three 🐍"]
    expected = [tokenizer.encode(text, add_bos=True, add_eos=True) for text in texts]
    assert tokenizer.encode_batch(texts, add_bos=True, add_eos=True) == expected
    assert tokenizer.encode_batch([]) == []


def test_implements_public_protocol(tokenizer: ByteLevelBPETokenizer) -> None:
    assert isinstance(tokenizer, TextTokenizer)


def test_save_load_round_trip_and_metadata(
    tmp_path: Path,
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    artifact = tmp_path / "tokenizer"
    tokenizer.save(artifact)

    assert (artifact / "tokenizer.json").is_file()
    metadata = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["format"] == "nano-megatron-byte-level-bpe"
    assert metadata["version"] == 1
    assert metadata["encode_special_tokens_as_text"] is True
    assert metadata["vocab_size"] == tokenizer.vocab_size
    assert metadata["fingerprint"] == tokenizer.fingerprint
    assert metadata["special_token_ids"]["eos"] == tokenizer.eos_token_id

    loaded = ByteLevelBPETokenizer.load(artifact)
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.fingerprint == tokenizer.fingerprint
    assert loaded.special_tokens == tokenizer.special_tokens
    assert loaded.encode_batch(_CORPUS[:3], add_eos=True) == tokenizer.encode_batch(
        _CORPUS[:3], add_eos=True
    )
    assert loaded.decode(loaded.encode("存档 parity 🚀")) == "存档 parity 🚀"


def test_training_is_deterministic() -> None:
    config = ByteBPETrainingConfig(vocab_size=300, min_frequency=1)
    first = ByteLevelBPETokenizer.train(_CORPUS, config)
    second = ByteLevelBPETokenizer.train(iter(_CORPUS), config)

    assert first.vocab_size == second.vocab_size
    assert first.fingerprint == second.fingerprint
    assert first.encode_batch(_CORPUS[:3]) == second.encode_batch(_CORPUS[:3])


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ByteBPETrainingConfig(vocab_size=0),
        lambda: ByteBPETrainingConfig(vocab_size=259),
        lambda: ByteBPETrainingConfig(vocab_size=300, min_frequency=0),
        lambda: ByteBPETrainingConfig(vocab_size=True),
        lambda: ByteBPETrainingConfig(vocab_size=300, show_progress=1),
        lambda: ByteBPETrainingConfig(vocab_size=300, special_tokens="wrong"),
    ],
)
def test_training_config_rejects_invalid_values(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SpecialTokens(unk=""),
        lambda: SpecialTokens(unk="<|same|>", bos="<|same|>"),
        lambda: SpecialTokens(eos="e"),
        lambda: SpecialTokens(eos="ab"),
        lambda: SpecialTokens(eos="Ġab"),
        lambda: SpecialTokens(pad=3),
    ],
)
def test_special_tokens_must_be_non_empty_unique_strings(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


@pytest.mark.parametrize(
    "texts",
    [
        [],
        "one string is not a corpus",
        ["valid", 7],
    ],
)
def test_training_rejects_invalid_corpora(texts) -> None:
    with pytest.raises((TypeError, ValueError)):
        ByteLevelBPETokenizer.train(
            texts,
            ByteBPETrainingConfig(vocab_size=300, min_frequency=1),
        )


def test_default_control_string_stays_literal_when_present_in_training_text() -> None:
    marker = SpecialTokens().eos
    text = f"a literal {marker} remains ordinary text"

    tokenizer = ByteLevelBPETokenizer.train(
        [text] * 16,
        ByteBPETrainingConfig(vocab_size=320, min_frequency=1),
    )

    ids = tokenizer.encode(text)
    assert tokenizer.eos_token_id not in ids
    assert tokenizer.decode(ids) == text


def test_encode_decode_validate_public_inputs(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    with pytest.raises(TypeError, match="text must be"):
        tokenizer.encode(7)
    with pytest.raises(TypeError, match="sequence of strings"):
        tokenizer.encode_batch("not-a-batch")
    with pytest.raises(TypeError, match="integer"):
        tokenizer.decode([True])
    with pytest.raises(ValueError, match="outside"):
        tokenizer.decode([tokenizer.vocab_size])
    assert tokenizer.id_to_token(-1) is None


def test_save_refuses_to_overwrite_non_empty_directory(
    tmp_path: Path,
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty"):
        tokenizer.save(artifact)
    assert (artifact / "keep.txt").read_text(encoding="utf-8") == "user data"


def test_save_replaces_an_existing_empty_directory_atomically(
    tmp_path: Path,
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()

    tokenizer.save(artifact)

    assert ByteLevelBPETokenizer.load(artifact).fingerprint == tokenizer.fingerprint
    assert not list(tmp_path.glob(".artifact.tmp-*"))


def test_atomic_save_cleans_temporary_artifact_after_write_failure(
    tmp_path: Path,
    tokenizer: ByteLevelBPETokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"

    original_write_text = Path.write_text

    def fail_metadata(self: Path, *args, **kwargs):
        if self.name == "metadata.json":
            raise RuntimeError("injected write failure")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_metadata)
    with pytest.raises(RuntimeError, match="injected write failure"):
        tokenizer.save(artifact)

    assert not artifact.exists()
    assert not list(tmp_path.glob(".artifact.tmp-*"))


def _saved_artifact(tmp_path: Path, tokenizer: ByteLevelBPETokenizer) -> Path:
    artifact = tmp_path / "artifact"
    tokenizer.save(artifact)
    return artifact


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format", "another-format", "format"),
        ("version", 999, "version"),
        ("vocab_size", 999, "vocabulary size"),
        ("fingerprint", "0" * 64, "fingerprint"),
    ],
)
def test_load_rejects_bad_metadata_header_or_identity(
    tmp_path: Path,
    tokenizer: ByteLevelBPETokenizer,
    field: str,
    value: object,
    message: str,
) -> None:
    artifact = _saved_artifact(tmp_path, tokenizer)
    metadata_path = artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(TokenizerArtifactError, match=message):
        ByteLevelBPETokenizer.load(artifact)


def test_load_rejects_special_id_mismatch(
    tmp_path: Path,
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    artifact = _saved_artifact(tmp_path, tokenizer)
    metadata_path = artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["special_token_ids"]["eos"] = tokenizer.pad_token_id
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(TokenizerArtifactError, match="unique|special-token ids"):
        ByteLevelBPETokenizer.load(artifact)


def test_load_detects_tokenizer_json_fingerprint_change(
    tmp_path: Path,
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    artifact = _saved_artifact(tmp_path, tokenizer)
    tokenizer_path = artifact / "tokenizer.json"
    payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    payload["pre_tokenizer"]["add_prefix_space"] = True
    tokenizer_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TokenizerArtifactError, match="fingerprint"):
        ByteLevelBPETokenizer.load(artifact)


def test_load_rejects_missing_or_malformed_artifact(
    tmp_path: Path,
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ByteLevelBPETokenizer.load(tmp_path / "missing")

    artifact = _saved_artifact(tmp_path, tokenizer)
    (artifact / "metadata.json").write_text("not JSON", encoding="utf-8")
    with pytest.raises(TokenizerArtifactError, match="valid tokenizer metadata"):
        ByteLevelBPETokenizer.load(artifact)


def test_validate_tokenizer_for_model_checks_vocab_and_special_ids(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    validate_tokenizer_for_model(
        tokenizer,
        SimpleNamespace(vocab_size=tokenizer.vocab_size),
    )

    with pytest.raises(ValueError) as captured:
        validate_tokenizer_for_model(
            tokenizer,
            SimpleNamespace(vocab_size=tokenizer.vocab_size + 1),
        )
    message = str(captured.value)
    assert f"tokenizer.vocab_size={tokenizer.vocab_size}" in message
    assert f"model_config.vocab_size={tokenizer.vocab_size + 1}" in message


@pytest.mark.parametrize(
    "special_ids",
    [
        (0, 0, 2, 3),
        (0, 1, 2, 300),
    ],
)
def test_validate_tokenizer_for_model_rejects_bad_special_ids(
    special_ids: tuple[int, int, int, int],
) -> None:
    fake = SimpleNamespace(
        vocab_size=300,
        unk_token_id=special_ids[0],
        bos_token_id=special_ids[1],
        eos_token_id=special_ids[2],
        pad_token_id=special_ids[3],
    )
    with pytest.raises(ValueError, match="unique|outside"):
        validate_tokenizer_for_model(fake, SimpleNamespace(vocab_size=300))
