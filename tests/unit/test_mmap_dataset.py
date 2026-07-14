from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from nano_megatron.data import (
    FixedLengthTokenDataset,
    MMapTokenCorpus,
    MMapTokenDataset,
    TokenCorpus,
    build_train_dataloader,
    build_train_dataset,
    preprocess_jsonl_mmap,
)


class _Tokenizer:
    vocab_size = 64
    eos_token_id = 63
    fingerprint = "mmap-dataset-test-tokenizer"

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids = [(ord(character) % 60) + 1 for character in text]
        if add_bos:
            ids.insert(0, 62)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids


def _write_jsonl(path: Path) -> None:
    documents = ["abcdefghij", "klmnopqrst", "uvwxyz0123"]
    path.write_text(
        "".join(json.dumps({"text": text}) + "\n" for text in documents),
        encoding="utf-8",
    )


def _build_artifacts(tmp_path: Path) -> tuple[Path, Path, TokenCorpus, MMapTokenCorpus]:
    source = tmp_path / "documents.jsonl"
    tensor_path = tmp_path / "tokens.pt"
    mmap_path = tmp_path / "tokens.mmap"
    _write_jsonl(source)
    tokenizer = _Tokenizer()
    tensor_corpus = TokenCorpus.from_jsonl(source, tokenizer)
    tensor_corpus.save(tensor_path)
    mmap_corpus = preprocess_jsonl_mmap(source, mmap_path, tokenizer)
    return tensor_path, mmap_path, tensor_corpus, mmap_corpus


def _config(
    *,
    path: Path | None = None,
    mmap_path: Path | None = None,
    text_path: Path | None = None,
    vocab_size: int = 64,
    append_eos: bool = True,
    tokenizer_configured: bool = True,
    num_workers: int = 0,
) -> SimpleNamespace:
    tokenizer_config = SimpleNamespace(append_eos=append_eos) if tokenizer_configured else None
    return SimpleNamespace(
        data=SimpleNamespace(
            path=path,
            mmap_path=mmap_path,
            text_path=text_path,
            text_key="text",
            tokenizer=tokenizer_config,
            num_workers=num_workers,
            shuffle=False,
        ),
        model=SimpleNamespace(seq_length=4, vocab_size=vocab_size),
        training=SimpleNamespace(
            max_steps=1,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            seed=19,
        ),
        parallel=SimpleNamespace(data=1, expert=1),
    )


def _parallel(*, source_rank: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        batch_replica=SimpleNamespace(rank=source_rank),
        coordinate=SimpleNamespace(dp=0, ep=0),
        topology=SimpleNamespace(batch_replica_size=1, expert_parallel_size=1),
        runtime=SimpleNamespace(device_type="cpu"),
    )


def test_mmap_dataset_matches_tensor_windows_and_fingerprint(tmp_path: Path) -> None:
    _, mmap_path, tensor_corpus, mmap_corpus = _build_artifacts(tmp_path)
    tensor_dataset = FixedLengthTokenDataset(
        tensor_corpus.tokens,
        sequence_length=4,
        corpus_fingerprint=tensor_corpus.fingerprint,
    )
    mmap_dataset = MMapTokenDataset(mmap_path, sequence_length=4)

    assert tensor_corpus.fingerprint == mmap_corpus.fingerprint
    assert mmap_dataset.fingerprint == tensor_dataset.fingerprint
    assert mmap_dataset.token_count == tensor_corpus.tokens.numel()
    assert mmap_dataset.document_count == tensor_corpus.documents
    assert len(mmap_dataset) == len(tensor_dataset)
    for index in range(len(mmap_dataset)):
        expected = tensor_dataset[index]
        actual = mmap_dataset[index]
        assert actual["input_ids"].dtype is torch.long
        assert actual["input_ids"].device.type == "cpu"
        torch.testing.assert_close(actual["input_ids"], expected["input_ids"])
        torch.testing.assert_close(actual["labels"], expected["labels"])


def test_mmap_dataset_returns_owned_copies_and_checks_bounds(tmp_path: Path) -> None:
    _, mmap_path, _, _ = _build_artifacts(tmp_path)
    dataset = MMapTokenDataset(mmap_path, sequence_length=4)
    original = dataset[0]["input_ids"].clone()
    mutated = dataset[0]["input_ids"]
    mutated[0] = -1

    torch.testing.assert_close(dataset[0]["input_ids"], original)
    with pytest.raises(IndexError):
        dataset[-1]
    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_mmap_dataset_pickle_reopens_storage(tmp_path: Path) -> None:
    _, mmap_path, _, _ = _build_artifacts(tmp_path)
    dataset = MMapTokenDataset(mmap_path, sequence_length=4)
    expected = dataset[1]

    restored = pickle.loads(pickle.dumps(dataset))

    assert restored._corpus._tokens is None
    actual = restored[1]
    assert restored._corpus._tokens is not None
    torch.testing.assert_close(actual["input_ids"], expected["input_ids"])
    torch.testing.assert_close(actual["labels"], expected["labels"])


def test_mmap_dataset_works_with_spawn_dataloader_workers(tmp_path: Path) -> None:
    _, mmap_path, _, _ = _build_artifacts(tmp_path)
    dataset = MMapTokenDataset(mmap_path, sequence_length=4)
    loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        multiprocessing_context="spawn",
        shuffle=False,
    )

    batches = list(loader)

    assert sum(batch["input_ids"].shape[0] for batch in batches) == len(dataset)
    assert all(batch["input_ids"].dtype is torch.long for batch in batches)


def test_dataset_factory_selects_and_validates_mmap_source(tmp_path: Path) -> None:
    _, mmap_path, _, _ = _build_artifacts(tmp_path)
    tokenizer = _Tokenizer()

    dataset = build_train_dataset(_config(mmap_path=mmap_path), tokenizer)

    assert isinstance(dataset, MMapTokenDataset)
    with pytest.raises(ValueError, match="mmap token corpus vocab_size"):
        build_train_dataset(_config(mmap_path=mmap_path, vocab_size=65), tokenizer)
    with pytest.raises(ValueError, match="explicitly supplied tokenizer"):
        build_train_dataset(_config(mmap_path=mmap_path), None)
    with pytest.raises(ValueError, match="append_eos"):
        build_train_dataset(_config(mmap_path=mmap_path, append_eos=False), tokenizer)

    class WrongVocabularyTokenizer(_Tokenizer):
        vocab_size = 65

    with pytest.raises(ValueError, match="tokenizer vocab_size"):
        build_train_dataset(_config(mmap_path=mmap_path), WrongVocabularyTokenizer())


def test_dataset_factory_rejects_multiple_source_paths(tmp_path: Path) -> None:
    tensor_path, mmap_path, _, _ = _build_artifacts(tmp_path)

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_train_dataset(
            _config(path=tensor_path, mmap_path=mmap_path),
            _Tokenizer(),
        )


def test_tensor_and_mmap_loaders_have_identical_data_fingerprints(tmp_path: Path) -> None:
    tensor_path, mmap_path, _, _ = _build_artifacts(tmp_path)
    tokenizer = _Tokenizer()
    tensor_loader = build_train_dataloader(
        _config(path=tensor_path),
        _parallel(),
        tokenizer,
    )
    mmap_loader = build_train_dataloader(
        _config(mmap_path=mmap_path),
        _parallel(),
        tokenizer,
    )

    assert isinstance(tensor_loader, DataLoader)
    assert isinstance(mmap_loader, DataLoader)
    assert isinstance(mmap_loader.dataset, MMapTokenDataset)
    assert tensor_loader.data_fingerprint == mmap_loader.data_fingerprint


def test_non_source_rank_does_not_open_mmap_artifact(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"

    loader = build_train_dataloader(
        _config(mmap_path=missing),
        _parallel(source_rank=1),
        _Tokenizer(),
    )

    assert list(loader) == []
