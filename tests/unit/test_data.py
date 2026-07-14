from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from nano_megatron.data import (  # noqa: E402
    FixedLengthTokenDataset,
    RandomTokenDataset,
    TokenCorpus,
    build_token_corpus,
    build_train_dataloader,
    build_train_dataset,
    iter_jsonl_text,
    preprocess_jsonl,
)
from nano_megatron.training import Trainer, TrainerState  # noqa: E402


class FakeTokenizer:
    vocab_size = 32
    eos_token_id = 31
    fingerprint = "fake-tokenizer-v1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool, bool]] = []

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        self.calls.append((text, add_bos, add_eos))
        ids = [((ord(character) - ord("a")) % 29) + 1 for character in text]
        if add_bos:
            ids.insert(0, 30)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids


def _write_jsonl(path: Path, texts: list[str], *, key: str = "text") -> None:
    path.write_text(
        "".join(json.dumps({key: text}, ensure_ascii=False) + "\n" for text in texts),
        encoding="utf-8",
    )


def _config(
    *,
    path: Path | None = None,
    text_path: Path | None = None,
    text_key: str = "text",
    append_eos: bool = True,
    sequence_length: int = 2,
    vocab_size: int = 32,
    max_steps: int = 2,
    micro_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    num_workers: int = 0,
    shuffle: bool = False,
    configured_data: int | None = 1,
    expert: int = 1,
) -> SimpleNamespace:
    tokenizer_config = SimpleNamespace(append_eos=append_eos) if text_path is not None else None
    return SimpleNamespace(
        data=SimpleNamespace(
            path=path,
            text_path=text_path,
            text_key=text_key,
            tokenizer=tokenizer_config,
            num_workers=num_workers,
            shuffle=shuffle,
        ),
        model=SimpleNamespace(seq_length=sequence_length, vocab_size=vocab_size),
        training=SimpleNamespace(
            max_steps=max_steps,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            seed=17,
        ),
        parallel=SimpleNamespace(data=configured_data, expert=expert),
    )


def _parallel(
    *,
    source_rank: int = 0,
    dp: int = 0,
    ep: int = 0,
    expert_size: int = 1,
    replica_count: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        batch_replica=SimpleNamespace(rank=source_rank),
        coordinate=SimpleNamespace(dp=dp, ep=ep),
        topology=SimpleNamespace(
            expert_parallel_size=expert_size,
            batch_replica_size=replica_count,
        ),
        runtime=SimpleNamespace(device_type="cpu"),
    )


def test_iter_jsonl_text_uses_requested_key_and_preserves_text(tmp_path: Path) -> None:
    path = tmp_path / "stories.jsonl"
    _write_jsonl(path, [" first\nparagraph ", "智能引号“ok”"], key="body")

    assert list(iter_jsonl_text(path, text_key="body")) == [
        " first\nparagraph ",
        "智能引号“ok”",
    ]


def test_iter_jsonl_text_rejects_an_empty_file_with_location(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.touch()

    with pytest.raises(ValueError, match=re.escape(f"{path}:1")):
        list(iter_jsonl_text(path))


@pytest.mark.parametrize(
    ("bad_line", "error_type", "message"),
    [
        ("\n", ValueError, "JSONL record must not be empty"),
        ("{\n", ValueError, "invalid JSON"),
        ("[]\n", TypeError, "JSONL record must be an object"),
        ('{"other":"value"}\n', ValueError, "missing text key"),
        ('{"text":3}\n', TypeError, "text must be a string"),
        ('{"text":"   "}\n', ValueError, "text must not be empty"),
    ],
)
def test_iter_jsonl_text_reports_path_and_line(
    tmp_path: Path,
    bad_line: str,
    error_type: type[Exception],
    message: str,
) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('{"text":"valid"}\n' + bad_line, encoding="utf-8")

    with pytest.raises(error_type, match=re.escape(f"{path}:2")) as error:
        list(iter_jsonl_text(path))
    assert message in str(error.value)


@pytest.mark.parametrize("text_key", ["", "   ", 7])
def test_iter_jsonl_text_rejects_invalid_text_key(tmp_path: Path, text_key: object) -> None:
    path = tmp_path / "unused.jsonl"
    expected = TypeError if isinstance(text_key, int) else ValueError
    with pytest.raises(expected):
        list(iter_jsonl_text(path, text_key=text_key))  # type: ignore[arg-type]


def test_token_corpus_appends_one_eos_and_tracks_offsets() -> None:
    tokenizer = FakeTokenizer()
    corpus = build_token_corpus(["ab", "c"], tokenizer, append_eos=True)

    torch.testing.assert_close(corpus.tokens, torch.tensor([1, 2, 31, 3, 31]))
    torch.testing.assert_close(corpus.document_offsets, torch.tensor([0, 3, 5]))
    assert corpus.documents == 2
    assert corpus.vocab_size == tokenizer.vocab_size
    assert corpus.eos_id == tokenizer.eos_token_id
    assert corpus.tokenizer_fingerprint == tokenizer.fingerprint
    assert corpus.append_eos is True
    assert tokenizer.calls == [("ab", False, False), ("c", False, False)]


def test_token_corpus_can_preserve_document_boundaries_without_eos() -> None:
    corpus = TokenCorpus.from_texts(["ab", "c"], FakeTokenizer(), append_eos=False)

    torch.testing.assert_close(corpus.tokens, torch.tensor([1, 2, 3]))
    torch.testing.assert_close(corpus.document_offsets, torch.tensor([0, 2, 3]))
    assert corpus.append_eos is False
    assert corpus.eos_id == 31


@pytest.mark.parametrize("texts", [[], [""], ["   "], ["a", 3]])
def test_token_corpus_rejects_invalid_document_iterables(texts: list[object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        TokenCorpus.from_texts(texts, FakeTokenizer())  # type: ignore[arg-type]


def test_token_corpus_round_trip_preserves_metadata(tmp_path: Path) -> None:
    path = tmp_path / "tokens.pt"
    expected = TokenCorpus.from_texts(["ab", "c"], FakeTokenizer())
    expected.save(path)

    actual = TokenCorpus.load(path)

    assert actual == expected
    assert actual.fingerprint == expected.fingerprint
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["artifact"] == "nano_megatron.token_corpus"
    assert payload["format_version"] == 2
    assert payload["fingerprint"] == expected.fingerprint


def test_token_corpus_loads_version_one_structured_artifact(tmp_path: Path) -> None:
    path = tmp_path / "tokens-v1.pt"
    expected = TokenCorpus.from_texts(["ab", "c"], FakeTokenizer())
    torch.save(
        {
            "artifact": "nano_megatron.token_corpus",
            "format_version": 1,
            "tokens": expected.tokens,
            "document_offsets": expected.document_offsets,
            "documents": expected.documents,
            "vocab_size": expected.vocab_size,
            "eos_id": expected.eos_id,
            "tokenizer_fingerprint": expected.tokenizer_fingerprint,
            "append_eos": expected.append_eos,
        },
        path,
    )

    actual = TokenCorpus.load(path)

    assert actual == expected


def test_token_corpus_save_refuses_to_overwrite_user_data(tmp_path: Path) -> None:
    path = tmp_path / "tokens.pt"
    path.write_bytes(b"keep me")
    corpus = TokenCorpus.from_texts(["ab"], FakeTokenizer())

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        corpus.save(path)

    assert path.read_bytes() == b"keep me"


@pytest.mark.parametrize("wrapped", [False, True])
def test_token_corpus_loads_legacy_tensor_formats(tmp_path: Path, wrapped: bool) -> None:
    path = tmp_path / "legacy.pt"
    tokens = torch.tensor([3, 1, 4, 1], dtype=torch.int32)
    torch.save({"tokens": tokens} if wrapped else tokens, path)

    corpus = TokenCorpus.load(path)

    torch.testing.assert_close(corpus.tokens, tokens.long())
    torch.testing.assert_close(corpus.document_offsets, torch.tensor([0, 4]))
    assert corpus.documents == 1
    assert corpus.vocab_size == 5
    assert corpus.eos_id is None
    assert corpus.tokenizer_fingerprint is None
    assert corpus.append_eos is False
    assert corpus.is_legacy


def test_token_corpus_rejects_corrupt_structured_metadata(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.pt"
    TokenCorpus.from_texts(["ab"], FakeTokenizer()).save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload.pop("document_offsets")
    torch.save(payload, path)

    with pytest.raises(ValueError, match="missing keys: document_offsets"):
        TokenCorpus.load(path)


def test_token_corpus_rejects_boolean_format_version(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.pt"
    TokenCorpus.from_texts(["ab"], FakeTokenizer()).save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["format_version"] = True
    torch.save(payload, path)

    with pytest.raises(ValueError, match="unsupported TokenCorpus format_version"):
        TokenCorpus.load(path)


def test_token_corpus_rejects_valid_but_fingerprint_changed_token(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.pt"
    TokenCorpus.from_texts(["ab"], FakeTokenizer()).save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["tokens"][0] = (payload["tokens"][0] + 1) % payload["vocab_size"]
    torch.save(payload, path)

    with pytest.raises(ValueError, match="content fingerprint"):
        TokenCorpus.load(path)


def test_token_corpus_validates_offsets_and_terminal_eos() -> None:
    common = {
        "tokens": torch.tensor([1, 31, 2, 3]),
        "documents": 2,
        "vocab_size": 32,
        "eos_id": 31,
        "tokenizer_fingerprint": "fingerprint",
        "append_eos": True,
    }
    with pytest.raises(ValueError, match="document_offsets must end"):
        TokenCorpus(document_offsets=torch.tensor([0, 2, 3]), **common)
    with pytest.raises(ValueError, match="every document must end"):
        TokenCorpus(document_offsets=torch.tensor([0, 2, 4]), **common)


def test_preprocess_jsonl_saves_loadable_corpus(tmp_path: Path) -> None:
    input_path = tmp_path / "stories.jsonl"
    output_path = tmp_path / "tokens.pt"
    _write_jsonl(input_path, ["ab", "c"])

    returned = preprocess_jsonl(input_path, output_path, FakeTokenizer())

    assert output_path.is_file()
    assert TokenCorpus.load(output_path) == returned


def test_fixed_length_dataset_uses_overlapping_boundary_token() -> None:
    dataset = FixedLengthTokenDataset(torch.arange(15), sequence_length=4)

    assert len(dataset) == 3
    first = dataset[0]
    second = dataset[1]
    torch.testing.assert_close(second["input_ids"], torch.tensor([4, 5, 6, 7]))
    torch.testing.assert_close(second["labels"], torch.tensor([5, 6, 7, 8]))
    assert first["labels"][-1] == second["input_ids"][0]


@pytest.mark.parametrize(
    ("tokens", "sequence_length", "error_type"),
    [
        ([1, 2, 3], 2, TypeError),
        (torch.tensor([1.0, 2.0, 3.0]), 2, TypeError),
        (torch.tensor([1, 2, 3]), True, ValueError),
    ],
)
def test_fixed_length_dataset_rejects_ambiguous_inputs(
    tokens: object,
    sequence_length: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        FixedLengthTokenDataset(tokens, sequence_length)  # type: ignore[arg-type]


@pytest.mark.parametrize("wrapped", [False, True])
def test_fixed_length_dataset_from_file_accepts_legacy_formats(
    tmp_path: Path, wrapped: bool
) -> None:
    path = tmp_path / "legacy.pt"
    tokens = torch.arange(9)
    torch.save({"tokens": tokens} if wrapped else tokens, path)

    dataset = FixedLengthTokenDataset.from_file(path, sequence_length=4)

    assert len(dataset) == 2
    torch.testing.assert_close(dataset[1]["input_ids"], torch.tensor([4, 5, 6, 7]))


def test_random_dataset_is_deterministic_and_checks_bounds() -> None:
    dataset = RandomTokenDataset(num_samples=2, sequence_length=8, vocab_size=32, seed=9)
    torch.testing.assert_close(dataset[1]["input_ids"], dataset[1]["input_ids"])
    assert not torch.equal(dataset[0]["input_ids"], dataset[1]["input_ids"])
    with pytest.raises(IndexError):
        dataset[2]


def test_build_train_dataset_constructs_text_deterministically(tmp_path: Path) -> None:
    path = tmp_path / "stories.jsonl"
    _write_jsonl(path, ["ab", "cd"])
    config = _config(text_path=path, sequence_length=2, max_steps=1)

    first = build_train_dataset(config, FakeTokenizer())
    second = build_train_dataset(config, FakeTokenizer())

    assert isinstance(first, FixedLengthTokenDataset)
    assert len(first) == len(second)
    for index in range(len(first)):
        torch.testing.assert_close(first[index]["input_ids"], second[index]["input_ids"])
        torch.testing.assert_close(first[index]["labels"], second[index]["labels"])


def test_direct_jsonl_and_preprocessed_dataset_are_sample_identical(tmp_path: Path) -> None:
    text_path = tmp_path / "stories.jsonl"
    token_path = tmp_path / "stories.tokens.pt"
    _write_jsonl(text_path, ["ab", "cd", "ef"])
    tokenizer = FakeTokenizer()
    preprocess_jsonl(text_path, token_path, tokenizer)

    direct = build_train_dataset(
        _config(text_path=text_path, sequence_length=2, max_steps=1),
        tokenizer,
    )
    offline = build_train_dataset(
        _config(path=token_path, sequence_length=2, max_steps=1),
        tokenizer,
    )

    assert len(direct) == len(offline)
    assert direct.fingerprint == offline.fingerprint
    for index in range(len(direct)):
        torch.testing.assert_close(direct[index]["input_ids"], offline[index]["input_ids"])
        torch.testing.assert_close(direct[index]["labels"], offline[index]["labels"])


def test_build_train_dataset_requires_matching_text_tokenizer(tmp_path: Path) -> None:
    path = tmp_path / "stories.jsonl"
    _write_jsonl(path, ["ab"])
    config = _config(text_path=path, vocab_size=33, max_steps=1)

    with pytest.raises(ValueError, match="does not match model.vocab_size"):
        build_train_dataset(config, FakeTokenizer())
    with pytest.raises(ValueError, match="explicitly supplied tokenizer"):
        build_train_dataset(config, None)


def test_build_train_dataset_rejects_two_input_paths(tmp_path: Path) -> None:
    config = _config(path=tmp_path / "tokens.pt", text_path=tmp_path / "text.jsonl")

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_train_dataset(config, FakeTokenizer())


def test_build_train_dataset_validates_structured_tokenizer_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tokens.pt"
    TokenCorpus.from_texts(["abcd"], FakeTokenizer()).save(path)
    config = _config(path=path, max_steps=1)

    mismatched = FakeTokenizer()
    mismatched.fingerprint = "other"  # type: ignore[misc]
    with pytest.raises(ValueError, match="fingerprint"):
        build_train_dataset(config, mismatched)

    config.data.tokenizer = SimpleNamespace(append_eos=False)
    with pytest.raises(ValueError, match="explicitly supplied tokenizer"):
        build_train_dataset(config, None)
    with pytest.raises(ValueError, match="append_eos"):
        build_train_dataset(config, FakeTokenizer())


def test_build_train_dataset_validates_optional_tokenizer_for_legacy_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.pt"
    torch.save(torch.tensor([1, 2, 3]), path)
    config = _config(path=path, vocab_size=33, max_steps=1)

    with pytest.raises(ValueError, match="tokenizer vocab_size"):
        build_train_dataset(config, FakeTokenizer())

    config = _config(path=path, vocab_size=32, max_steps=1)
    config.data.tokenizer = SimpleNamespace(append_eos=True)
    with pytest.raises(ValueError, match="legacy token files do not record append_eos"):
        build_train_dataset(config, FakeTokenizer())


def test_build_train_dataset_keeps_random_data_compatible() -> None:
    dataset = build_train_dataset(_config(path=None, text_path=None))

    assert isinstance(dataset, RandomTokenDataset)
    assert len(dataset) >= 1024


def test_loader_skips_all_data_construction_on_non_source_rank(tmp_path: Path) -> None:
    config = _config(path=tmp_path / "does-not-exist.pt")

    loader = build_train_dataloader(config, _parallel(source_rank=1))

    assert list(loader) == []


def test_loader_shards_samples_across_dp_ep_replicas(tmp_path: Path) -> None:
    path = tmp_path / "tokens.pt"
    torch.save(torch.arange(17) % 32, path)
    config = _config(path=path, sequence_length=2, max_steps=2)
    parallel_zero = _parallel(ep=0, expert_size=2, replica_count=2)
    parallel_one = _parallel(ep=1, expert_size=2, replica_count=2)

    loader_zero = build_train_dataloader(config, parallel_zero)
    loader_one = build_train_dataloader(config, parallel_one)
    indices_zero = set(loader_zero.sampler)
    indices_one = set(loader_one.sampler)

    assert indices_zero.isdisjoint(indices_one)
    assert indices_zero | indices_one == set(range(len(loader_zero.dataset)))
    assert len(loader_zero) >= config.training.max_steps
    assert len(loader_one) >= config.training.max_steps


def test_loader_preflight_accounts_for_both_drop_last_levels(tmp_path: Path) -> None:
    path = tmp_path / "tokens.pt"
    # Seven GPT samples become three samples per replica, then only one full local batch.
    torch.save(torch.arange(15) % 32, path)
    config = _config(
        path=path,
        sequence_length=2,
        max_steps=2,
        micro_batch_size=2,
    )

    with pytest.raises(ValueError, match="need at least 8 samples"):
        build_train_dataloader(config, _parallel(expert_size=2, replica_count=2))


def test_loader_capacity_uses_invocation_max_steps_override(tmp_path: Path) -> None:
    path = tmp_path / "tokens.pt"
    # Four samples are enough for one local batch/step, but not config.max_steps=3.
    torch.save(torch.arange(9) % 32, path)
    config = _config(
        path=path,
        sequence_length=2,
        max_steps=3,
        micro_batch_size=4,
    )

    loader = build_train_dataloader(config, _parallel(), max_steps=1)
    assert len(loader) == 1
    with pytest.raises(ValueError, match="3 batches per replica"):
        build_train_dataloader(config, _parallel())


def test_loader_rejects_invalid_invocation_max_steps() -> None:
    config = _config(max_steps=1)

    with pytest.raises(ValueError, match="max_steps must be positive"):
        build_train_dataloader(config, _parallel(), max_steps=0)


@pytest.mark.parametrize(
    ("replica_count", "start_step", "consumed_samples", "target_step", "required"),
    [
        (2, 10, 10, 20, 30),
        (1, 10, 20, 20, 30),
    ],
)
def test_loader_capacity_accounts_for_resume_and_replica_degree_change(
    tmp_path: Path,
    replica_count: int,
    start_step: int,
    consumed_samples: int,
    target_step: int,
    required: int,
) -> None:
    path = tmp_path / "tokens.pt"
    # sequence_length=1 gives one sample per token transition.
    torch.save(torch.arange(required + 1) % 32, path)
    config = _config(path=path, sequence_length=1, max_steps=target_step)

    loader = build_train_dataloader(
        config,
        _parallel(replica_count=replica_count),
        max_steps=target_step,
        start_step=start_step,
        consumed_samples=consumed_samples,
    )

    assert len(loader.dataset) == required


def test_loader_resume_rejects_consumed_samples_incompatible_with_current_batch() -> None:
    config = _config(max_steps=2)

    with pytest.raises(ValueError, match="consumed_samples is incompatible"):
        build_train_dataloader(
            config,
            _parallel(replica_count=2),
            start_step=1,
            consumed_samples=1,
        )


def test_loader_expands_lazy_random_dataset_for_runtime_replica_count() -> None:
    config = _config(max_steps=300, configured_data=None)

    loader = build_train_dataloader(config, _parallel(replica_count=4))

    assert isinstance(loader.dataset, RandomTokenDataset)
    assert len(loader.dataset) == 1200


def test_loader_preserves_num_workers_without_eager_worker_start() -> None:
    config = _config(max_steps=1, num_workers=2)

    loader = build_train_dataloader(config, _parallel())

    assert loader.num_workers == 2


def test_loader_fingerprint_binds_shuffle_and_seed() -> None:
    config = _config(max_steps=1, shuffle=True)
    first = build_train_dataloader(config, _parallel())
    repeated = build_train_dataloader(config, _parallel())
    config.training.seed += 1
    changed = build_train_dataloader(config, _parallel())

    assert first.data_fingerprint == repeated.data_fingerprint
    assert changed.data_fingerprint != first.data_fingerprint


def test_num_workers_zero_and_two_emit_identical_token_batches(tmp_path: Path) -> None:
    path = tmp_path / "tokens.pt"
    torch.save(torch.arange(65) % 32, path)
    single = build_train_dataloader(
        _config(path=path, sequence_length=2, max_steps=3, num_workers=0),
        _parallel(),
    )
    workers = build_train_dataloader(
        _config(path=path, sequence_length=2, max_steps=3, num_workers=2),
        _parallel(),
    )

    single_batches = list(single)
    worker_batches = list(workers)
    assert len(single_batches) == len(worker_batches)
    for expected, actual in zip(single_batches, worker_batches, strict=True):
        torch.testing.assert_close(actual["input_ids"], expected["input_ids"])
        torch.testing.assert_close(actual["labels"], expected["labels"])


@pytest.mark.parametrize(("old_replicas", "new_replicas"), [(1, 2), (2, 1), (2, 4)])
def test_distributed_sampler_degree_change_preserves_consumed_global_prefix(
    old_replicas: int,
    new_replicas: int,
) -> None:
    dataset = RandomTokenDataset(
        num_samples=48,
        sequence_length=2,
        vocab_size=32,
        seed=7,
    )

    def global_order(replicas: int) -> list[int]:
        shards = [
            list(
                torch.utils.data.DistributedSampler(
                    dataset,
                    num_replicas=replicas,
                    rank=rank,
                    shuffle=True,
                    seed=19,
                    drop_last=True,
                )
            )
            for rank in range(replicas)
        ]
        return [shards[rank][step] for step in range(len(shards[0])) for rank in range(replicas)]

    consumed_samples = 8
    old = global_order(old_replicas)
    new = global_order(new_replicas)
    assert old[:consumed_samples] == new[:consumed_samples]
    assert old[consumed_samples:] == new[consumed_samples:]


def test_token_loader_resume_replays_to_the_exact_next_shuffled_batch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tokens.pt"
    torch.save(torch.arange(65) % 32, path)
    config = _config(path=path, sequence_length=2, max_steps=3, shuffle=True)
    first_loader = build_train_dataloader(config, _parallel())
    first_iterator = iter(first_loader)
    next(first_iterator)
    expected = next(first_iterator)

    resumed_loader = build_train_dataloader(
        config,
        _parallel(),
        start_step=1,
        consumed_samples=1,
    )
    trainer = object.__new__(Trainer)
    trainer.state = TrainerState(step=1, consumed_samples=1)
    trainer.data_iterator = iter(resumed_loader)
    trainer._data_batches_consumed = 0
    trainer.batch_router = SimpleNamespace(is_source=True, data_replica_count=1)
    trainer.config = config
    trainer.restore_data_position()
    actual = next(trainer.data_iterator)

    torch.testing.assert_close(actual["input_ids"], expected["input_ids"])
    torch.testing.assert_close(actual["labels"], expected["labels"])
