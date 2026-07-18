from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DistributedSampler

from nano_megatron.config import (
    DataConfig,
    DistributedConfig,
    GPTConfig,
    ParallelConfig,
    TrainConfig,
    TrainingConfig,
)
from nano_megatron.data import (
    StatefulDataLoader,
    TokenCorpus,
    build_train_dataloader,
    preprocess_jsonl_mmap,
)
from nano_megatron.distributed import DistributedRuntime
from nano_megatron.parallel import ParallelContext

_WORLD_SIZE = 2
_SEQUENCE_LENGTH = 4
_VOCAB_SIZE = 32
_TOKEN_COUNT = 33
_EXPECTED_SAMPLES = 8


class _MMapTokenizer:
    vocab_size = _VOCAB_SIZE
    eos_token_id = 2
    fingerprint = "distributed-cpu-test-tokenizer"

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        assert text
        assert not add_bos
        assert not add_eos
        return torch.arange(_TOKEN_COUNT).remainder(_VOCAB_SIZE).tolist()


def _write_token_files(directory: Path) -> tuple[Path, Path, Path]:
    tokens = torch.arange(_TOKEN_COUNT, dtype=torch.long).remainder(_VOCAB_SIZE)
    structured = directory / "structured.pt"
    TokenCorpus(
        tokens=tokens,
        document_offsets=torch.tensor([0, _TOKEN_COUNT]),
        documents=1,
        vocab_size=_VOCAB_SIZE,
        eos_id=2,
        tokenizer_fingerprint="distributed-cpu-test-tokenizer",
        append_eos=False,
    ).save(structured)

    legacy = directory / "legacy.pt"
    torch.save(tokens, legacy)

    source = directory / "mmap.jsonl"
    source.write_text('{"text":"synthetic token stream"}\n', encoding="utf-8")
    mmap = directory / "mmap"
    preprocess_jsonl_mmap(
        source,
        mmap,
        _MMapTokenizer(),
        append_eos=False,
    ).close()
    return structured, legacy, mmap


def _initialize_runtime(rank: int, rendezvous: str) -> DistributedRuntime:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(_WORLD_SIZE), LOCAL_RANK=str(rank))
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    return DistributedRuntime(
        DistributedConfig(
            backend="gloo",
            device="cpu",
            init_method=f"file://{rendezvous}",
        )
    ).initialize()


def _parallel_config(axis: str) -> ParallelConfig:
    if axis == "data":
        return ParallelConfig(data=_WORLD_SIZE)
    if axis == "expert":
        return ParallelConfig(expert=_WORLD_SIZE, data=1)
    if axis == "tensor":
        return ParallelConfig(tensor=_WORLD_SIZE, data=1)
    raise ValueError(f"unsupported test axis {axis!r}")


def _train_config(path: str, parallel: ParallelConfig, *, mmap: bool = False) -> TrainConfig:
    return TrainConfig(
        parallel=parallel,
        model=GPTConfig(seq_length=_SEQUENCE_LENGTH, vocab_size=_VOCAB_SIZE),
        training=TrainingConfig(
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=1,
            seed=17,
        ),
        data=DataConfig(
            path=None if mmap else path,
            mmap_path=path if mmap else None,
            shuffle=False,
        ),
    )


def _sampler_worker(
    rank: int,
    rendezvous: str,
    axis: str,
    structured_path: str,
    legacy_path: str,
    mmap_path: str,
) -> None:
    runtime = _initialize_runtime(rank, rendezvous)
    parallel_config = _parallel_config(axis)
    parallel = ParallelContext.create(runtime, parallel_config)
    try:
        assert parallel.batch_replica.rank == 0
        for token_path, is_mmap in (
            (structured_path, False),
            (legacy_path, False),
            (mmap_path, True),
        ):
            loader = build_train_dataloader(
                _train_config(token_path, parallel_config, mmap=is_mmap),
                parallel,
            )
            assert isinstance(loader, StatefulDataLoader)
            assert isinstance(loader.sampler, DistributedSampler)
            assert len(loader.dataset) == _EXPECTED_SAMPLES

            local_indices = list(loader.sampler)
            gathered: list[list[int] | None] = [None] * _WORLD_SIZE
            dist.all_gather_object(gathered, local_indices)
            assert all(indices is not None for indices in gathered)
            rank_indices = [indices for indices in gathered if indices is not None]

            first, second = map(set, rank_indices)
            assert first.isdisjoint(second)
            assert first | second == set(range(_EXPECTED_SAMPLES))
            assert sum(len(indices) for indices in rank_indices) == _EXPECTED_SAMPLES
    finally:
        parallel.close()
        runtime.close()


def _tensor_source_worker(rank: int, rendezvous: str, mmap_path: str) -> None:
    runtime = _initialize_runtime(rank, rendezvous)
    parallel_config = _parallel_config("tensor")
    parallel = ParallelContext.create(runtime, parallel_config)
    try:
        loader = build_train_dataloader(
            _train_config(mmap_path, parallel_config, mmap=True),
            parallel,
        )
        is_loader = isinstance(loader, StatefulDataLoader)
        is_empty = False if is_loader else list(loader) == []
        local = {
            "batch_replica_rank": parallel.batch_replica.rank,
            "is_empty": is_empty,
            "is_loader": is_loader,
        }
        gathered: list[dict[str, int | bool] | None] = [None] * _WORLD_SIZE
        dist.all_gather_object(gathered, local)

        assert gathered == [
            {"batch_replica_rank": 0, "is_empty": False, "is_loader": True},
            {"batch_replica_rank": 1, "is_empty": True, "is_loader": False},
        ]
    finally:
        parallel.close()
        runtime.close()


@pytest.mark.distributed
@pytest.mark.parametrize("axis", ["data", "expert"])
def test_structured_and_legacy_sampler_partitions_batch_replicas(
    tmp_path: Path,
    axis: str,
) -> None:
    structured, legacy, mmap = _write_token_files(tmp_path)
    mp.spawn(
        _sampler_worker,
        args=(
            str(tmp_path / "rendezvous"),
            axis,
            str(structured),
            str(legacy),
            str(mmap),
        ),
        nprocs=_WORLD_SIZE,
        join=True,
    )


@pytest.mark.distributed
def test_tensor_parallel_batch_replica_only_builds_loader_on_source_rank(
    tmp_path: Path,
) -> None:
    _, _, mmap = _write_token_files(tmp_path)
    mp.spawn(
        _tensor_source_worker,
        args=(str(tmp_path / "rendezvous"), str(mmap)),
        nprocs=_WORLD_SIZE,
        join=True,
    )
