"""Build a loader for the independent `(DP, EP)` batch-replica coordinates."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from torch.utils.data import DataLoader, DistributedSampler

from .datasets import FixedLengthTokenDataset, RandomTokenDataset


def build_train_dataloader(config: Any, parallel: Any) -> DataLoader | Iterator[None]:
    """Return data only on the rank-zero member of each batch-replica group."""

    if parallel.batch_replica.rank != 0:
        return iter(())

    if config.data.path is None:
        dataset = RandomTokenDataset(
            num_samples=max(
                config.training.max_steps
                * config.training.micro_batch_size
                * config.training.gradient_accumulation_steps
                * 2,
                1024,
            ),
            sequence_length=config.model.seq_length,
            vocab_size=config.model.vocab_size,
            seed=config.training.seed,
        )
    else:
        dataset = FixedLengthTokenDataset.from_file(
            config.data.path, config.model.seq_length
        )

    replica_index = (
        parallel.coordinate.dp * parallel.topology.expert_parallel_size
        + parallel.coordinate.ep
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=parallel.topology.batch_replica_size,
        rank=replica_index,
        shuffle=config.data.shuffle,
        seed=config.training.seed,
        drop_last=True,
    )
    return DataLoader(
        dataset,
        batch_size=(
            config.training.micro_batch_size
            * config.training.gradient_accumulation_steps
        ),
        sampler=sampler,
        num_workers=config.data.num_workers,
        drop_last=True,
        pin_memory=parallel.runtime.device_type == "cuda",
    )
