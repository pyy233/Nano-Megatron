"""Build a loader for the independent `(DP, EP)` batch-replica coordinates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any

from torch.utils.data import DataLoader, DistributedSampler

from nano_megatron.tokenizer import TextTokenizer

from .datasets import RandomTokenDataset, build_train_dataset


def _local_batch_size(config: Any) -> int:
    return int(config.training.micro_batch_size) * int(config.training.gradient_accumulation_steps)


def _target_steps(config: Any, max_steps: int | None) -> int:
    target = config.training.max_steps if max_steps is None else max_steps
    if isinstance(target, bool) or not isinstance(target, int):
        raise TypeError("max_steps must be an integer")
    if target < 1:
        raise ValueError("max_steps must be positive")
    return target


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _required_batches_per_replica(
    config: Any,
    replica_count: int,
    *,
    target_steps: int,
    start_step: int,
    consumed_samples: int,
) -> int:
    start = _nonnegative_integer("start_step", start_step)
    consumed = _nonnegative_integer("consumed_samples", consumed_samples)
    current_global_batch = replica_count * _local_batch_size(config)
    if consumed % current_global_batch:
        raise ValueError(
            "consumed_samples is incompatible with the current DP/EP replica count and "
            f"local batch size: {consumed} % {current_global_batch} != 0"
        )
    skipped_batches = consumed // current_global_batch
    future_steps = max(0, target_steps - start)
    return skipped_batches + future_steps


def _required_samples(
    config: Any,
    replica_count: int,
    required_batches: int,
) -> int:
    return required_batches * _local_batch_size(config) * replica_count


def _ensure_training_capacity(
    dataset: Any,
    config: Any,
    replica_count: int,
    required_batches: int,
) -> None:
    local_batch_size = _local_batch_size(config)
    samples_per_replica = len(dataset) // replica_count
    batches_per_replica = samples_per_replica // local_batch_size
    if batches_per_replica < required_batches:
        required = _required_samples(config, replica_count, required_batches)
        raise ValueError(
            "training dataset is too short after DistributedSampler/DataLoader drop_last: "
            f"got {len(dataset)} samples for {replica_count} DP×EP replicas and "
            f"local_batch_size={local_batch_size}, which provides {batches_per_replica} "
            f"batches per replica; need at least {required} samples for "
            f"{required_batches} batches per replica"
        )


def build_train_dataloader(
    config: Any,
    parallel: Any,
    tokenizer: TextTokenizer | None = None,
    *,
    max_steps: int | None = None,
    start_step: int = 0,
    consumed_samples: int = 0,
) -> DataLoader | Iterator[None]:
    """Return data only on the rank-zero member of each batch-replica group."""

    target_steps = _target_steps(config, max_steps)
    replica_count = int(parallel.topology.batch_replica_size)
    required_batches = _required_batches_per_replica(
        config,
        replica_count,
        target_steps=target_steps,
        start_step=start_step,
        consumed_samples=consumed_samples,
    )
    if parallel.batch_replica.rank != 0:
        return iter(())

    dataset = build_train_dataset(config, tokenizer)
    required_samples = _required_samples(config, replica_count, required_batches)
    if isinstance(dataset, RandomTokenDataset) and len(dataset) < required_samples:
        dataset = RandomTokenDataset(
            num_samples=required_samples,
            sequence_length=config.model.seq_length,
            vocab_size=config.model.vocab_size,
            seed=config.training.seed,
        )
    _ensure_training_capacity(dataset, config, replica_count, required_batches)

    replica_index = (
        parallel.coordinate.dp * parallel.topology.expert_parallel_size + parallel.coordinate.ep
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=parallel.topology.batch_replica_size,
        rank=replica_index,
        shuffle=config.data.shuffle,
        seed=config.training.seed,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=_local_batch_size(config),
        sampler=sampler,
        num_workers=config.data.num_workers,
        drop_last=True,
        pin_memory=parallel.runtime.device_type == "cuda",
    )
    dataset_fingerprint = getattr(dataset, "fingerprint", None)
    if not isinstance(dataset_fingerprint, str) or not dataset_fingerprint:
        raise TypeError("training dataset must expose a non-empty fingerprint")
    loader.data_fingerprint = hashlib.sha256(  # type: ignore[attr-defined]
        json.dumps(
            {
                "dataset": dataset_fingerprint,
                "sampler": "torch-distributed-sampler-v1",
                "seed": int(config.training.seed),
                "shuffle": bool(config.data.shuffle),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return loader
