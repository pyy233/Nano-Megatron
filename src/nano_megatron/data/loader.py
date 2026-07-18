"""Deterministic, resumable data loading over independent `(DP, EP)` shards."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from itertools import islice
from typing import Any

from torch.utils.data import DataLoader, DistributedSampler

from nano_megatron.tokenizer import TextTokenizer

from .datasets import RandomTokenDataset, build_train_dataset


def _local_batch_size(config: Any) -> int:
    return int(config.training.micro_batch_size) * int(
        config.training.gradient_accumulation_steps
    )


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


class _OffsetDistributedSampler(DistributedSampler):
    """Start a deterministic distributed epoch at a committed local offset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.start_index = 0

    def set_start_index(self, start_index: int) -> None:
        value = _nonnegative_integer("sampler start_index", start_index)
        if value > self.num_samples:
            raise ValueError(
                f"sampler start_index exceeds samples per rank: {value} > {self.num_samples}"
            )
        self.start_index = value

    def __iter__(self) -> Iterator[int]:
        return islice(super().__iter__(), self.start_index, None)

    def __len__(self) -> int:
        return max(0, self.num_samples - self.start_index)


class StatefulDataLoader(Iterator[Any]):
    """Cycle epochs forever while checkpointing only committed batch progress.

    DataLoader workers may prefetch sampler indices, so sampler iteration itself
    is not durable state. This wrapper advances ``sample_offset`` only when a
    complete batch is returned to the trainer. The offset is global across all
    `(DP, EP)` shards, making the saved position topology-explicit and allowing
    compatible DP reshards to reconstruct the same global prefix.
    """

    _STATE_VERSION = 1

    def __init__(
        self,
        loader: DataLoader,
        sampler: _OffsetDistributedSampler,
        *,
        local_batch_size: int,
        replica_count: int,
        shuffle_seed: int,
        data_fingerprint: str,
    ) -> None:
        self.loader = loader
        self.sampler = sampler
        self.local_batch_size = int(local_batch_size)
        self.replica_count = int(replica_count)
        self.shuffle_seed = _nonnegative_integer("shuffle_seed", shuffle_seed)
        self.data_fingerprint = data_fingerprint
        self.epoch = 0
        self.sample_offset = 0
        self._iterator: Iterator[Any] | None = None

        local_epoch_samples = (sampler.num_samples // self.local_batch_size) * self.local_batch_size
        self.samples_per_epoch = local_epoch_samples * self.replica_count
        self.global_batch_size = self.local_batch_size * self.replica_count
        if self.samples_per_epoch < self.global_batch_size:
            raise ValueError("training dataset does not provide one complete global batch")

    @property
    def dataset(self) -> Any:
        return self.loader.dataset

    @property
    def num_workers(self) -> int:
        return self.loader.num_workers

    @property
    def epoch_seed(self) -> int:
        return self.shuffle_seed + self.epoch

    @property
    def batches_per_epoch(self) -> int:
        return self.samples_per_epoch // self.global_batch_size

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> StatefulDataLoader:
        return self

    def __next__(self) -> Any:
        while True:
            if self._iterator is None:
                self._start_epoch_iterator()
            assert self._iterator is not None
            try:
                batch = next(self._iterator)
            except StopIteration:
                if self.sample_offset != 0:
                    raise RuntimeError(
                        "stateful DataLoader ended before its committed epoch boundary: "
                        f"offset={self.sample_offset}, epoch_samples={self.samples_per_epoch}"
                    ) from None
                self._iterator = None
                continue

            self.sample_offset += self.global_batch_size
            if self.sample_offset > self.samples_per_epoch:
                raise RuntimeError("stateful DataLoader advanced beyond the epoch boundary")
            if self.sample_offset == self.samples_per_epoch:
                self.epoch += 1
                self.sample_offset = 0
                self._iterator = None
            return batch

    def state_dict(self) -> dict[str, int]:
        return {
            "version": self._STATE_VERSION,
            "epoch": self.epoch,
            "shuffle_seed": self.shuffle_seed,
            "epoch_seed": self.epoch_seed,
            "sample_offset": self.sample_offset,
            "samples_per_epoch": self.samples_per_epoch,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        version = int(state.get("version", self._STATE_VERSION))
        if version != self._STATE_VERSION:
            raise ValueError(f"unsupported stateful DataLoader state version: {version}")
        epoch = _nonnegative_integer("data epoch", int(state.get("epoch", 0)))
        shuffle_seed = _nonnegative_integer(
            "data shuffle_seed", int(state.get("shuffle_seed", self.shuffle_seed))
        )
        if shuffle_seed != self.shuffle_seed:
            raise RuntimeError(
                "training shuffle seed changed across resume: "
                f"{shuffle_seed} != {self.shuffle_seed}"
            )
        epoch_seed = int(state.get("epoch_seed", shuffle_seed + epoch))
        if epoch_seed != shuffle_seed + epoch:
            raise ValueError(
                "data epoch_seed is inconsistent with shuffle_seed + epoch: "
                f"{epoch_seed} != {shuffle_seed + epoch}"
            )
        saved_epoch_samples = int(state.get("samples_per_epoch", self.samples_per_epoch))
        if saved_epoch_samples != self.samples_per_epoch:
            raise RuntimeError(
                "samples per epoch changed across resume; exact data restoration is impossible: "
                f"{saved_epoch_samples} != {self.samples_per_epoch}"
            )
        sample_offset = _nonnegative_integer(
            "data sample_offset", int(state.get("sample_offset", 0))
        )
        self._validate_offset(sample_offset)
        self.epoch = epoch
        self.sample_offset = sample_offset
        self._iterator = None

    def seek_consumed_samples(self, consumed_samples: int) -> None:
        """Upgrade an old checkpoint that only stored total global samples."""

        consumed = _nonnegative_integer("consumed_samples", consumed_samples)
        if consumed % self.global_batch_size:
            raise RuntimeError(
                "restored consumed_samples is incompatible with the current global batch size: "
                f"{consumed} % {self.global_batch_size} != 0"
            )
        self.epoch, self.sample_offset = divmod(consumed, self.samples_per_epoch)
        self._validate_offset(self.sample_offset)
        self._iterator = None

    def _validate_offset(self, sample_offset: int) -> None:
        if sample_offset >= self.samples_per_epoch and sample_offset != 0:
            raise ValueError(
                "data sample_offset must be inside the current epoch: "
                f"{sample_offset} >= {self.samples_per_epoch}"
            )
        if sample_offset % self.global_batch_size:
            raise RuntimeError(
                "data sample_offset is incompatible with the current global batch size: "
                f"{sample_offset} % {self.global_batch_size} != 0"
            )

    def _start_epoch_iterator(self) -> None:
        if self.sample_offset % self.replica_count:
            raise RuntimeError(
                "global data sample_offset cannot be divided across current replicas: "
                f"{self.sample_offset} % {self.replica_count} != 0"
            )
        local_offset = self.sample_offset // self.replica_count
        self.sampler.set_epoch(self.epoch)
        self.sampler.set_start_index(local_offset)
        self._iterator = iter(self.loader)


def _ensure_epoch_capacity(dataset: Any, replica_count: int, local_batch_size: int) -> None:
    samples_per_replica = len(dataset) // replica_count
    if samples_per_replica < local_batch_size:
        required = replica_count * local_batch_size
        raise ValueError(
            "training dataset is too short for one complete epoch batch after sharding: "
            f"got {len(dataset)} samples for {replica_count} DP×EP replicas and "
            f"local_batch_size={local_batch_size}; need at least {required} samples"
        )


def build_train_dataloader(
    config: Any,
    parallel: Any,
    tokenizer: TextTokenizer | None = None,
    *,
    max_steps: int | None = None,
    start_step: int = 0,
    consumed_samples: int = 0,
) -> StatefulDataLoader | Iterator[None]:
    """Build an infinite, exact-resume loader only on each batch-replica source."""

    target_steps = _target_steps(config, max_steps)
    start = _nonnegative_integer("start_step", start_step)
    consumed = _nonnegative_integer("consumed_samples", consumed_samples)
    if parallel.batch_replica.rank != 0:
        return iter(())

    replica_count = int(parallel.topology.batch_replica_size)
    local_batch_size = _local_batch_size(config)
    dataset = build_train_dataset(config, tokenizer)
    random_required_samples = target_steps * local_batch_size * replica_count
    if isinstance(dataset, RandomTokenDataset) and len(dataset) < random_required_samples:
        dataset = RandomTokenDataset(
            num_samples=random_required_samples,
            sequence_length=config.model.seq_length,
            vocab_size=config.model.vocab_size,
            seed=config.training.seed,
        )
    _ensure_epoch_capacity(dataset, replica_count, local_batch_size)

    replica_index = (
        parallel.coordinate.dp * parallel.topology.expert_parallel_size + parallel.coordinate.ep
    )
    sampler = _OffsetDistributedSampler(
        dataset,
        num_replicas=replica_count,
        rank=replica_index,
        shuffle=config.data.shuffle,
        seed=config.training.seed,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        num_workers=config.data.num_workers,
        drop_last=True,
        pin_memory=parallel.runtime.device_type == "cuda",
    )
    dataset_fingerprint = getattr(dataset, "fingerprint", None)
    if not isinstance(dataset_fingerprint, str) or not dataset_fingerprint:
        raise TypeError("training dataset must expose a non-empty fingerprint")
    data_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "dataset": dataset_fingerprint,
                "sampler": "stateful-torch-distributed-sampler-v2",
                "seed": int(config.training.seed),
                "shuffle": bool(config.data.shuffle),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    stateful = StatefulDataLoader(
        loader,
        sampler,
        local_batch_size=local_batch_size,
        replica_count=replica_count,
        shuffle_seed=int(config.training.seed),
        data_fingerprint=data_fingerprint,
    )
    if consumed:
        stateful.seek_consumed_samples(consumed)
    elif start:
        stateful.seek_consumed_samples(start * stateful.global_batch_size)
    return stateful


def build_validation_dataloader(
    config: Any,
    parallel: Any,
    tokenizer: TextTokenizer | None = None,
) -> DataLoader | Iterator[None]:
    """Build a deterministic finite validation shard on each batch source."""

    validation = config.validation
    if int(validation.interval) <= 0:
        return iter(())
    if validation.data is None:
        raise ValueError("validation.interval > 0 requires validation.data")
    if parallel.batch_replica.rank != 0:
        return iter(())

    replica_count = int(parallel.topology.batch_replica_size)
    local_batch_size = _local_batch_size(config)
    dataset = build_train_dataset(
        config,
        tokenizer,
        data_config=validation.data,
    )
    _ensure_epoch_capacity(dataset, replica_count, local_batch_size)
    replica_index = (
        parallel.coordinate.dp * parallel.topology.expert_parallel_size + parallel.coordinate.ep
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=replica_count,
        rank=replica_index,
        shuffle=False,
        seed=int(config.training.seed),
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        num_workers=validation.data.num_workers,
        drop_last=True,
        pin_memory=parallel.runtime.device_type == "cuda",
    )
    if len(loader) < int(validation.batches):
        required = int(validation.batches) * local_batch_size * replica_count
        raise ValueError(
            "validation dataset is too short for validation.batches complete global batches: "
            f"got {len(dataset)} samples, need at least {required}"
        )
    return loader


__all__ = [
    "StatefulDataLoader",
    "build_train_dataloader",
    "build_validation_dataloader",
]
