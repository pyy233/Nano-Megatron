"""Route one microbatch to all TP/PP/CP ranks sharing a `(DP, EP)` replica."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist

from nano_megatron.parallel.group import require_parallel_group

_CP_SEQUENCE_FIELDS = frozenset({"input_ids", "labels", "position_ids", "loss_mask"})


def _batch_group(parallel: Any) -> Any:
    group_getter = getattr(parallel, "group", None)
    if callable(group_getter):
        from nano_megatron.parallel import GroupKey

        try:
            group = group_getter(GroupKey.BATCH_REPLICA)
        except (KeyError, AttributeError) as error:
            raise TypeError("parallel must expose an explicit batch-replica group") from error
    else:
        group = getattr(parallel, "batch_replica", None)
    return require_parallel_group(group, name="batch-replica group")


class BatchRouter:
    """A correctness-first object-broadcast batch router.

    Dataset batches are small relative to model activations. A later optimized router may broadcast
    tensors directly while preserving this interface.
    """

    def __init__(self, parallel: Any) -> None:
        self.parallel = parallel
        self.group = _batch_group(parallel)

    @property
    def is_source(self) -> bool:
        return self.group is None or self.group.rank == 0

    @property
    def data_replica_index(self) -> int:
        coordinate = self.parallel.coordinate
        ep_size = self.parallel.topology.expert_parallel_size
        return coordinate.dp * ep_size + coordinate.ep

    @property
    def data_replica_count(self) -> int:
        return self.parallel.topology.batch_replica_size

    def route(self, batch: Any | None) -> Any:
        if self.group is None or self.group.size == 1:
            if batch is None:
                raise ValueError("the batch source must provide a batch")
            routed = batch
        else:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("batch routing with group size > 1 requires torch.distributed")
            payload = [batch if self.is_source else None]
            dist.broadcast_object_list(
                payload,
                src=self.group.ranks[0],
                group=self.group.process_group,
                device=getattr(getattr(self.parallel, "runtime", None), "device", None),
            )
            routed = payload[0]
        return shard_batch_for_context_parallel(routed, self.parallel)


def shard_batch_for_context_parallel(batch: Any, parallel: Any) -> Any:
    """Slice standard token fields over the explicitly supplied CP axis.

    The batch is broadcast across ``BATCH_REPLICA`` first because every TP/PP rank needs the
    metadata.  CP ranks then keep only a contiguous sequence chunk.  Keeping this operation in
    the router makes it impossible for the model to accidentally all-gather duplicate full
    sequences while still allowing non-sequence metadata to pass through unchanged.
    """

    cp_group = getattr(parallel, "cp", None)
    cp_group = require_parallel_group(cp_group, name="context-parallel group")
    cp_size = int(cp_group.size)
    cp_rank = int(cp_group.rank)
    if cp_size == 1:
        return batch
    if not 0 <= cp_rank < cp_size:
        raise ValueError(f"CP rank must be in [0, {cp_size}), got {cp_rank}")
    if not isinstance(batch, Mapping):
        raise TypeError("context-parallel batch routing requires a mapping batch")

    input_ids = batch.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim < 1:
        raise ValueError(
            "context-parallel batch routing requires tensor batch['input_ids']"
        )
    global_sequence_length = int(input_ids.shape[-1])
    if global_sequence_length % cp_size:
        raise ValueError(
            "input sequence length must be divisible by context-parallel size: "
            f"{global_sequence_length} % {cp_size} != 0"
        )
    local_sequence_length = global_sequence_length // cp_size
    start = cp_rank * local_sequence_length
    end = start + local_sequence_length

    routed: dict[str, Any] = dict(batch)
    for name in _CP_SEQUENCE_FIELDS.intersection(batch):
        value = batch[name]
        if value is None:
            continue
        if not isinstance(value, torch.Tensor) or value.ndim < 1:
            raise TypeError(f"context-parallel batch field {name!r} must be a tensor")
        if value.shape[-1] != global_sequence_length:
            raise ValueError(
                f"context-parallel batch field {name!r} has sequence length "
                f"{value.shape[-1]}, expected {global_sequence_length}"
            )
        routed[name] = value[..., start:end].contiguous()
    return type(batch)(routed) if type(batch) is not dict else routed
