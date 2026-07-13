"""Small process-group helpers without any global parallel-state dependency."""

from __future__ import annotations

from typing import Any

import torch.distributed as dist

from nano_megatron.parallel.group import require_parallel_group


def unwrap_process_group(group: Any) -> dist.ProcessGroup | None:
    """Return the communicator from an explicitly supplied CP group."""

    return require_parallel_group(group, name="context-parallel group").process_group


def group_size(group: Any) -> int:
    return int(require_parallel_group(group, name="context-parallel group").size)


def group_rank(group: Any) -> int:
    return int(require_parallel_group(group, name="context-parallel group").rank)


def group_global_rank(group: Any, rank: int) -> int:
    """Translate a group-local rank to the global peer expected by P2P APIs."""

    group = require_parallel_group(group, name="context-parallel group")
    ranks = getattr(group, "ranks", None)
    if isinstance(ranks, tuple):
        return int(ranks[rank])
    process_group = unwrap_process_group(group)
    if process_group is None:
        return rank
    get_global_rank = getattr(dist, "get_global_rank", None)
    if callable(get_global_rank):
        return int(get_global_rank(process_group, rank))
    # Older PyTorch has no public translation helper. This fallback is correct
    # for the default group; explicit ParallelGroup callers always carry ranks.
    return rank
