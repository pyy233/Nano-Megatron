"""Internal helpers shared by tensor-parallel modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import torch.distributed as dist

if TYPE_CHECKING:
    from nano_megatron.parallel import ParallelContext, ParallelGroup


@runtime_checkable
class ParallelGroupLike(Protocol):
    process_group: dist.ProcessGroup | None
    rank: int
    size: int


def tensor_parallel_group(
    parallel: ParallelContext | ParallelGroup | ParallelGroupLike,
) -> ParallelGroupLike:
    """Resolve the explicitly supplied context/group to its TP group.

    This is intentionally not a global fallback: callers must pass either a
    ``ParallelContext`` or the exact ``ParallelGroup`` they want to use.
    """

    group = getattr(parallel, "tp", parallel)
    if not isinstance(group, ParallelGroupLike):
        raise TypeError("parallel must be a ParallelContext or ParallelGroup-like value")
    if group.size <= 0 or not 0 <= group.rank < group.size:
        raise ValueError(f"invalid parallel group rank/size: rank={group.rank}, size={group.size}")
    return cast("ParallelGroupLike", group)


def require_distributed(group: ParallelGroupLike) -> dist.ProcessGroup | None:
    if group.size == 1:
        return None
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "a multi-rank ParallelGroup requires torch.distributed to be initialized"
        )
    if group.process_group is None:
        raise RuntimeError("a multi-rank ParallelGroup must contain a materialized process_group")
    return group.process_group
