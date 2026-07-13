"""Runtime value object for one process group visible to the current rank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .group_plan import GroupKeyLike


@dataclass(frozen=True, slots=True)
class ParallelGroup:
    key: GroupKeyLike
    ranks: tuple[int, ...]
    process_group: Any | None
    rank: int
    size: int
    backend: str
    channel: str = "default"

    def __post_init__(self) -> None:
        if not self.ranks:
            raise ValueError("parallel group ranks cannot be empty")
        if len(set(self.ranks)) != len(self.ranks):
            raise ValueError("parallel group ranks cannot contain duplicates")
        if tuple(sorted(self.ranks)) != self.ranks:
            raise ValueError("parallel group ranks must be sorted")
        if self.size != len(self.ranks):
            raise ValueError(f"group size {self.size} does not match {len(self.ranks)} ranks")
        if not 0 <= self.rank < self.size:
            raise ValueError(f"local group rank must be in [0, {self.size}), got {self.rank}")
        if self.size > 1 and self.process_group is None:
            raise ValueError("a multi-rank ParallelGroup requires a materialized process_group")

    @property
    def global_rank(self) -> int:
        return self.ranks[self.rank]

    @property
    def is_distributed(self) -> bool:
        return self.size > 1

    def contains(self, global_rank: int) -> bool:
        return global_rank in self.ranks

    def global_rank_at(self, group_rank: int) -> int:
        try:
            return self.ranks[group_rank]
        except IndexError as error:
            raise ValueError(f"group rank must be in [0, {self.size}), got {group_rank}") from error


def require_parallel_group(group: Any, *, name: str) -> Any:
    """Validate an explicitly injected ParallelGroup-like value.

    Size-one fake groups may use ``process_group=None`` for local tests. A
    multi-rank group must carry its materialized communicator so no collective
    can accidentally interpret ``None`` as the default WORLD group.
    """

    if group is None:
        raise TypeError(f"{name} must be supplied explicitly")
    rank = getattr(group, "rank", None)
    size = getattr(group, "size", None)
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError(f"{name} must expose an integer rank")
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError(f"{name} must expose an integer size")
    if size < 1 or not 0 <= rank < size:
        raise ValueError(f"invalid {name} rank/size: rank={rank}, size={size}")
    if not hasattr(group, "process_group"):
        raise TypeError(f"{name} must expose process_group")
    if size > 1 and group.process_group is None:
        raise ValueError(f"a multi-rank {name} requires a materialized process_group")
    return group
