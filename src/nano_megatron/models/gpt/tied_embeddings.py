"""Pipeline-safe synchronization for tied GPT input/output embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import Tensor, nn

if TYPE_CHECKING:
    from nano_megatron.parallel import ParallelGroup


class TiedEmbeddingSynchronizer:
    """Synchronize one endpoint's embedding shard over an explicit PP group.

    The embedding group contains only the first and last PP stages for one
    fixed TP/CP/EP/DP coordinate. Weight broadcast must happen before pipeline
    P2P starts. Gradient reduction must happen after the complete pipeline
    backward; doing it from a parameter hook can deadlock with activation-
    gradient P2P.
    """

    def __init__(self, weight: nn.Parameter, group: ParallelGroup) -> None:
        if weight.ndim != 2:
            raise ValueError(
                f"a tied embedding shard must be a matrix, got shape {tuple(weight.shape)}"
            )
        if group.size not in (1, 2):
            raise ValueError(
                "the embedding group must contain the first and last pipeline stages; "
                f"got {group.size} ranks"
            )
        self.weight = weight
        self.group = group
        self.weight._nano_megatron_tied_source_rank = self.source_rank  # type: ignore[attr-defined]
        self._global_shape = tuple(weight.shape)
        self._weight_is_synchronized = group.size == 1

    def bind_weight(self, weight: Tensor) -> None:
        """Rebind after wrappers such as FSDP2 replace Parameters with DTensors."""

        if tuple(weight.shape) != self._global_shape:
            raise ValueError(
                "replacement tied embedding has a different global shape: "
                f"expected {self._global_shape}, got {tuple(weight.shape)}"
            )
        self.weight = weight
        self.weight._nano_megatron_tied_source_rank = self.source_rank  # type: ignore[attr-defined]

    @staticmethod
    def _local_tensor(tensor: Tensor) -> Tensor:
        """Return the owned DTensor shard, or the tensor itself when replicated."""

        to_local = getattr(tensor, "to_local", None)
        return to_local() if callable(to_local) else tensor

    @property
    def source_rank(self) -> int:
        ranks = getattr(self.group, "ranks", None)
        if ranks is not None:
            return int(ranks[0])
        global_rank_at = getattr(self.group, "global_rank_at", None)
        if callable(global_rank_at):
            return int(global_rank_at(0))
        # Pure unit-test groups may carry only rank/size and no process group.
        # Real multi-rank synchronization is still rejected by _process_group.
        return 0

    @property
    def weight_is_synchronized(self) -> bool:
        return self._weight_is_synchronized

    def _process_group(self) -> dist.ProcessGroup:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "a multi-rank tied embedding group requires initialized torch.distributed"
            )
        if self.group.process_group is None:
            raise RuntimeError(
                "a multi-rank tied embedding group requires a materialized process_group"
            )
        return self.group.process_group

    @property
    def active(self) -> bool:
        return self.group.size > 1

    @torch.no_grad()
    def synchronize_weight(self, *, force: bool = False) -> None:
        """Broadcast the first-stage embedding shard to the last-stage LM head."""

        if not self.active or (self._weight_is_synchronized and not force):
            return
        dist.broadcast(
            self._local_tensor(self.weight),
            src=self.source_rank,
            group=self._process_group(),
        )
        self._weight_is_synchronized = True

    @torch.no_grad()
    def synchronize_gradient(self) -> None:
        """Sum input-embedding and LM-head contributions on both endpoints."""

        if not self.active:
            return
        if self.weight.grad is None:
            raise RuntimeError(
                "tied embedding gradient synchronization requires a completed backward "
                "on both pipeline endpoints"
            )
        dist.all_reduce(
            self._local_tensor(self.weight.grad),
            op=dist.ReduceOp.SUM,
            group=self._process_group(),
        )
