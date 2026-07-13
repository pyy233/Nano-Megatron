from __future__ import annotations

from ._zero import FlatShardZeROStrategy


class Zero2Strategy(FlatShardZeROStrategy):
    """Reduce-scattered gradients and optimizer state partitioned by parameter domain."""

    mode = "zero2"
    gradient_partitioned = True
