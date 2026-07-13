from __future__ import annotations

from ._zero import FlatShardZeROStrategy


class Zero1Strategy(FlatShardZeROStrategy):
    """Replicated gradients with optimizer state partitioned by parameter domain."""

    mode = "zero1"
    gradient_partitioned = False
