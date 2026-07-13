"""Context-parallel attention backends."""

from .all_gather import AllGatherContextParallelAttention
from .interface import ContextParallelAttention
from .ring import RingContextParallelAttention


def build_context_parallel_attention(config, parallel):
    """Build the configured CP backend with an explicitly supplied CP group."""

    if parallel.cp.size == 1:
        return None
    configured = getattr(config, "backend", config)
    backend = str(getattr(configured, "value", configured))
    backend = backend.lower()
    dropout_p = float(getattr(config, "dropout", 0.0))
    if backend == "all_gather":
        return AllGatherContextParallelAttention(parallel.cp, dropout_p=dropout_p)
    if backend == "ring":
        return RingContextParallelAttention(parallel.cp, dropout_p=dropout_p)
    raise ValueError(f"unknown context-parallel backend: {backend}")

__all__ = [
    "AllGatherContextParallelAttention",
    "ContextParallelAttention",
    "RingContextParallelAttention",
    "build_context_parallel_attention",
]
