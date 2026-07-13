"""Small config accessors kept local to the GPT implementation."""

from __future__ import annotations

from typing import Any

_MISSING = object()


def config_value(config: Any, *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    if default is not _MISSING:
        return default
    joined = ", ".join(names)
    raise AttributeError(f"configuration is missing required field (one of: {joined})")


def sequence_parallel_enabled(parallel: Any, override: bool | None) -> bool:
    if override is not None:
        return override
    return bool(getattr(parallel, "sequence_parallel", False))
