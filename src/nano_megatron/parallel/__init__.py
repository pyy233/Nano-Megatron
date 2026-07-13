"""Five-dimensional topology and explicitly injected parallel state.

Imports are resolved lazily to keep the pure ``parallel.axes`` module usable
while the configuration package itself is being imported.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DEFAULT_RANK_ORDER": (".axes", "DEFAULT_RANK_ORDER"),
    "PARALLEL_AXES": (".axes", "PARALLEL_AXES"),
    "ParallelAxis": (".axes", "ParallelAxis"),
    "ParallelCoordinate": (".axes", "ParallelCoordinate"),
    "normalize_rank_order": (".axes", "normalize_rank_order"),
    "ParallelTopology": (".topology", "ParallelTopology"),
    "DEFAULT_GROUP_PLAN": (".group_plan", "DEFAULT_GROUP_PLAN"),
    "GroupKey": (".group_plan", "GroupKey"),
    "GroupKeyLike": (".group_plan", "GroupKeyLike"),
    "GroupPlan": (".group_plan", "GroupPlan"),
    "GroupSpec": (".group_plan", "GroupSpec"),
    "PlannedGroup": (".group_plan", "PlannedGroup"),
    "ParallelGroup": (".group", "ParallelGroup"),
    "GroupRegistry": (".registry", "GroupRegistry"),
    "ParallelGroupRegistry": (".registry", "ParallelGroupRegistry"),
    "ParallelContext": (".context", "ParallelContext"),
    "ParameterDomain": (".domains", "ParameterDomain"),
    "ParameterDomainRegistry": (".domains", "ParameterDomainRegistry"),
    "ParameterPlacement": (".domains", "ParameterPlacement"),
    "default_replica_group": (".domains", "default_replica_group"),
    "TensorLayout": (".layout", "TensorLayout"),
    "ParallelRNG": (".rng", "ParallelRNG"),
    "RNGStream": (".rng", "RNGStream"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
