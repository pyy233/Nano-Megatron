from __future__ import annotations

from typing import TYPE_CHECKING

from .buckets import (
    BucketGradientReducer,
    FlatBucket,
    GradientReductionRequest,
    ParameterSlice,
    build_flat_buckets,
)
from .ddp import DDPStrategy, ReplicatedStrategy
from .interface import DataParallelStrategy
from .offload import ActivationOffloader, OffloadPolicy, OptimizerStateStorage
from .zero1 import Zero1Strategy
from .zero2 import Zero2Strategy
from .zero3 import Zero3Strategy

if TYPE_CHECKING:
    from nano_megatron.parallel import ParallelContext, ParameterDomainRegistry


def build_data_parallel_strategy(
    config: object,
    offload: object | None,
    parallel: ParallelContext,
    parameter_domains: ParameterDomainRegistry,
) -> DataParallelStrategy:
    mode = str(getattr(config, "mode", "ddp")).lower().replace("-", "")
    strategies: dict[str, type[DataParallelStrategy]] = {
        "ddp": DDPStrategy,
        "replicated": ReplicatedStrategy,
        "zero1": Zero1Strategy,
        "zero2": Zero2Strategy,
        "zero3": Zero3Strategy,
    }
    try:
        strategy_type = strategies[mode]
    except KeyError as error:
        choices = ", ".join(sorted(strategies))
        raise ValueError(f"unknown data-parallel mode {mode!r}; choose one of {choices}") from error
    return strategy_type(
        config=config,
        offload=offload,
        parallel=parallel,
        parameter_domains=parameter_domains,
    )


ZeRO1Strategy = Zero1Strategy
ZeRO2Strategy = Zero2Strategy
ZeRO3Strategy = Zero3Strategy

__all__ = [
    "ActivationOffloader",
    "BucketGradientReducer",
    "DDPStrategy",
    "DataParallelStrategy",
    "FlatBucket",
    "GradientReductionRequest",
    "OffloadPolicy",
    "OptimizerStateStorage",
    "ParameterSlice",
    "ReplicatedStrategy",
    "Zero1Strategy",
    "Zero2Strategy",
    "Zero3Strategy",
    "ZeRO1Strategy",
    "ZeRO2Strategy",
    "ZeRO3Strategy",
    "build_data_parallel_strategy",
    "build_flat_buckets",
]
