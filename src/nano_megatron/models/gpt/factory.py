"""Typed GPT component factory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from torch import nn

from nano_megatron.models.gpt._config import config_value
from nano_megatron.models.gpt.attention import ContextParallelAttention, GPTAttention
from nano_megatron.models.gpt.mlp import GPTMLP

if TYPE_CHECKING:
    from nano_megatron.config import GPTConfig
    from nano_megatron.nn.kernels.interface import KernelBackend
    from nano_megatron.parallel import ParallelContext


class GPTComponentFactory(Protocol):
    def build_attention(
        self,
        config: GPTConfig,
        layer_index: int,
        *,
        parallel: ParallelContext,
        kernels: KernelBackend,
        rng: Any | None = None,
    ) -> nn.Module: ...

    def build_mlp(
        self,
        config: GPTConfig,
        layer_index: int,
        *,
        parallel: ParallelContext,
        kernels: KernelBackend,
        parameter_domains: Any | None = None,
    ) -> nn.Module: ...

    def build_norm(
        self,
        config: GPTConfig,
        *,
        kernels: KernelBackend,
    ) -> nn.Module: ...


class DenseGPTComponents:
    """Default dense GPT components; MoE can later replace only ``build_mlp``."""

    def __init__(
        self,
        *,
        cp_attention: ContextParallelAttention | nn.Module | None = None,
        sequence_parallel: bool | None = None,
    ) -> None:
        self.cp_attention = cp_attention
        self.sequence_parallel = sequence_parallel

    def build_attention(
        self,
        config: GPTConfig,
        layer_index: int,
        *,
        parallel: ParallelContext,
        kernels: KernelBackend,
        rng: Any | None = None,
    ) -> nn.Module:
        return GPTAttention(
            config,
            layer_index,
            parallel=parallel,
            kernels=kernels,
            rng=rng,
            cp_attention=self.cp_attention,
            sequence_parallel=self.sequence_parallel,
        )

    def build_mlp(
        self,
        config: GPTConfig,
        layer_index: int,
        *,
        parallel: ParallelContext,
        kernels: KernelBackend,
        parameter_domains: Any | None = None,
    ) -> nn.Module:
        del parameter_domains
        return GPTMLP(
            config,
            layer_index,
            parallel=parallel,
            kernels=kernels,
            sequence_parallel=self.sequence_parallel,
        )

    def build_norm(
        self,
        config: GPTConfig,
        *,
        kernels: KernelBackend,
    ) -> nn.Module:
        return kernels.rms_norm(
            int(config_value(config, "hidden_size")),
            float(
                config_value(
                    config,
                    "norm_epsilon",
                    "norm_eps",
                    "rms_norm_eps",
                    default=1.0e-6,
                )
            ),
        )
