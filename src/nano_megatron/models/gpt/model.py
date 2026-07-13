"""Decoder-only GPT model assembled from explicitly injected dependencies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from nano_megatron.models.gpt._config import config_value, sequence_parallel_enabled
from nano_megatron.models.gpt.factory import DenseGPTComponents, GPTComponentFactory
from nano_megatron.models.gpt.layer import GPTLayer
from nano_megatron.nn.activation_checkpoint import activation_checkpoint
from nano_megatron.nn.dropout import parallel_dropout
from nano_megatron.tensor_parallel.cross_entropy import VocabParallelCrossEntropy
from nano_megatron.tensor_parallel.embedding import VocabParallelEmbedding
from nano_megatron.tensor_parallel.layers import VocabParallelLinear
from nano_megatron.tensor_parallel.sequence_parallel import (
    register_sequence_parallel_gradient_hooks,
)

if TYPE_CHECKING:
    from nano_megatron.config import GPTConfig
    from nano_megatron.models.gpt.attention import ContextParallelAttention
    from nano_megatron.nn.kernels.interface import KernelBackend
    from nano_megatron.parallel import ParallelContext


@dataclass
class GPTOutput:
    """Model output; logits remain vocabulary-sharded when present."""

    logits: Tensor | None
    loss: Tensor | None
    hidden_states: Tensor


class GPTModel(nn.Module):
    def __init__(
        self,
        config: GPTConfig,
        *,
        parallel: ParallelContext,
        kernels: KernelBackend,
        rng: Any | None = None,
        components: GPTComponentFactory | None = None,
        cp_attention: ContextParallelAttention | nn.Module | None = None,
        parameter_domains: Any | None = None,
        activation_checkpoint_config: Any | None = None,
        sequence_parallel: bool | None = None,
        layer_start: int = 0,
        layer_end: int | None = None,
        owns_embedding: bool = True,
        owns_final_norm: bool = True,
        owns_lm_head: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.parallel = parallel
        self.kernels = kernels
        self.rng = rng
        self.activation_checkpoint_config = activation_checkpoint_config
        self.sequence_parallel = sequence_parallel_enabled(parallel, sequence_parallel)
        self.hidden_size = int(config_value(config, "hidden_size"))
        self.num_layers = int(config_value(config, "layers", "num_layers"))
        self.original_vocab_size = int(config_value(config, "vocab_size"))
        self.padded_vocab_size = (
            math.ceil(self.original_vocab_size / parallel.tp.size) * parallel.tp.size
        )
        self.dropout = float(config_value(config, "embedding_dropout", "dropout", default=0.0))
        self.owns_embedding = owns_embedding
        self.owns_final_norm = owns_final_norm
        self.owns_lm_head = owns_lm_head
        self.layer_start = layer_start
        self.layer_end = self.num_layers if layer_end is None else layer_end
        if not 0 <= self.layer_start <= self.layer_end <= self.num_layers:
            raise ValueError(
                f"invalid layer range [{self.layer_start}, {self.layer_end}) "
                f"for {self.num_layers} layers"
            )
        if self.layer_start == self.layer_end:
            raise ValueError("a GPT stage must own at least one transformer layer")

        init_std = float(config_value(config, "init_std", default=0.02))
        self.embedding = (
            VocabParallelEmbedding(
                self.padded_vocab_size,
                self.hidden_size,
                parallel=parallel,
                kernels=kernels,
                sequence_parallel=self.sequence_parallel,
                init_std=init_std,
            )
            if owns_embedding
            else None
        )
        if components is None:
            components = DenseGPTComponents(
                cp_attention=cp_attention,
                sequence_parallel=self.sequence_parallel,
            )
        self.layers = nn.ModuleList(
            [
                GPTLayer(
                    config,
                    layer_index,
                    parallel=parallel,
                    kernels=kernels,
                    rng=rng,
                    components=components,
                    parameter_domains=parameter_domains,
                    checkpoint_config=activation_checkpoint_config,
                )
                for layer_index in range(self.layer_start, self.layer_end)
            ]
        )
        self.final_norm = (
            components.build_norm(config, kernels=kernels) if owns_final_norm else None
        )
        if self.sequence_parallel and self.final_norm is not None:
            register_sequence_parallel_gradient_hooks(self.final_norm, parallel.tp)

        tied_weight = None
        if (
            bool(config_value(config, "tie_embeddings", default=True))
            and self.embedding is not None
        ):
            tied_weight = self.embedding.weight
        self.lm_head = (
            VocabParallelLinear(
                self.hidden_size,
                self.padded_vocab_size,
                parallel=parallel,
                sequence_parallel=self.sequence_parallel,
                weight=tied_weight,
                init_std=init_std,
            )
            if owns_lm_head
            else None
        )
        self.loss_fn = (
            VocabParallelCrossEntropy(
                parallel=parallel,
                reduction="mean",
                original_vocab_size=self.original_vocab_size,
            )
            if owns_lm_head
            else None
        )

    def _sequence_offset(self, local_cp_sequence_length: int) -> int:
        cp_group = self.parallel.cp
        return cp_group.rank * local_cp_sequence_length

    def _activation_offset(self, hidden_states: Tensor, sequence_offset: int) -> int:
        if not self.sequence_parallel:
            return sequence_offset
        return sequence_offset + self.parallel.tp.rank * hidden_states.shape[1]

    def _checkpoint_layer(self, layer: GPTLayer) -> bool:
        config = self.activation_checkpoint_config
        if config is None or getattr(config, "mode", "none") != "full" or not self.training:
            return False
        interval = int(getattr(config, "block_interval", 1))
        if interval <= 0:
            raise ValueError("activation checkpoint block_interval must be positive")
        return layer.layer_index % interval == 0

    def forward(
        self,
        input_ids: Tensor | None = None,
        *,
        labels: Tensor | None = None,
        position_ids: Tensor | None = None,
        hidden_states: Tensor | None = None,
        microbatch_index: int = 0,
    ) -> GPTOutput:
        embedded = self.embedding is not None
        if self.embedding is not None:
            if input_ids is None:
                raise ValueError("the first pipeline stage requires input_ids")
            hidden_states = self.embedding(input_ids)
            local_cp_sequence_length = input_ids.shape[1]
        else:
            if hidden_states is None:
                raise ValueError("a non-first pipeline stage requires hidden_states")
            local_cp_sequence_length = hidden_states.shape[1]
            if self.sequence_parallel:
                local_cp_sequence_length *= self.parallel.tp.size

        if position_ids is None:
            offset = self._sequence_offset(local_cp_sequence_length)
            position_ids = torch.arange(
                offset,
                offset + local_cp_sequence_length,
                device=hidden_states.device,
            )
        else:
            offset = int(position_ids.reshape(-1)[0].item())

        if embedded:
            hidden_states = parallel_dropout(
                hidden_states,
                self.dropout,
                training=self.training,
                rng=self.rng,
                layer=0,
                microbatch=microbatch_index,
                global_token_offset=self._activation_offset(hidden_states, offset),
                op="embedding",
            )

        for layer in self.layers:
            if self._checkpoint_layer(layer):
                hidden_states = activation_checkpoint(
                    lambda x, current_layer=layer: current_layer(
                        x,
                        position_ids=position_ids,
                        sequence_offset=offset,
                        microbatch_index=microbatch_index,
                    ),
                    hidden_states,
                    rng=self.rng,
                    offload_saved_tensors=bool(
                        getattr(
                            self.activation_checkpoint_config,
                            "offload_saved_tensors",
                            False,
                        )
                    ),
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    position_ids=position_ids,
                    sequence_offset=offset,
                    microbatch_index=microbatch_index,
                )

        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states) if self.lm_head is not None else None
        loss = None
        if labels is not None:
            if logits is None or self.loss_fn is None:
                raise ValueError("labels may only be supplied to a stage that owns the LM head")
            loss = self.loss_fn(logits, labels)
        return GPTOutput(logits=logits, loss=loss, hidden_states=hidden_states)
