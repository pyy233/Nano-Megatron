"""Pre-norm GPT transformer layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import Tensor, nn

from nano_megatron.models.gpt._config import config_value
from nano_megatron.nn.activation_checkpoint import activation_checkpoint
from nano_megatron.nn.dropout import parallel_dropout
from nano_megatron.tensor_parallel import register_sequence_parallel_gradient_hooks

if TYPE_CHECKING:
    from nano_megatron.config import GPTConfig
    from nano_megatron.models.gpt.attention import ContextParallelAttention
    from nano_megatron.models.gpt.factory import GPTComponentFactory
    from nano_megatron.nn.kernels.interface import KernelBackend
    from nano_megatron.parallel import ParallelContext


class GPTLayer(nn.Module):
    """Readable pre-norm attention + SwiGLU transformer block."""

    def __init__(
        self,
        config: GPTConfig,
        layer_index: int,
        *,
        parallel: ParallelContext,
        kernels: KernelBackend,
        cp_attention: ContextParallelAttention | nn.Module | None = None,
        rng: Any | None = None,
        components: GPTComponentFactory | None = None,
        parameter_domains: Any | None = None,
        checkpoint_config: Any | None = None,
    ) -> None:
        super().__init__()
        if components is None:
            from nano_megatron.models.gpt.factory import DenseGPTComponents

            components = DenseGPTComponents(cp_attention=cp_attention)
        self.layer_index = layer_index
        self.parallel = parallel
        self.rng = rng
        self.checkpoint_config = checkpoint_config
        self.input_norm = components.build_norm(config, kernels=kernels)
        self.attention = components.build_attention(
            config,
            layer_index,
            parallel=parallel,
            kernels=kernels,
            rng=rng,
        )
        self.post_attention_norm = components.build_norm(config, kernels=kernels)
        self.mlp = components.build_mlp(
            config,
            layer_index,
            parallel=parallel,
            kernels=kernels,
            parameter_domains=parameter_domains,
        )
        self.dropout = float(config_value(config, "hidden_dropout", "dropout", default=0.0))
        if parallel.sequence_parallel:
            register_sequence_parallel_gradient_hooks(self.input_norm, parallel.tp)
            register_sequence_parallel_gradient_hooks(self.post_attention_norm, parallel.tp)

    def _selectively_checkpointed(self, name: str) -> bool:
        config = self.checkpoint_config
        if config is None or getattr(config, "mode", "none") != "selective":
            return False
        return name in tuple(getattr(config, "selective_ops", ("attention", "mlp")))

    def _run(
        self,
        name: str,
        function,
        hidden_states: Tensor,
    ) -> Tensor:
        if not self._selectively_checkpointed(name) or not self.training:
            return function(hidden_states)
        return activation_checkpoint(
            function,
            hidden_states,
            rng=self.rng,
            offload_saved_tensors=bool(
                getattr(self.checkpoint_config, "offload_saved_tensors", False)
            ),
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        position_ids: Tensor | None = None,
        sequence_offset: int = 0,
        microbatch_index: int = 0,
    ) -> Tensor:
        activation_offset = sequence_offset
        if self.parallel.sequence_parallel:
            activation_offset += self.parallel.tp.rank * hidden_states.shape[1]
        residual = hidden_states
        normalized = self.input_norm(hidden_states)
        attention_output = self._run(
            "attention",
            lambda x: self.attention(
                x,
                position_ids=position_ids,
                sequence_offset=sequence_offset,
                microbatch_index=microbatch_index,
            ),
            normalized,
        )
        hidden_states = residual + parallel_dropout(
            attention_output,
            self.dropout,
            training=self.training,
            rng=self.rng,
            layer=self.layer_index,
            microbatch=microbatch_index,
            global_token_offset=activation_offset,
            op="attention_residual",
        )

        residual = hidden_states
        normalized = self.post_attention_norm(hidden_states)
        mlp_output = self._run("mlp", self.mlp, normalized)
        return residual + parallel_dropout(
            mlp_output,
            self.dropout,
            training=self.training,
            rng=self.rng,
            layer=self.layer_index,
            microbatch=microbatch_index,
            global_token_offset=activation_offset,
            op="mlp_residual",
        )
