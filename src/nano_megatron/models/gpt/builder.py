"""Build the local GPT stage for an explicit pipeline-parallel context."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from torch import Tensor, nn

from nano_megatron.models.gpt.factory import DenseGPTComponents, GPTComponentFactory
from nano_megatron.models.gpt.model import GPTModel
from nano_megatron.models.gpt.tied_embeddings import TiedEmbeddingSynchronizer
from nano_megatron.parallel import GroupKey, ParameterDomain, ParameterDomainRegistry
from nano_megatron.pipeline_parallel import LayerPartition, partition_for_rank
from nano_megatron.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    VocabParallelLinear,
)

if TYPE_CHECKING:
    from nano_megatron.config import GPTConfig
    from nano_megatron.nn.kernels.interface import KernelBackend
    from nano_megatron.parallel import ParallelContext


@dataclass(frozen=True)
class BuiltGPTStage:
    model: GPTPipelineStage
    partition: LayerPartition
    parameter_domains: ParameterDomainRegistry


class GPTModelBuilder:
    def __init__(
        self,
        components: GPTComponentFactory | None = None,
        *,
        parameter_domains: ParameterDomainRegistry | None = None,
        activation_checkpoint_config: Any | None = None,
    ) -> None:
        self.components = components or DenseGPTComponents()
        self.parameter_domains = parameter_domains
        self.activation_checkpoint_config = activation_checkpoint_config

    @staticmethod
    def _register_parameter_domains(
        model: GPTModel,
        registry: ParameterDomainRegistry,
    ) -> None:
        tensor_shard_dims: dict[int, int] = {}
        for module in model.modules():
            if isinstance(module, ColumnParallelLinear):
                tensor_shard_dims[id(module.weight)] = 0
                if module.bias is not None:
                    tensor_shard_dims[id(module.bias)] = 0
            elif isinstance(module, RowParallelLinear):
                tensor_shard_dims[id(module.weight)] = 1
            elif isinstance(module, (VocabParallelEmbedding, VocabParallelLinear)):
                tensor_shard_dims[id(module.weight)] = 0
                if isinstance(module, VocabParallelLinear) and module.bias is not None:
                    tensor_shard_dims[id(module.bias)] = 0

        for name, parameter in model.named_parameters():
            if parameter in registry:
                continue
            registry.register(
                parameter,
                domain=ParameterDomain.DENSE,
                tensor_sharded=id(parameter) in tensor_shard_dims,
                tensor_shard_dim=tensor_shard_dims.get(id(parameter)),
                name=name,
            )
        registry.validate_complete(model.parameters())

    def build_stage(
        self,
        model_config: GPTConfig,
        parallel: ParallelContext,
        kernels: KernelBackend,
        *,
        rng: Any | None = None,
    ) -> BuiltGPTStage:
        partition = partition_for_rank(
            model_config.num_layers,
            parallel.pp.size,
            parallel.pp.rank,
        )
        core_model = GPTModel(
            model_config,
            parallel=parallel,
            kernels=kernels,
            rng=rng,
            components=self.components,
            parameter_domains=self.parameter_domains,
            activation_checkpoint_config=self.activation_checkpoint_config,
            layer_start=partition.start_layer,
            layer_end=partition.end_layer,
            owns_embedding=partition.owns_embedding,
            owns_final_norm=partition.owns_final_norm,
            owns_lm_head=partition.owns_lm_head,
        )
        registry = (
            self.parameter_domains
            if self.parameter_domains is not None
            else ParameterDomainRegistry()
        )
        self._register_parameter_domains(core_model, registry)
        tied_embeddings = self._build_tied_embedding_synchronizer(
            model_config,
            parallel,
            core_model,
            partition,
        )
        model = GPTPipelineStage(core_model, partition, tied_embeddings=tied_embeddings)
        return BuiltGPTStage(
            model=model,
            partition=partition,
            parameter_domains=registry,
        )

    @staticmethod
    def _build_tied_embedding_synchronizer(
        model_config: GPTConfig,
        parallel: ParallelContext,
        model: GPTModel,
        partition: LayerPartition,
    ) -> TiedEmbeddingSynchronizer | None:
        if not model_config.tie_embeddings or parallel.pp.size == 1:
            return None
        if not (partition.owns_embedding or partition.owns_lm_head):
            return None
        if not parallel.has_group(GroupKey.EMBEDDING):
            raise RuntimeError(
                "a tied GPT pipeline endpoint is missing its declared embedding group"
            )
        if partition.owns_embedding:
            if model.embedding is None:
                raise RuntimeError("the first GPT pipeline stage has no embedding weight")
            weight = model.embedding.weight
        else:
            if model.lm_head is None:
                raise RuntimeError("the last GPT pipeline stage has no LM-head weight")
            weight = model.lm_head.weight
        return TiedEmbeddingSynchronizer(weight, parallel.group(GroupKey.EMBEDDING))


_ABSENT = object()


def _batch_value(batch: Any, key: str, default: Any = _ABSENT) -> Any:
    if isinstance(batch, Mapping):
        if key in batch:
            return batch[key]
    elif batch is not None and hasattr(batch, key):
        return getattr(batch, key)
    if default is not _ABSENT:
        return default
    raise KeyError(f"pipeline batch is missing required field {key!r}")


class GPTPipelineStage(nn.Module):
    """Adapt :class:`GPTModel` to the narrow pipeline schedule protocol."""

    def __init__(
        self,
        model: GPTModel,
        partition: LayerPartition,
        *,
        tied_embeddings: TiedEmbeddingSynchronizer | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.partition = partition
        self.tied_embeddings = tied_embeddings

    def _refresh_tied_embedding_weight(self) -> None:
        if self.tied_embeddings is None:
            return
        if self.partition.owns_embedding:
            if self.model.embedding is None:
                raise RuntimeError("the first GPT pipeline stage has no embedding weight")
            weight = self.model.embedding.weight
        else:
            if self.model.lm_head is None:
                raise RuntimeError("the last GPT pipeline stage has no LM-head weight")
            weight = self.model.lm_head.weight
        self.tied_embeddings.bind_weight(weight)

    def synchronize_tied_embedding_weights(self) -> None:
        if self.tied_embeddings is not None:
            self._refresh_tied_embedding_weight()
            self.tied_embeddings.synchronize_weight()

    def synchronize_tied_embedding_gradients(self) -> None:
        if self.tied_embeddings is not None:
            self._refresh_tied_embedding_weight()
            self.tied_embeddings.synchronize_gradient()

    def forward(self, hidden_states: Tensor | None, batch: Any) -> Tensor:
        input_ids = _batch_value(batch, "input_ids") if self.partition.owns_embedding else None
        labels = _batch_value(batch, "labels", None) if self.partition.owns_lm_head else None
        position_ids = _batch_value(batch, "position_ids", None)
        microbatch_index = int(_batch_value(batch, "_microbatch_index", 0))
        output = self.model(
            input_ids,
            labels=labels,
            position_ids=position_ids,
            hidden_states=hidden_states,
            microbatch_index=microbatch_index,
        )
        if not self.partition.owns_lm_head:
            return output.hidden_states
        if output.loss is None:
            raise ValueError("the last GPT pipeline stage requires batch['labels']")
        return output.loss
