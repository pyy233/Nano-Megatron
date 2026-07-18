from __future__ import annotations

from dataclasses import dataclass

import torch

from nano_megatron.config import ActivationCheckpointConfig
from nano_megatron.models.gpt import (
    DenseGPTComponents,
    GPTAttention,
    GPTModel,
    GPTModelBuilder,
    LayerPartition,
)
from nano_megatron.nn.kernels import (
    TorchKernelBackend,
    build_kernel_backend,
)
from nano_megatron.parallel import (
    ParallelCoordinate,
    ParallelRNG,
    ParameterDomain,
)
from nano_megatron.pipeline_parallel import partition_for_rank


@dataclass(frozen=True)
class FakeGroup:
    rank: int = 0
    size: int = 1
    process_group: object | None = None


@dataclass(frozen=True)
class FakeParallel:
    tp: FakeGroup = FakeGroup()
    pp: FakeGroup = FakeGroup()
    cp: FakeGroup = FakeGroup()
    sequence_parallel: bool = False

    def has_group(self, key: object) -> bool:
        return str(getattr(key, "value", key)) == "embedding" and self.pp.size > 1

    def group(self, key: object) -> FakeGroup:
        if not self.has_group(key):
            raise KeyError(key)
        endpoint_rank = 0 if self.pp.rank == 0 else 1
        return FakeGroup(rank=endpoint_rank, size=2)


@dataclass
class TinyGPTConfig:
    layers: int = 2
    hidden_size: int = 16
    ffn_hidden_size: int = 32
    heads: int = 4
    kv_heads: int = 2
    seq_length: int = 8
    vocab_size: int = 31
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    tie_embeddings: bool = True

    @property
    def num_layers(self) -> int:
        return self.layers


def test_tiny_gpt_forward_loss_backward_and_tied_weights() -> None:
    torch.manual_seed(17)
    config = TinyGPTConfig()
    model = GPTModel(
        config,
        parallel=FakeParallel(),
        kernels=TorchKernelBackend(),
    )
    input_ids = torch.randint(0, config.vocab_size, (2, config.seq_length))
    labels = torch.randint(0, config.vocab_size, (2, config.seq_length))
    output = model(input_ids, labels=labels)
    assert output.logits is not None
    assert output.logits.shape == (2, config.seq_length, config.vocab_size)
    assert output.loss is not None and output.loss.ndim == 0
    assert torch.isfinite(output.loss)
    assert model.embedding is not None and model.lm_head is not None
    assert model.embedding.weight is model.lm_head.weight
    output.loss.backward()
    assert model.embedding.weight.grad is not None
    assert torch.isfinite(model.embedding.weight.grad).all()


def test_sequence_parallel_size_one_has_same_model_semantics() -> None:
    torch.manual_seed(19)
    config = TinyGPTConfig(layers=1)
    parallel = FakeParallel(sequence_parallel=True)
    model = GPTModel(config, parallel=parallel, kernels=TorchKernelBackend())
    input_ids = torch.randint(0, config.vocab_size, (1, config.seq_length))
    output = model(input_ids)
    assert output.logits is not None
    assert output.logits.shape == (1, config.seq_length, config.vocab_size)


class RecordingCPAttention:
    def __init__(self, kernels: TorchKernelBackend) -> None:
        self.kernels = kernels
        self.shapes: list[tuple[torch.Size, torch.Size, torch.Size]] = []

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        sequence_offset: int = 0,
    ) -> torch.Tensor:
        self.shapes.append((q.shape, k.shape, v.shape))
        return self.kernels.local_attention(
            q,
            k,
            v,
            causal=causal,
            sequence_offset=sequence_offset,
        )


def test_attention_passes_bhsd_layout_to_cp_backend() -> None:
    config = TinyGPTConfig(layers=1)
    kernels = TorchKernelBackend()
    recorder = RecordingCPAttention(kernels)
    attention = GPTAttention(
        config,
        0,
        parallel=FakeParallel(),
        kernels=kernels,
        cp_attention=recorder,
    )
    output = attention(torch.randn(2, 5, config.hidden_size))
    assert output.shape == (2, 5, config.hidden_size)
    assert recorder.shapes == [
        (
            torch.Size((2, config.heads, 5, config.hidden_size // config.heads)),
            torch.Size((2, config.kv_heads, 5, config.hidden_size // config.heads)),
            torch.Size((2, config.kv_heads, 5, config.hidden_size // config.heads)),
        )
    ]


def test_layer_partition_distributes_remainder_to_early_stages() -> None:
    assert partition_for_rank(7, 3, 0) == LayerPartition(0, 0, 3, True, False, False)
    assert partition_for_rank(7, 3, 1) == LayerPartition(1, 3, 5, False, False, False)
    assert partition_for_rank(7, 3, 2) == LayerPartition(2, 5, 7, False, True, True)


def test_builder_constructs_the_local_pipeline_stage() -> None:
    config = TinyGPTConfig(layers=4)
    parallel = FakeParallel(pp=FakeGroup(rank=1, size=2))
    builder = GPTModelBuilder(DenseGPTComponents())
    built = builder.build_stage(config, parallel, TorchKernelBackend())
    assert built.partition == LayerPartition(1, 2, 4, False, True, True)
    assert built.model.model.embedding is None
    assert built.model.model.final_norm is not None
    assert built.model.model.lm_head is not None
    assert built.model.tied_embeddings is not None
    built.model.to(dtype=torch.float64)
    assert built.model.tied_embeddings.weight is built.model.model.lm_head.weight
    assert built.model.tied_embeddings.weight.dtype is torch.float64
    assert len(built.parameter_domains) == len(tuple(built.model.parameters()))
    assert all(
        built.parameter_domains.domain(parameter) is ParameterDomain.DENSE
        for parameter in built.model.parameters()
    )
    for module in built.model.modules():
        if isinstance(module, GPTAttention):
            assert built.parameter_domains.placement(module.q_proj.weight).tensor_shard_dim == 0
            assert built.parameter_domains.placement(module.out_proj.weight).tensor_shard_dim == 1


def test_builder_constructs_explicit_virtual_pipeline_chunks() -> None:
    config = TinyGPTConfig(layers=8)
    parallel = FakeParallel(pp=FakeGroup(rank=0, size=2))
    built = GPTModelBuilder(DenseGPTComponents()).build_pipeline(
        config,
        parallel,
        TorchKernelBackend(),
        virtual_stages_per_rank=2,
    )

    assert [
        (partition.start_layer, partition.end_layer)
        for partition in built.partitions
    ] == [(0, 2), (4, 6)]
    assert built.model.chunk(0).partition.owns_embedding
    assert not built.model.chunk(1).partition.owns_lm_head
    assert built.model.sharding_units() == tuple(built.model.chunks)
    assert len(built.parameter_domains) == len(tuple(built.model.parameters()))
    assert all(
        built.parameter_domains.domain(parameter) is ParameterDomain.DENSE
        for parameter in built.model.parameters()
    )


def test_single_stage_pipeline_wrapper_unpacks_batch_and_returns_loss() -> None:
    config = TinyGPTConfig(layers=1)
    built = GPTModelBuilder().build_stage(
        config,
        FakeParallel(),
        TorchKernelBackend(),
    )
    batch = {
        "input_ids": torch.randint(0, config.vocab_size, (2, config.seq_length)),
        "labels": torch.randint(0, config.vocab_size, (2, config.seq_length)),
    }
    output = built.model(None, batch)
    assert isinstance(output, torch.Tensor)
    assert output.ndim == 0


def test_kernel_backend_factory_builds_torch_reference() -> None:
    config = type("KernelConfig", (), {"backend": "torch"})()
    assert isinstance(build_kernel_backend(config, FakeParallel()), TorchKernelBackend)


def test_explicit_activation_rng_keys_dropout_by_microbatch_identity() -> None:
    torch.manual_seed(23)
    config = TinyGPTConfig(layers=1, dropout=0.35)
    rng = ParallelRNG(91, ParallelCoordinate(0, 0, 0, 0, 0))
    model = GPTModel(
        config,
        parallel=FakeParallel(),
        kernels=TorchKernelBackend(),
        rng=rng,
    ).train()
    input_ids = torch.randint(0, config.vocab_size, (2, config.seq_length))

    first = model(input_ids, microbatch_index=4).logits
    torch.manual_seed(999_999)
    repeated = model(input_ids, microbatch_index=4).logits
    different = model(input_ids, microbatch_index=5).logits

    assert first is not None and repeated is not None and different is not None
    torch.testing.assert_close(first, repeated)
    assert not torch.equal(first, different)


def test_selective_checkpoint_recomputes_the_selected_attention() -> None:
    class CountingAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, hidden_states, **_):
            self.calls += 1
            return torch.sin(hidden_states)

    config = TinyGPTConfig(layers=1)
    checkpoint_config = ActivationCheckpointConfig(
        mode="selective",
        selective_ops=("attention",),
    )
    model = GPTModel(
        config,
        parallel=FakeParallel(),
        kernels=TorchKernelBackend(),
        activation_checkpoint_config=checkpoint_config,
    ).train()
    attention = CountingAttention()
    model.layers[0].attention = attention
    input_ids = torch.randint(0, config.vocab_size, (1, config.seq_length))
    labels = torch.randint(0, config.vocab_size, (1, config.seq_length))

    output = model(input_ids, labels=labels)
    assert output.loss is not None
    output.loss.backward()

    assert attention.calls >= 2


def test_full_checkpoint_recomputes_the_selected_layer() -> None:
    class CountingLayer(torch.nn.Module):
        def __init__(self, layer: torch.nn.Module) -> None:
            super().__init__()
            self.layer = layer
            self.layer_index = layer.layer_index
            self.calls = 0

        def forward(self, *args, **kwargs):
            self.calls += 1
            return self.layer(*args, **kwargs)

    config = TinyGPTConfig(layers=1)
    model = GPTModel(
        config,
        parallel=FakeParallel(),
        kernels=TorchKernelBackend(),
        activation_checkpoint_config=ActivationCheckpointConfig(
            mode="full",
            block_interval=1,
        ),
    ).train()
    layer = CountingLayer(model.layers[0])
    model.layers[0] = layer
    input_ids = torch.randint(0, config.vocab_size, (1, config.seq_length))
    labels = torch.randint(0, config.vocab_size, (1, config.seq_length))

    output = model(input_ids, labels=labels)
    assert output.loss is not None
    output.loss.backward()

    assert layer.calls >= 2
