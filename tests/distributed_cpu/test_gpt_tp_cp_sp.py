from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from nano_megatron.config import (
    CheckpointConfig,
    ContextParallelConfig,
    DataParallelConfig,
    DistributedConfig,
    GPTConfig,
    KernelConfig,
    OffloadConfig,
    OptimizerConfig,
    ParallelConfig,
    PipelineConfig,
    PrecisionConfig,
    PrecisionDType,
    TrainConfig,
    TrainingConfig,
)
from nano_megatron.context_parallel import build_context_parallel_attention
from nano_megatron.data_parallel import build_data_parallel_strategy
from nano_megatron.distributed import DistributedRuntime
from nano_megatron.models.gpt import DenseGPTComponents, GPTModel, GPTModelBuilder
from nano_megatron.nn import RMSNorm
from nano_megatron.nn.kernels import TorchKernelBackend
from nano_megatron.parallel import ParallelContext, ParameterDomainRegistry
from nano_megatron.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    VocabParallelLinear,
)
from nano_megatron.training import Trainer, split_microbatches


@dataclass(frozen=True)
class _LocalGroup:
    rank: int = 0
    size: int = 1
    process_group: object | None = None


@dataclass(frozen=True)
class _SingleParallel:
    tp: _LocalGroup = _LocalGroup()
    pp: _LocalGroup = _LocalGroup()
    cp: _LocalGroup = _LocalGroup()
    sequence_parallel: bool = False


def _config(checkpoint_directory: Path) -> TrainConfig:
    return TrainConfig(
        distributed=DistributedConfig(backend="gloo", device="cpu"),
        parallel=ParallelConfig(
            tensor=2,
            context=2,
            data=1,
            sequence_parallel=True,
        ),
        model=GPTConfig(
            layers=2,
            hidden_size=16,
            ffn_hidden_size=32,
            heads=4,
            kv_heads=2,
            seq_length=8,
            vocab_size=32,
            dropout=0.0,
            tie_embeddings=True,
            bias=False,
        ),
        precision=PrecisionConfig(
            params=PrecisionDType.FLOAT32,
            compute=PrecisionDType.FLOAT32,
            grad_reduce=PrecisionDType.FLOAT32,
        ),
        kernels=KernelConfig(),
        data_parallel=DataParallelConfig(mode="ddp", bucket_bytes=4096),
        pipeline=PipelineConfig(schedule="gpipe"),
        context_parallel=ContextParallelConfig(backend="ring"),
        offload=OffloadConfig(),
        optimizer=OptimizerConfig(
            lr=1.0e-2,
            weight_decay=0.0,
            clip_grad_norm=None,
        ),
        training=TrainingConfig(
            micro_batch_size=1,
            gradient_accumulation_steps=2,
            max_steps=1,
        ),
        checkpoint=CheckpointConfig(directory=checkpoint_directory),
    )


def _copy_reference_to_tp(
    reference: GPTModel,
    target: GPTModel,
    *,
    tp_rank: int,
    tp_size: int,
) -> list[tuple[nn.Parameter, nn.Parameter, int | None]]:
    reference_modules = dict(reference.named_modules())
    mappings: list[tuple[nn.Parameter, nn.Parameter, int | None]] = []
    recorded: set[int] = set()

    def copy_parameter(
        target_parameter: nn.Parameter | None,
        reference_parameter: nn.Parameter | None,
        shard_dim: int | None,
    ) -> None:
        if target_parameter is None or reference_parameter is None:
            assert target_parameter is reference_parameter
            return
        if id(target_parameter) in recorded:
            return
        value = reference_parameter
        if shard_dim is not None:
            value = reference_parameter.chunk(tp_size, dim=shard_dim)[tp_rank]
        with torch.no_grad():
            target_parameter.copy_(value)
        recorded.add(id(target_parameter))
        mappings.append((target_parameter, reference_parameter, shard_dim))

    for name, module in target.named_modules():
        if isinstance(module, ColumnParallelLinear):
            reference_module = reference_modules[name]
            assert isinstance(reference_module, ColumnParallelLinear)
            copy_parameter(module.weight, reference_module.weight, 0)
            copy_parameter(module.bias, reference_module.bias, 0)
        elif isinstance(module, RowParallelLinear):
            reference_module = reference_modules[name]
            assert isinstance(reference_module, RowParallelLinear)
            copy_parameter(module.weight, reference_module.weight, 1)
            copy_parameter(module.bias, reference_module.bias, None)
        elif isinstance(module, (VocabParallelEmbedding, VocabParallelLinear)):
            reference_module = reference_modules[name]
            assert isinstance(reference_module, type(module))
            copy_parameter(module.weight, reference_module.weight, 0)
            if isinstance(module, VocabParallelLinear):
                copy_parameter(module.bias, reference_module.bias, 0)
        elif isinstance(module, RMSNorm):
            reference_module = reference_modules[name]
            assert isinstance(reference_module, RMSNorm)
            copy_parameter(module.weight, reference_module.weight, None)

    assert len(recorded) == len(tuple(target.parameters()))
    return mappings


def _expected_shard(
    tensor: torch.Tensor,
    shard_dim: int | None,
    *,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    if shard_dim is None:
        return tensor
    return tensor.chunk(tp_size, dim=shard_dim)[tp_rank]


def _worker(rank: int, rendezvous: str, checkpoint_directory: str) -> None:
    world_size = 4
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    config = _config(Path(checkpoint_directory))
    runtime = DistributedRuntime(
        DistributedConfig(
            backend="gloo",
            device="cpu",
            init_method=f"file://{rendezvous}",
        )
    ).initialize()
    parallel = ParallelContext.create(runtime, config.parallel)
    try:
        torch.manual_seed(211)
        reference = GPTModel(
            config.model,
            parallel=_SingleParallel(),
            kernels=TorchKernelBackend(),
        )
        components = DenseGPTComponents(
            cp_attention=build_context_parallel_attention(
                config.context_parallel,
                parallel,
            )
        )
        domains = ParameterDomainRegistry()
        built = GPTModelBuilder(components, parameter_domains=domains).build_stage(
            config.model,
            parallel,
            TorchKernelBackend(),
        )
        target_core = built.model.model
        mappings = _copy_reference_to_tp(
            reference,
            target_core,
            tp_rank=parallel.tp.rank,
            tp_size=parallel.tp.size,
        )

        strategy = build_data_parallel_strategy(
            config.data_parallel,
            config.offload,
            parallel,
            built.parameter_domains,
        )
        trainer = Trainer(
            config=config,
            model=built.model,
            parallel=parallel,
            data_parallel=strategy,
        )
        batch = {
            "input_ids": torch.tensor(
                [
                    [0, 3, 7, 12, 5, 18, 9, 24],
                    [31, 2, 14, 6, 20, 11, 4, 27],
                ]
            ),
            "labels": torch.tensor(
                [
                    [3, 7, 12, 5, 18, 9, 24, 1],
                    [2, 14, 6, 20, 11, 4, 27, 8],
                ]
            ),
        }
        routed = trainer.batch_router.route(
            batch if trainer.batch_router.is_source else None
        )

        with torch.no_grad():
            target_output = target_core(
                routed["input_ids"][:1],
                labels=routed["labels"][:1],
            )
            reference_output = reference(
                batch["input_ids"][:1],
                labels=batch["labels"][:1],
            )
            assert target_output.logits is not None
            assert target_output.loss is not None
            assert reference_output.logits is not None
            assert reference_output.loss is not None

            expected_hidden = reference_output.hidden_states.chunk(
                parallel.cp.size,
                dim=1,
            )[parallel.cp.rank].chunk(parallel.tp.size, dim=1)[parallel.tp.rank]
            expected_logits = reference_output.logits.chunk(
                parallel.cp.size,
                dim=1,
            )[parallel.cp.rank].chunk(parallel.tp.size, dim=-1)[parallel.tp.rank]
            torch.testing.assert_close(
                target_output.hidden_states,
                expected_hidden,
                atol=4.0e-5,
                rtol=4.0e-4,
            )
            torch.testing.assert_close(
                target_output.logits,
                expected_logits,
                atol=4.0e-5,
                rtol=4.0e-4,
            )

            reduced_loss = target_output.loss.detach().clone()
            dist.all_reduce(reduced_loss, group=parallel.cp.process_group)
            reduced_loss /= parallel.cp.size
            torch.testing.assert_close(
                reduced_loss,
                reference_output.loss,
                atol=4.0e-5,
                rtol=4.0e-4,
            )

        reference_optimizer = torch.optim.AdamW(
            reference.parameters(),
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        )
        for microbatch in range(config.training.gradient_accumulation_steps):
            output = reference(
                batch["input_ids"][microbatch : microbatch + 1],
                labels=batch["labels"][microbatch : microbatch + 1],
            )
            assert output.loss is not None
            (output.loss / config.training.gradient_accumulation_steps).backward()

        strategy.zero_grad()
        trainer.schedule.forward_backward(
            stage=trainer.model,
            microbatches=split_microbatches(
                routed,
                config.training.micro_batch_size,
            ),
            data_parallel=strategy,
        )
        strategy.finalize_gradients()

        for target_parameter, reference_parameter, shard_dim in mappings:
            assert target_parameter.grad is not None
            assert reference_parameter.grad is not None
            expected_gradient = _expected_shard(
                reference_parameter.grad,
                shard_dim,
                tp_rank=parallel.tp.rank,
                tp_size=parallel.tp.size,
            )
            torch.testing.assert_close(
                target_parameter.grad,
                expected_gradient,
                atol=8.0e-5,
                rtol=8.0e-4,
            )

        reference_optimizer.step()
        strategy.optimizer_step()
        for target_parameter, reference_parameter, shard_dim in mappings:
            expected_parameter = _expected_shard(
                reference_parameter,
                shard_dim,
                tp_rank=parallel.tp.rank,
                tp_size=parallel.tp.size,
            )
            torch.testing.assert_close(
                target_parameter,
                expected_parameter,
                atol=8.0e-5,
                rtol=8.0e-4,
            )
    finally:
        parallel.close()
        runtime.close()


@pytest.mark.distributed
def test_tp2_cp2_sp_ring_gpt_step_matches_full_sequence_eager(tmp_path: Path) -> None:
    rendezvous = tmp_path / "gpt-tp2-cp2-sp-ring.rendezvous"
    mp.spawn(
        _worker,
        args=(str(rendezvous), str(tmp_path / "checkpoints")),
        nprocs=4,
        join=True,
    )
