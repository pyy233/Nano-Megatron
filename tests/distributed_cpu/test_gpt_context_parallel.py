from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

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
from nano_megatron.nn.kernels import TorchKernelBackend
from nano_megatron.parallel import ParallelContext, ParameterDomainRegistry
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


def _config(checkpoint_directory: Path, backend: str) -> TrainConfig:
    return TrainConfig(
        distributed=DistributedConfig(backend="gloo", device="cpu"),
        parallel=ParallelConfig(context=2, data=1),
        model=GPTConfig(
            layers=1,
            hidden_size=8,
            ffn_hidden_size=16,
            heads=2,
            kv_heads=1,
            seq_length=6,
            vocab_size=16,
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
        data_parallel=DataParallelConfig(mode="ddp"),
        pipeline=PipelineConfig(schedule="gpipe"),
        context_parallel=ContextParallelConfig(backend=backend),
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


def _worker(
    rank: int,
    rendezvous: str,
    checkpoint_directory: str,
    backend: str,
) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank))
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    config = _config(Path(checkpoint_directory), backend)
    runtime = DistributedRuntime(
        DistributedConfig(
            backend="gloo",
            device="cpu",
            init_method=f"file://{rendezvous}",
        )
    ).initialize()
    parallel = ParallelContext.create(runtime, config.parallel)
    try:
        torch.manual_seed(73)
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
        reference = GPTModel(
            config.model,
            parallel=_SingleParallel(),
            kernels=TorchKernelBackend(),
        )
        reference.load_state_dict(target_core.state_dict())

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
                [[0, 2, 5, 9, 3, 7], [4, 1, 8, 6, 10, 12]],
            ),
            "labels": torch.tensor(
                [[2, 5, 9, 3, 7, 11], [1, 8, 6, 10, 12, 15]],
            ),
        }

        reference_optimizer = torch.optim.AdamW(
            reference.parameters(),
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        )
        with torch.no_grad():
            routed = trainer.batch_router.route(
                batch if trainer.batch_router.is_source else None
            )
            target_output = target_core(
                routed["input_ids"][:1],
                labels=routed["labels"][:1],
            )
            full_output = reference(
                batch["input_ids"][:1],
                labels=batch["labels"][:1],
            )
            assert target_output.logits is not None and full_output.logits is not None
            sequence_chunk = full_output.logits.chunk(2, dim=1)[parallel.cp.rank]
            torch.testing.assert_close(
                target_output.logits,
                sequence_chunk,
                atol=2.0e-5,
                rtol=2.0e-4,
            )
        for microbatch in range(2):
            output = reference(
                batch["input_ids"][microbatch : microbatch + 1],
                labels=batch["labels"][microbatch : microbatch + 1],
            )
            assert output.loss is not None
            (output.loss / 2).backward()
        reference_optimizer.step()

        strategy.zero_grad()
        trainer.schedule.forward_backward(
            stage=trainer.model,
            microbatches=split_microbatches(routed, 1),
            data_parallel=strategy,
        )
        strategy.finalize_gradients()
        wrapped = trainer.model
        stage = getattr(wrapped, "module", wrapped)
        actual_parameters = dict(stage.model.named_parameters())
        expected_parameters = dict(reference.named_parameters())
        assert actual_parameters.keys() == expected_parameters.keys()
        for name, actual in actual_parameters.items():
            assert actual.grad is not None
            assert expected_parameters[name].grad is not None
            torch.testing.assert_close(
                actual.grad,
                expected_parameters[name].grad,
                atol=3.0e-5,
                rtol=3.0e-4,
                msg=lambda message, parameter_name=name: f"{parameter_name}: {message}",
            )
        strategy.optimizer_step()
        actual_state = stage.model.state_dict()
        expected_state = reference.state_dict()
        assert actual_state.keys() == expected_state.keys()
        for name, actual in actual_state.items():
            torch.testing.assert_close(
                actual,
                expected_state[name],
                atol=3.0e-5,
                rtol=3.0e-4,
                msg=lambda message, parameter_name=name: f"{parameter_name}: {message}",
            )
    finally:
        parallel.close()
        runtime.close()


@pytest.mark.distributed
@pytest.mark.parametrize("backend", ["all_gather", "ring"])
def test_cp2_gpt_ddp_step_matches_full_sequence_eager(
    tmp_path: Path,
    backend: str,
) -> None:
    rendezvous = tmp_path / f"gpt-cp2-{backend}.rendezvous"
    mp.spawn(
        _worker,
        args=(str(rendezvous), str(tmp_path / "checkpoints"), backend),
        nprocs=2,
        join=True,
    )
