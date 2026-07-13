from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from nano_megatron.checkpoint import CheckpointManager
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
from nano_megatron.data import RandomTokenDataset
from nano_megatron.data_parallel import build_data_parallel_strategy
from nano_megatron.distributed import DistributedRuntime
from nano_megatron.models.gpt import DenseGPTComponents, GPTModelBuilder
from nano_megatron.nn.kernels import build_kernel_backend
from nano_megatron.parallel import ParallelContext, ParameterDomainRegistry
from nano_megatron.training import Trainer


def _config(
    directory: Path,
    *,
    param_dtype: PrecisionDType = PrecisionDType.FLOAT32,
) -> TrainConfig:
    return TrainConfig(
        distributed=DistributedConfig(backend="gloo", device="cpu"),
        parallel=ParallelConfig(data=1),
        model=GPTConfig(
            layers=2,
            hidden_size=32,
            ffn_hidden_size=64,
            heads=4,
            seq_length=8,
            vocab_size=64,
            dropout=0.0,
        ),
        precision=PrecisionConfig(
            params=param_dtype,
            compute=param_dtype,
            grad_reduce=PrecisionDType.FLOAT32,
        ),
        kernels=KernelConfig(),
        data_parallel=DataParallelConfig(),
        pipeline=PipelineConfig(schedule="gpipe"),
        context_parallel=ContextParallelConfig(),
        offload=OffloadConfig(),
        optimizer=OptimizerConfig(lr=1.0e-3, weight_decay=0.0),
        training=TrainingConfig(
            micro_batch_size=2,
            gradient_accumulation_steps=2,
            max_steps=1,
        ),
        checkpoint=CheckpointConfig(directory=directory, save_interval=100),
    )


@pytest.mark.parametrize(
    "param_dtype",
    [PrecisionDType.FLOAT32, PrecisionDType.BFLOAT16],
)
def test_single_process_training_step_updates_parameters(
    tmp_path: Path,
    param_dtype: PrecisionDType,
) -> None:
    config = _config(tmp_path, param_dtype=param_dtype)
    with (
        DistributedRuntime(config.distributed) as runtime,
        ParallelContext.create(runtime, config.parallel) as parallel,
    ):
        kernels = build_kernel_backend(config.kernels, parallel)
        components = DenseGPTComponents(
            cp_attention=build_context_parallel_attention(
                config.context_parallel, parallel
            )
        )
        domains = ParameterDomainRegistry()
        built = GPTModelBuilder(components, parameter_domains=domains).build_stage(
            config.model, parallel, kernels
        )
        strategy = build_data_parallel_strategy(
            config.data_parallel,
            config.offload,
            parallel,
            built.parameter_domains,
        )
        checkpoint = CheckpointManager(
            config=config.checkpoint,
            parallel=parallel,
            run_config=config,
        )
        dataset = RandomTokenDataset(
            num_samples=8,
            sequence_length=config.model.seq_length,
            vocab_size=config.model.vocab_size,
        )
        loader = DataLoader(dataset, batch_size=4)
        assert next(built.model.parameters()).dtype is torch.float32
        trainer = Trainer(
            config=config,
            model=built.model,
            parallel=parallel,
            data_parallel=strategy,
            checkpoint=checkpoint,
            data_iterator=iter(loader),
        )
        parameter = next(trainer.model.parameters())
        assert parameter.dtype is getattr(torch, param_dtype.value)
        assert parameter.device == runtime.device
        before = parameter.detach().clone()
        state = trainer.fit(max_steps=1)
        after = next(trainer.model.parameters()).detach()

        assert state.step == 1
        assert state.consumed_samples == 4
        assert not torch.equal(before, after)
