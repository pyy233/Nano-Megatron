"""Train a first-phase GPT model from one YAML configuration."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import torch

from nano_megatron.checkpoint import CheckpointManager
from nano_megatron.config import load_config, validate_config
from nano_megatron.context_parallel import build_context_parallel_attention
from nano_megatron.data import build_train_dataloader
from nano_megatron.data_parallel import build_data_parallel_strategy
from nano_megatron.distributed import DistributedRuntime
from nano_megatron.models.gpt import DenseGPTComponents, GPTModelBuilder
from nano_megatron.nn.kernels import build_kernel_backend
from nano_megatron.parallel import ParallelContext, ParallelRNG, ParameterDomainRegistry, RNGStream
from nano_megatron.tokenizer import ByteLevelBPETokenizer, validate_tokenizer_for_model
from nano_megatron.training import Trainer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML training configuration")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a dotted configuration key; may be repeated",
    )
    parser.add_argument("--resume", type=Path, help="checkpoint directory to restore")
    parser.add_argument("--max-steps", type=int, help="override only this invocation's stop step")
    return parser


def _dtype(value) -> torch.dtype:
    name = str(getattr(value, "value", value))
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def run(config_path: Path, *, overrides: Sequence[str], resume: Path | None, max_steps: int | None):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    config = load_config(config_path, overrides=overrides, world_size=world_size)
    validate_config(config, world_size=world_size)
    tokenizer = None
    if config.data.tokenizer is not None:
        tokenizer = ByteLevelBPETokenizer.load(config.data.tokenizer.path)
        validate_tokenizer_for_model(tokenizer, config.model)

    with (
        DistributedRuntime(config.distributed) as runtime,
        ParallelContext.create(runtime, config.parallel) as parallel,
    ):
        rng = ParallelRNG.from_context(config.training.seed, parallel)
        torch.manual_seed(rng.seed(RNGStream.DENSE_INIT))
        kernels = build_kernel_backend(config.kernels, parallel)
        cp_attention = build_context_parallel_attention(config.context_parallel, parallel)
        domains = ParameterDomainRegistry()
        components = DenseGPTComponents(cp_attention=cp_attention)
        builder = GPTModelBuilder(
            components,
            parameter_domains=domains,
            activation_checkpoint_config=config.activation_checkpoint,
        )
        if config.pipeline.virtual_stages_per_rank > 1:
            built = builder.build_pipeline(
                config.model,
                parallel,
                kernels,
                virtual_stages_per_rank=config.pipeline.virtual_stages_per_rank,
                rng=rng,
            )
        else:
            built = builder.build_stage(config.model, parallel, kernels, rng=rng)
        model = built.model.to(device=runtime.device, dtype=_dtype(config.precision.params))

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
        trainer = Trainer(
            config=config,
            model=model,
            parallel=parallel,
            data_parallel=strategy,
            checkpoint=checkpoint,
            data_iterator=None,
            rng=rng,
        )
        try:
            if resume is not None:
                trainer.load_checkpoint(resume)
            data = build_train_dataloader(
                config,
                parallel,
                tokenizer=tokenizer,
                max_steps=max_steps,
                start_step=trainer.state.step,
                consumed_samples=trainer.state.consumed_samples,
            )
            data_fingerprint = (
                getattr(data, "data_fingerprint", None) if trainer.batch_router.is_source else None
            )
            if trainer.batch_router.is_source and not isinstance(data_fingerprint, str):
                raise TypeError("source DataLoader must expose data_fingerprint")
            if trainer.batch_router.is_source:
                trainer.bind_data_iterator(
                    data,
                    data_fingerprint=data_fingerprint,
                )
            else:
                trainer.data_iterator = iter(data)
            if resume is not None:
                trainer.restore_data_position()
            state = trainer.fit(max_steps=max_steps)
        finally:
            trainer.close()
        if runtime.is_primary:
            print(
                f"finished step={state.step} samples={state.consumed_samples} "
                f"tokens={state.consumed_tokens}"
            )
        return state


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    run(
        args.config,
        overrides=args.overrides,
        resume=args.resume,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
