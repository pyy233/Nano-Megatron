"""Train a first-phase GPT model from one YAML configuration."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import torch

from nano_megatron.checkpoint import CheckpointManager
from nano_megatron.config import load_config, validate_config
from nano_megatron.context_parallel import build_context_parallel_attention
from nano_megatron.data import build_train_dataloader, build_validation_dataloader
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


def _checkpoint_run_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")


def _allocate_checkpoint_run_directory(base: Path, runtime: DistributedRuntime) -> Path:
    """Create one timestamped checkpoint directory and publish it from rank zero."""

    local_result: tuple[str | None, str | None] | None = None
    if runtime.is_primary:
        try:
            base.mkdir(parents=True, exist_ok=True)
            timestamp = _checkpoint_run_timestamp()
            candidate = base / timestamp
            suffix = 0
            while True:
                try:
                    candidate.mkdir(parents=False, exist_ok=False)
                    break
                except FileExistsError:
                    suffix += 1
                    candidate = base / f"{timestamp}-{suffix:02d}"
            local_result = (str(candidate), None)
        except BaseException as error:
            local_result = (None, f"{type(error).__name__}: {error}")

    gathered = runtime.all_gather_object(local_result)
    coordinator_result = gathered[0]
    if not isinstance(coordinator_result, tuple) or len(coordinator_result) != 2:
        raise RuntimeError("rank 0 returned an invalid checkpoint directory result")
    directory, error = coordinator_result
    if error is not None:
        raise RuntimeError(f"failed to allocate checkpoint run directory: {error}")
    if not isinstance(directory, str) or not directory:
        raise RuntimeError("rank 0 returned an empty checkpoint run directory")
    return Path(directory)


def run(config_path: Path, *, overrides: Sequence[str], resume: Path | None, max_steps: int | None):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    config = load_config(config_path, overrides=overrides, world_size=world_size)
    validate_config(config, world_size=world_size)
    tokenizer = None
    validation_data = config.validation.data
    tokenizer_config = config.data.tokenizer
    if tokenizer_config is None and validation_data is not None:
        tokenizer_config = validation_data.tokenizer
    if tokenizer_config is not None:
        tokenizer = ByteLevelBPETokenizer.load(tokenizer_config.path)
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
        checkpoint_run_directory = _allocate_checkpoint_run_directory(
            config.checkpoint.directory,
            runtime,
        )
        config = replace(
            config,
            checkpoint=replace(
                config.checkpoint,
                directory=checkpoint_run_directory,
            ),
        )
        if runtime.is_primary:
            print(f"checkpoint run directory: {checkpoint_run_directory}")
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
            metrics_path=checkpoint_run_directory / "metrics.jsonl",
            resume_metrics_path=(
                None if resume is None else resume.parent / "metrics.jsonl"
            ),
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
            if config.validation.interval > 0:
                validation_loader = build_validation_dataloader(
                    config,
                    parallel,
                    tokenizer=tokenizer,
                )
                trainer.bind_validation_data(validation_loader)
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
