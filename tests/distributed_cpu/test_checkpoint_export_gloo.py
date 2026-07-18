from __future__ import annotations

import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nano_megatron.checkpoint import CheckpointManager, CheckpointManifest
from nano_megatron.config import (
    CheckpointConfig,
    DataConfig,
    DataParallelConfig,
    DistributedConfig,
    GPTConfig,
    OffloadConfig,
    OptimizerConfig,
    ParallelConfig,
    PrecisionConfig,
    TokenizerConfig,
    TrainConfig,
)
from nano_megatron.data_parallel import build_data_parallel_strategy
from nano_megatron.distributed import DistributedRuntime
from nano_megatron.inference import (
    InferenceArtifactError,
    export_checkpoint,
    generate_from_artifact,
    load_model_for_generation,
)
from nano_megatron.models.gpt import GPTModelBuilder
from nano_megatron.nn.kernels import TorchKernelBackend
from nano_megatron.parallel import ParallelContext, ParameterDomainRegistry
from nano_megatron.tokenizer import ByteBPETrainingConfig, ByteLevelBPETokenizer


def _train_tokenizer(path: Path) -> ByteLevelBPETokenizer:
    texts = (
        "Once upon a time, a fox found a lantern and shared it with every friend. "
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz 0123456789",
        "A small blue bird flew over the green hill and sang a gentle song.",
    ) * 32
    tokenizer = ByteLevelBPETokenizer.train(
        texts,
        ByteBPETrainingConfig(vocab_size=300, min_frequency=1),
    )
    assert tokenizer.vocab_size == 300
    tokenizer.save(path)
    return tokenizer


def _config(
    root: Path,
    tokenizer_path: Path,
    *,
    data_parallel_mode: str = "ddp",
) -> TrainConfig:
    return TrainConfig(
        distributed=DistributedConfig(backend="gloo", device="cpu"),
        parallel=ParallelConfig(
            tensor=2,
            pipeline=2,
            context=1,
            expert=1,
            data=2,
            sequence_parallel=False,
        ),
        model=GPTConfig(
            layers=4,
            hidden_size=16,
            ffn_hidden_size=32,
            heads=4,
            kv_heads=2,
            seq_length=8,
            vocab_size=300,
            dropout=0.0,
        ),
        precision=PrecisionConfig(
            params="float32",
            compute="float32",
            grad_reduce="float32",
        ),
        data_parallel=DataParallelConfig(mode=data_parallel_mode, bucket_bytes=4096),
        checkpoint=CheckpointConfig(directory=root / "checkpoints", save_interval=1),
        data=DataConfig(
            path=root / "unused.tokens.pt",
            tokenizer=TokenizerConfig(path=tokenizer_path),
        ),
    )


def _runtime(rank: int, world_size: int, rendezvous: str) -> DistributedRuntime:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    return DistributedRuntime(
        DistributedConfig(
            backend="gloo",
            device="cpu",
            init_method=f"file://{rendezvous}",
        )
    ).initialize()


def _save_and_reference_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    root: str,
    tokenizer_path: str,
    data_parallel_mode: str = "ddp",
) -> None:
    root_path = Path(root)
    config = _config(
        root_path,
        Path(tokenizer_path),
        data_parallel_mode=data_parallel_mode,
    )
    runtime = _runtime(rank, world_size, rendezvous)
    parallel = ParallelContext.create(runtime, config.parallel)
    manager: CheckpointManager | None = None
    try:
        torch.manual_seed(2027)
        registry = ParameterDomainRegistry()
        built = GPTModelBuilder(parameter_domains=registry).build_stage(
            config.model,
            parallel,
            TorchKernelBackend(),
        )
        model = built.model.to(dtype=torch.float32)
        strategy = build_data_parallel_strategy(
            config.data_parallel,
            OffloadConfig(),
            parallel,
            registry,
        )
        strategy.setup(model, OptimizerConfig(lr=0.001), registry)
        model.synchronize_tied_embedding_weights()

        manager = CheckpointManager(
            config=config.checkpoint,
            parallel=parallel,
            run_config=config,
        )
        checkpoint = manager.save(
            3,
            model=model,
            data_parallel=strategy,
            trainer_state={"step": 3},
        )
        assert checkpoint == config.checkpoint.directory / "step_00000003"

        input_ids = torch.tensor([[1, 7, 11, 19, 23, 29, 31, 2]], dtype=torch.long)
        if parallel.pp.rank == 0:
            hidden = model.model(input_ids).hidden_states
            if parallel.tp.rank == 0:
                torch.save(hidden, root_path / f"hidden_dp{parallel.dp.rank}.pt")
        runtime.barrier()
        if parallel.pp.rank == 1:
            hidden = torch.load(
                root_path / f"hidden_dp{parallel.dp.rank}.pt",
                map_location="cpu",
                weights_only=True,
            )
            output = model.model(hidden_states=hidden)
            assert output.logits is not None
            gathered = [torch.empty_like(output.logits) for _ in range(parallel.tp.size)]
            dist.all_gather(
                gathered,
                output.logits,
                group=parallel.tp.process_group,
            )
            if parallel.tp.rank == 0 and parallel.dp.rank == 0:
                full_logits = torch.cat(gathered, dim=-1)[..., : config.model.vocab_size]
                torch.save(full_logits, root_path / "distributed_logits.pt")
        runtime.barrier()
    finally:
        if manager is not None:
            manager.close()
        parallel.close()
        runtime.close()


@pytest.mark.distributed
def test_tp2_pp2_dp2_checkpoint_exports_to_equivalent_single_device_gpt(
    tmp_path: Path,
) -> None:
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer = _train_tokenizer(tokenizer_path)
    world_size = 8
    rendezvous = tmp_path / "export.rendezvous"
    mp.spawn(
        _save_and_reference_worker,
        args=(
            world_size,
            str(rendezvous),
            str(tmp_path),
            str(tokenizer_path),
        ),
        nprocs=world_size,
        join=True,
    )

    checkpoint = tmp_path / "checkpoints" / "step_00000003"
    source_manifest = CheckpointManifest.read(checkpoint / "manifest.json")
    assert source_manifest.parallel_sizes == {
        "tp": 2,
        "pp": 2,
        "cp": 1,
        "ep": 1,
        "dp": 2,
    }
    assert len(source_manifest.rank_local_shards) == 4

    artifact_path = tmp_path / "single-device"
    artifact = export_checkpoint(checkpoint, artifact_path)
    assert artifact.tokenizer.fingerprint == tokenizer.fingerprint
    assert artifact.manifest["source"]["parallel_sizes"] == source_manifest.parallel_sizes
    assert artifact.manifest["source"]["step"] == 3

    _, model, device = load_model_for_generation(
        artifact_path,
        device="cpu",
        dtype="float32",
    )
    input_ids = torch.tensor([[1, 7, 11, 19, 23, 29, 31, 2]], dtype=torch.long)
    with torch.inference_mode():
        actual = model(input_ids.to(device)).logits
    assert actual is not None
    expected = torch.load(
        tmp_path / "distributed_logits.pt",
        map_location="cpu",
        weights_only=True,
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=2.0e-5, rtol=2.0e-5)

    generated = generate_from_artifact(
        artifact_path,
        "A fox",
        device="cpu",
        dtype="float32",
        max_new_tokens=2,
        temperature=0.0,
    )
    assert len(generated.generated_token_ids) >= 1

    missing_checkpoint = tmp_path / "missing-shard"
    shutil.copytree(checkpoint, missing_checkpoint)
    missing_manifest_path = missing_checkpoint / "manifest.json"
    missing_manifest = CheckpointManifest.read(missing_manifest_path)
    replace(
        missing_manifest,
        rank_local_shards=missing_manifest.rank_local_shards[:-1],
    ).write(missing_manifest_path)
    with pytest.raises(InferenceArtifactError, match="dense TP/PP shards"):
        export_checkpoint(missing_checkpoint, tmp_path / "missing-shard-export")

    replicated_checkpoint = tmp_path / "replicated-mismatch"
    shutil.copytree(checkpoint, replicated_checkpoint)
    replicated_manifest = CheckpointManifest.read(
        replicated_checkpoint / "manifest.json"
    )
    replicated_shard = next(
        shard for shard in replicated_manifest.rank_local_shards
        if shard.pp == 0 and shard.tp == 1
    )
    replicated_key = next(
        key
        for key, metadata in replicated_shard.tensor_metadata.items()
        if not metadata.sharded_axes
    )
    replicated_state_path = replicated_checkpoint / replicated_shard.state_file
    replicated_payload = torch.load(
        replicated_state_path,
        map_location="cpu",
        weights_only=True,
    )
    replicated_payload["model"][replicated_key].add_(1.0)
    torch.save(replicated_payload, replicated_state_path)
    with pytest.raises(InferenceArtifactError, match="replicated TP copies differ"):
        export_checkpoint(replicated_checkpoint, tmp_path / "replicated-mismatch-export")

    overlap_checkpoint = tmp_path / "overlapping-shards"
    shutil.copytree(checkpoint, overlap_checkpoint)
    overlap_manifest_path = overlap_checkpoint / "manifest.json"
    overlap_manifest = CheckpointManifest.read(overlap_manifest_path)
    first_shard = next(
        shard for shard in overlap_manifest.rank_local_shards
        if shard.pp == 0 and shard.tp == 0
    )
    second_shard = next(
        shard for shard in overlap_manifest.rank_local_shards
        if shard.pp == 0 and shard.tp == 1
    )
    overlap_key = next(
        key
        for key, metadata in second_shard.tensor_metadata.items()
        if metadata.sharded_axes and key in first_shard.tensor_metadata
    )
    overlapping_metadata = replace(
        second_shard.tensor_metadata[overlap_key],
        global_offset=first_shard.tensor_metadata[overlap_key].global_offset,
    )
    second_metadata = dict(second_shard.tensor_metadata)
    second_metadata[overlap_key] = overlapping_metadata
    replacement_shard = replace(second_shard, tensor_metadata=second_metadata)
    replacement_shards = tuple(
        replacement_shard if shard is second_shard else shard
        for shard in overlap_manifest.rank_local_shards
    )
    replace(overlap_manifest, rank_local_shards=replacement_shards).write(
        overlap_manifest_path
    )
    with pytest.raises(InferenceArtifactError, match="overlapping TP shards"):
        export_checkpoint(overlap_checkpoint, tmp_path / "overlapping-shards-export")


@pytest.mark.distributed
@pytest.mark.parametrize("data_parallel_mode", ["zero1", "zero2"])
def test_optimizer_sharded_tp2_pp2_dp2_checkpoint_exports(
    tmp_path: Path,
    data_parallel_mode: str,
) -> None:
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer = _train_tokenizer(tokenizer_path)
    world_size = 8
    rendezvous = tmp_path / f"{data_parallel_mode}.rendezvous"
    mp.spawn(
        _save_and_reference_worker,
        args=(
            world_size,
            str(rendezvous),
            str(tmp_path),
            str(tokenizer_path),
            data_parallel_mode,
        ),
        nprocs=world_size,
        join=True,
    )

    checkpoint = tmp_path / "checkpoints" / "step_00000003"
    source_manifest = CheckpointManifest.read(checkpoint / "manifest.json")
    assert source_manifest.data_parallel_mode == data_parallel_mode
    assert len(source_manifest.rank_local_shards) == 4

    artifact = export_checkpoint(checkpoint, tmp_path / "single-device")
    assert artifact.tokenizer.fingerprint == tokenizer.fingerprint
    assert artifact.manifest["source"]["data_parallel_mode"] == data_parallel_mode
