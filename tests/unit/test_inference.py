from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from nano_megatron.checkpoint import CheckpointManifest, ShardedState
from nano_megatron.checkpoint.manifest import make_manifest
from nano_megatron.cli.export import main as export_main
from nano_megatron.cli.generate import main as generate_main
from nano_megatron.config import (
    DataConfig,
    DistributedConfig,
    GPTConfig,
    ParallelConfig,
    PrecisionConfig,
    TokenizerConfig,
    TrainConfig,
)
from nano_megatron.distributed import DistributedRuntime
from nano_megatron.inference import (
    InferenceArtifactError,
    export_checkpoint,
    generate_from_artifact,
    generate_text,
    load_inference_artifact,
)
from nano_megatron.models.gpt import GPTModelBuilder, GPTOutput
from nano_megatron.nn.kernels import TorchKernelBackend
from nano_megatron.parallel import ParallelContext, ParameterDomainRegistry
from nano_megatron.tokenizer import (
    ByteBPETrainingConfig,
    ByteLevelBPETokenizer,
)


def _tokenizer(path: Path) -> ByteLevelBPETokenizer:
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


def _single_process_checkpoint(root: Path) -> tuple[Path, TrainConfig, dict[str, torch.Tensor]]:
    tokenizer_path = root / "source-tokenizer"
    _tokenizer(tokenizer_path)
    config = TrainConfig(
        distributed=DistributedConfig(backend="gloo", device="cpu"),
        parallel=ParallelConfig(data=1),
        model=GPTConfig(
            layers=2,
            hidden_size=16,
            ffn_hidden_size=32,
            heads=4,
            kv_heads=2,
            seq_length=16,
            vocab_size=300,
            dropout=0.0,
        ),
        precision=PrecisionConfig(
            params="float32",
            compute="float32",
            grad_reduce="float32",
        ),
        data=DataConfig(
            path=root / "unused.tokens.pt",
            tokenizer=TokenizerConfig(path=tokenizer_path),
        ),
    )
    checkpoint = root / "checkpoint" / "step_00000007"
    checkpoint.mkdir(parents=True)
    with (
        DistributedRuntime(config.distributed) as runtime,
        ParallelContext.create(runtime, config.parallel) as parallel,
    ):
        registry = ParameterDomainRegistry()
        built = GPTModelBuilder(parameter_domains=registry).build_stage(
            config.model,
            parallel,
            TorchKernelBackend(),
        )
        with torch.no_grad():
            for parameter in built.model.parameters():
                parameter.zero_()
        sharded = ShardedState.from_model(
            built.model,
            parameter_domains=registry,
            parallel=parallel,
        )
        source_state = {
            key.removeprefix("model."): tensor.detach().clone()
            for key, tensor in sharded.state.items()
        }
        manifest = make_manifest(
            step=7,
            storage_backend="torch.save",
            parallel=parallel,
            metadata=sharded.metadata,
            data_parallel_mode="ddp",
            run_config=config,
        )
        torch.save(sharded.state, checkpoint / "model.pt")
        manifest.write(checkpoint / "manifest.json")
        (checkpoint / ".complete").write_text("complete\n", encoding="utf-8")
    return checkpoint, config, source_state


def test_single_process_checkpoint_exports_and_generates_on_cpu(tmp_path: Path) -> None:
    checkpoint, config, source_state = _single_process_checkpoint(tmp_path)
    output = tmp_path / "exported"

    artifact = export_checkpoint(checkpoint, output)

    assert artifact.model_config == config.model
    assert artifact.manifest["source"]["step"] == 7
    assert artifact.manifest["source"]["storage_backend"] == "torch.save"
    assert artifact.manifest["weights"]["dtype"] == "float32"
    assert (output / "model.pt").is_file()
    assert (output / "tokenizer" / "tokenizer.json").is_file()
    assert (output / ".complete").read_text(encoding="utf-8") == "complete\n"
    assert artifact.state_dict.keys() == source_state.keys()
    for key, expected in source_state.items():
        torch.testing.assert_close(artifact.state_dict[key], expected)

    result = generate_from_artifact(
        output,
        "Once upon a time",
        device="cpu",
        dtype="float32",
        max_new_tokens=3,
        temperature=0.0,
    )
    assert result.generated_token_ids == (0, 0, 0)
    assert result.stop_reason == "length"
    assert len(result.token_ids) == len(
        artifact.tokenizer.encode("Once upon a time", add_bos=True)
    ) + 3


def test_export_and_generate_cli_emit_stable_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint, _, _ = _single_process_checkpoint(tmp_path)
    output = tmp_path / "cli-export"
    export_main(["--checkpoint", str(checkpoint), "--output", str(output)])
    exported_text = capsys.readouterr().out
    exported = json.loads(exported_text)
    assert exported_text == json.dumps(exported, ensure_ascii=False, sort_keys=True) + "\n"
    assert exported["command"] == "export"
    assert exported["source_step"] == 7
    assert exported["output"] == str(output)

    generate_main(
        [
            "--model",
            str(output),
            "--prompt",
            "A fox",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--max-new-tokens",
            "2",
            "--temperature",
            "0",
            "--json",
        ]
    )
    generated_text = capsys.readouterr().out
    generated = json.loads(generated_text)
    assert generated_text == json.dumps(generated, ensure_ascii=False, sort_keys=True) + "\n"
    assert generated["generated_token_ids"] == [0, 0]
    assert generated["stop_reason"] == "length"


def test_inference_artifact_rejects_tampered_weights(tmp_path: Path) -> None:
    checkpoint, _, _ = _single_process_checkpoint(tmp_path)
    output = tmp_path / "exported"
    export_checkpoint(checkpoint, output)
    with (output / "model.pt").open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(InferenceArtifactError, match="SHA256"):
        load_inference_artifact(output)


def test_generation_rejects_context_overflow_and_export_collision(tmp_path: Path) -> None:
    checkpoint, _, _ = _single_process_checkpoint(tmp_path)
    output = tmp_path / "exported"
    artifact = export_checkpoint(checkpoint, output)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_checkpoint(checkpoint, output)
    with pytest.raises(ValueError, match="exceeds model sequence length"):
        generate_from_artifact(
            output,
            "A prompt that already consumes several tokenizer tokens",
            device="cpu",
            max_new_tokens=artifact.model_config.seq_length,
        )


@pytest.mark.parametrize("mode", ["zero1", "zero2"])
def test_export_accepts_optimizer_sharded_checkpoint_modes(
    tmp_path: Path,
    mode: str,
) -> None:
    checkpoint, _, _ = _single_process_checkpoint(tmp_path)
    manifest_path = checkpoint / "manifest.json"
    manifest = CheckpointManifest.read(manifest_path)
    replace(manifest, data_parallel_mode=mode).write(manifest_path)

    artifact = export_checkpoint(checkpoint, tmp_path / f"exported-{mode}")

    assert artifact.manifest["source"]["data_parallel_mode"] == mode


@pytest.mark.parametrize(
    ("storage_backend", "data_parallel_mode", "message"),
    [
        ("fsdp2_dcp", "zero3", "ZeRO-3 DCP"),
        ("unknown_backend", "ddp", "cannot be exported"),
    ],
)
def test_export_rejects_unsupported_checkpoint_layouts(
    tmp_path: Path,
    storage_backend: str,
    data_parallel_mode: str,
    message: str,
) -> None:
    checkpoint, _, _ = _single_process_checkpoint(tmp_path)
    manifest_path = checkpoint / "manifest.json"
    manifest = CheckpointManifest.read(manifest_path)
    replace(
        manifest,
        storage_backend=storage_backend,
        data_parallel_mode=data_parallel_mode,
    ).write(manifest_path)

    with pytest.raises(InferenceArtifactError, match=message):
        export_checkpoint(checkpoint, tmp_path / "unsupported")


def test_sampling_is_reproducible_and_generation_stops_at_eos(tmp_path: Path) -> None:
    checkpoint, _, _ = _single_process_checkpoint(tmp_path)
    output = tmp_path / "exported"
    artifact = export_checkpoint(checkpoint, output)

    first = generate_from_artifact(
        output,
        "A fox",
        device="cpu",
        dtype="float32",
        max_new_tokens=4,
        temperature=0.8,
        top_p=0.75,
        seed=2027,
    )
    second = generate_from_artifact(
        output,
        "A fox",
        device="cpu",
        dtype="float32",
        max_new_tokens=4,
        temperature=0.8,
        top_p=0.75,
        seed=2027,
    )
    assert first.token_ids == second.token_ids

    class _EosModel(torch.nn.Module):
        def forward(self, input_ids: torch.Tensor) -> GPTOutput:
            batch, sequence = input_ids.shape
            logits = torch.full(
                (batch, sequence, artifact.tokenizer.vocab_size),
                -1.0,
            )
            logits[..., artifact.tokenizer.eos_token_id] = 1.0
            hidden = torch.zeros(batch, sequence, artifact.model_config.hidden_size)
            return GPTOutput(logits=logits, loss=None, hidden_states=hidden)

    eos_result = generate_text(
        artifact,
        _EosModel(),  # type: ignore[arg-type]
        "A fox",
        device=torch.device("cpu"),
        max_new_tokens=4,
        temperature=0.0,
    )
    assert eos_result.generated_token_ids == (artifact.tokenizer.eos_token_id,)
    assert eos_result.stop_reason == "eos"
