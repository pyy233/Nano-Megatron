from __future__ import annotations

from pathlib import Path

from nano_megatron.cli.train import run


def test_cli_resume_restores_state_and_continues_to_absolute_stop_step(
    tmp_path: Path,
) -> None:
    config_path = Path(__file__).parents[2] / "examples" / "configs" / "gpt_single_cpu.yaml"
    checkpoint_directory = tmp_path / "checkpoints"
    overrides = [
        "model.layers=1",
        "model.hidden_size=16",
        "model.ffn_hidden_size=32",
        "model.heads=4",
        "model.kv_heads=2",
        "model.seq_length=4",
        "model.vocab_size=32",
        "model.dropout=0.2",
        "training.micro_batch_size=1",
        "training.gradient_accumulation_steps=1",
        "training.max_steps=2",
        "data_parallel.mode=zero2",
        "activation_checkpoint.mode=full",
        "activation_checkpoint.block_interval=1",
        "activation_checkpoint.offload_saved_tensors=true",
        "offload.optimizer_state=true",
        "offload.activations=true",
        "checkpoint.save_interval=1",
        f"checkpoint.directory={checkpoint_directory}",
    ]

    first = run(
        config_path,
        overrides=overrides,
        resume=None,
        max_steps=1,
    )
    checkpoint = checkpoint_directory / "step_00000001"
    assert first.step == 1
    assert checkpoint.is_dir()

    resumed = run(
        config_path,
        overrides=overrides,
        resume=checkpoint,
        max_steps=2,
    )
    assert resumed.step == 2
    assert resumed.consumed_samples == 2
    assert resumed.consumed_tokens == 8
