"""Export a Nano-Megatron training checkpoint for single-device inference."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from nano_megatron.inference import export_checkpoint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="completed step directory")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new inference artifact directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="training YAML override for checkpoints without a usable embedded run_config",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="tokenizer artifact override (otherwise read from the checkpoint config)",
    )
    return parser


def run_export(
    checkpoint: Path,
    output: Path,
    *,
    config: Path | None = None,
    tokenizer: Path | None = None,
) -> dict[str, Any]:
    artifact = export_checkpoint(
        checkpoint,
        output,
        config_path=config,
        tokenizer_path=tokenizer,
    )
    source = artifact.manifest["source"]
    weights = artifact.manifest["weights"]
    return {
        "artifact": artifact.manifest["artifact"],
        "command": "export",
        "format_version": artifact.manifest["format_version"],
        "output": str(output),
        "source_checkpoint": source["checkpoint"],
        "source_step": source["step"],
        "tokenizer_fingerprint": artifact.tokenizer.fingerprint,
        "vocab_size": artifact.model_config.vocab_size,
        "weight_dtype": weights["dtype"],
        "weights_sha256": weights["sha256"],
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = run_export(
        args.checkpoint,
        args.output,
        config=args.config,
        tokenizer=args.tokenizer,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
