#!/usr/bin/env python3
"""Train and verify the 8,192-token BPE from the exact TinyStories 500k sample."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/tinystories/train_500k.jsonl"),
    )
    parser.add_argument(
        "--sample-metadata",
        type=Path,
        default=Path("data/raw/tinystories/train_500k.metadata.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tokenizers/tinystories-8k-500k"),
    )
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--quiet", action="store_true", help="hide BPE trainer progress")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    os.environ.setdefault("RAYON_NUM_THREADS", str(args.threads))

    # Import after setting RAYON_NUM_THREADS so the Rust tokenizer pool sees it
    # before its first parallel operation.
    from nano_megatron.tokenizer.tinystories import train_tinystories_tokenizer

    result = train_tinystories_tokenizer(
        args.input,
        args.sample_metadata,
        args.output,
        show_progress=not args.quiet,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
