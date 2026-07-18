#!/usr/bin/env python3
"""Build verified full TinyStories train/validation mmap artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nano_megatron.data import prepare_tinystories_corpus


class _Progress:
    def __init__(self) -> None:
        self._last_percent = -1

    def __call__(self, completed: int, total: int) -> None:
        percent = min(100, int(completed * 100 / max(1, total)))
        if percent == self._last_percent and completed != total:
            return
        self._last_percent = percent
        print(f"progress {completed}/{total} ({percent}%)", file=sys.stderr)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/tinystories"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/tokenizers/tinystories-8k-500k"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/tinystories-full-8k"),
    )
    parser.add_argument("--train-url", action="append", default=None)
    parser.add_argument("--validation-url", action="append", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    kwargs = {}
    if args.train_url:
        kwargs["train_urls"] = tuple(args.train_url)
    if args.validation_url:
        kwargs["validation_urls"] = tuple(args.validation_url)
    result = prepare_tinystories_corpus(
        args.data_dir,
        args.tokenizer,
        args.output_dir,
        progress=None if args.quiet else _Progress(),
        **kwargs,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
