#!/usr/bin/env python3
"""Download pinned TinyStories train data and build the exact 500k JSONL sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nano_megatron.data.tinystories import (
    TINYSTORIES_TRAIN_URLS,
    prepare_tinystories_500k,
)


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
        help="destination directory (default: data/raw/tinystories)",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=None,
        help="download mirror URL; repeat to define fallback order",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = prepare_tinystories_500k(
        args.data_dir,
        urls=tuple(args.url) if args.url else TINYSTORIES_TRAIN_URLS,
        progress=None if args.quiet else _Progress(),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
