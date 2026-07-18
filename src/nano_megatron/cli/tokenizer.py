"""Train, inspect, and apply Nano-Megatron byte-level BPE tokenizers."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from nano_megatron.data import iter_jsonl_text, preprocess_jsonl
from nano_megatron.tokenizer import (
    ByteBPETrainingConfig,
    ByteLevelBPETokenizer,
    SpecialTokens,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train a byte-level BPE artifact from JSONL")
    train.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="input JSONL path; repeat to train on multiple files",
    )
    train.add_argument("--output", type=Path, required=True, help="artifact directory")
    train.add_argument("--vocab-size", type=int, default=8192)
    train.add_argument("--min-frequency", type=int, default=2)
    train.add_argument("--text-key", default="text")
    train.add_argument("--unk-token", default=SpecialTokens().unk)
    train.add_argument("--bos-token", default=SpecialTokens().bos)
    train.add_argument("--eos-token", default=SpecialTokens().eos)
    train.add_argument("--pad-token", default=SpecialTokens().pad)
    train.add_argument("--show-progress", action="store_true")

    preprocess = commands.add_parser(
        "preprocess", help="encode JSONL into a structured flat-token corpus"
    )
    preprocess.add_argument("--input", type=Path, required=True)
    preprocess.add_argument("--output", type=Path, required=True)
    preprocess.add_argument("--tokenizer", type=Path, required=True)
    preprocess.add_argument(
        "--format",
        choices=("pt", "mmap"),
        default="pt",
        help="output storage format (default: pt)",
    )
    preprocess.add_argument("--text-key", default="text")
    preprocess.add_argument(
        "--append-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="append exactly one EOS token after each document",
    )

    split = commands.add_parser(
        "split",
        help="deterministically split JSONL documents into train and validation files",
    )
    split.add_argument("--input", type=Path, required=True)
    split.add_argument("--train-output", type=Path, required=True)
    split.add_argument("--validation-output", type=Path, required=True)
    split.add_argument("--validation-fraction", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=1234)
    split.add_argument("--text-key", default="text")

    inspect = commands.add_parser(
        "inspect", help="inspect a tokenizer encoding or mmap corpus metadata"
    )
    inspect_target = inspect.add_mutually_exclusive_group(required=True)
    inspect_target.add_argument("--tokenizer", type=Path)
    inspect_target.add_argument("--mmap", type=Path, help="mmap token-corpus artifact directory")
    inspect.add_argument("--text")
    inspect.add_argument("--add-bos", action="store_true")
    inspect.add_argument("--add-eos", action="store_true")
    return parser


def _iter_inputs(paths: Sequence[Path], *, text_key: str) -> Iterator[str]:
    for path in paths:
        yield from iter_jsonl_text(path, text_key=text_key)


class _CountingTexts(Iterable[str]):
    def __init__(self, texts: Iterable[str]) -> None:
        self._texts = texts
        self.count = 0

    def __iter__(self) -> Iterator[str]:
        for text in self._texts:
            self.count += 1
            yield text


def run_train(
    inputs: Sequence[Path],
    output: Path,
    *,
    vocab_size: int = 8192,
    min_frequency: int = 2,
    text_key: str = "text",
    special_tokens: SpecialTokens | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Train and save one deterministic byte-level BPE artifact."""

    if not inputs:
        raise ValueError("tokenizer training requires at least one input JSONL path")
    ByteLevelBPETokenizer.validate_save_target(output)
    texts = _CountingTexts(_iter_inputs(tuple(inputs), text_key=text_key))
    config = ByteBPETrainingConfig(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens or SpecialTokens(),
        show_progress=show_progress,
    )
    tokenizer = ByteLevelBPETokenizer.train(texts, config)
    if texts.count < 1:
        raise ValueError("tokenizer training corpus contains no documents")
    tokenizer.save(output)
    return {
        "command": "train",
        "documents": texts.count,
        "eos_token_id": tokenizer.eos_token_id,
        "fingerprint": tokenizer.fingerprint,
        "output": str(output),
        "vocab_size": tokenizer.vocab_size,
    }


def run_preprocess(
    input_path: Path,
    output_path: Path,
    tokenizer_path: Path,
    *,
    text_key: str = "text",
    append_eos: bool = True,
    output_format: str = "pt",
) -> dict[str, Any]:
    """Encode JSONL into a PT or mmap token-corpus artifact."""

    tokenizer = ByteLevelBPETokenizer.load(tokenizer_path)
    if output_format == "pt":
        corpus = preprocess_jsonl(
            input_path,
            output_path,
            tokenizer,
            text_key=text_key,
            append_eos=append_eos,
        )
        return {
            "append_eos": corpus.append_eos,
            "command": "preprocess",
            "corpus_fingerprint": corpus.fingerprint,
            "documents": corpus.documents,
            "eos_token_id": corpus.eos_id,
            "output": str(output_path),
            "tokenizer_fingerprint": corpus.tokenizer_fingerprint,
            "tokens": int(corpus.tokens.numel()),
            "vocab_size": corpus.vocab_size,
        }
    if output_format != "mmap":
        raise ValueError(f"output_format must be 'pt' or 'mmap', got {output_format!r}")

    from nano_megatron.data.mmap_corpus import preprocess_jsonl_mmap

    mmap_corpus = preprocess_jsonl_mmap(
        input_path,
        output_path,
        tokenizer,
        text_key=text_key,
        append_eos=append_eos,
    )
    return {
        "append_eos": mmap_corpus.append_eos,
        "command": "preprocess",
        "corpus_fingerprint": mmap_corpus.fingerprint,
        "documents": mmap_corpus.documents,
        "eos_token_id": mmap_corpus.eos_id,
        "format": "mmap",
        "output": str(output_path),
        "token_dtype": str(mmap_corpus.token_dtype),
        "tokenizer_fingerprint": mmap_corpus.tokenizer_fingerprint,
        "tokens": mmap_corpus.token_count,
        "vocab_size": mmap_corpus.vocab_size,
    }


def run_split(
    input_path: Path,
    train_output: Path,
    validation_output: Path,
    *,
    validation_fraction: float = 0.1,
    seed: int = 1234,
    text_key: str = "text",
) -> dict[str, Any]:
    """Stream a stable, order-independent document split into two JSONL files."""

    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (int, float))
        or not 0.0 < validation_fraction < 1.0
    ):
        raise ValueError("validation_fraction must be between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("split seed must be a non-negative integer")
    source = input_path.resolve()
    train_target = train_output.resolve()
    validation_target = validation_output.resolve()
    if len({source, train_target, validation_target}) != 3:
        raise ValueError("input, train output, and validation output must be distinct paths")
    for target in (train_target, validation_target):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing split output: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)

    suffix = uuid.uuid4().hex
    train_temporary = train_target.with_name(f".{train_target.name}.{suffix}.tmp")
    validation_temporary = validation_target.with_name(
        f".{validation_target.name}.{suffix}.tmp"
    )
    threshold = int(float(validation_fraction) * (1 << 64))
    train_documents = 0
    validation_documents = 0
    train_published = False
    try:
        with (
            train_temporary.open("x", encoding="utf-8", newline="\n") as train_stream,
            validation_temporary.open(
                "x",
                encoding="utf-8",
                newline="\n",
            ) as validation_stream,
        ):
            for text in iter_jsonl_text(source, text_key=text_key):
                digest = hashlib.sha256(
                    f"nano-megatron-jsonl-split-v1\0{seed}\0{text}".encode()
                ).digest()
                is_validation = int.from_bytes(digest[:8], "big") < threshold
                stream = validation_stream if is_validation else train_stream
                stream.write(
                    json.dumps(
                        {text_key: text},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                if is_validation:
                    validation_documents += 1
                else:
                    train_documents += 1
        if train_documents == 0 or validation_documents == 0:
            raise ValueError(
                "the deterministic split produced an empty partition; use more documents "
                "or a different validation fraction/seed"
            )
        train_temporary.replace(train_target)
        train_published = True
        validation_temporary.replace(validation_target)
    except BaseException:
        train_temporary.unlink(missing_ok=True)
        validation_temporary.unlink(missing_ok=True)
        if train_published:
            train_target.unlink(missing_ok=True)
        raise

    return {
        "command": "split",
        "input": str(source),
        "seed": seed,
        "train_documents": train_documents,
        "train_output": str(train_target),
        "validation_documents": validation_documents,
        "validation_fraction": float(validation_fraction),
        "validation_output": str(validation_target),
    }


def run_inspect(
    tokenizer_path: Path,
    text: str,
    *,
    add_bos: bool = False,
    add_eos: bool = False,
) -> dict[str, Any]:
    """Return a stable JSON-compatible encoding report for one string."""

    tokenizer = ByteLevelBPETokenizer.load(tokenizer_path)
    ids = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
    return {
        "bos_token_id": tokenizer.bos_token_id,
        "decoded": tokenizer.decode(ids),
        "eos_token_id": tokenizer.eos_token_id,
        "fingerprint": tokenizer.fingerprint,
        "ids": ids,
        "pad_token_id": tokenizer.pad_token_id,
        "tokens": [tokenizer.id_to_token(token_id) for token_id in ids],
        "unk_token_id": tokenizer.unk_token_id,
        "vocab_size": tokenizer.vocab_size,
    }


def run_inspect_mmap(mmap_path: Path) -> dict[str, Any]:
    """Return stable JSON-compatible metadata for an mmap token corpus."""

    from nano_megatron.data.mmap_corpus import MMapTokenCorpus

    corpus = MMapTokenCorpus.load(mmap_path)
    return {
        "append_eos": corpus.append_eos,
        "corpus_fingerprint": corpus.fingerprint,
        "documents": corpus.documents,
        "eos_token_id": corpus.eos_id,
        "format": "mmap",
        "path": str(corpus.path),
        "token_dtype": str(corpus.token_dtype),
        "tokenizer_fingerprint": corpus.tokenizer_fingerprint,
        "tokens": corpus.token_count,
        "vocab_size": corpus.vocab_size,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        result = run_train(
            args.input,
            args.output,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            text_key=args.text_key,
            special_tokens=SpecialTokens(
                unk=args.unk_token,
                bos=args.bos_token,
                eos=args.eos_token,
                pad=args.pad_token,
            ),
            show_progress=args.show_progress,
        )
    elif args.command == "preprocess":
        result = run_preprocess(
            args.input,
            args.output,
            args.tokenizer,
            text_key=args.text_key,
            append_eos=args.append_eos,
            output_format=args.format,
        )
    elif args.command == "split":
        result = run_split(
            args.input,
            args.train_output,
            args.validation_output,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            text_key=args.text_key,
        )
    elif args.command == "inspect":
        if args.mmap is not None:
            if args.text is not None or args.add_bos or args.add_eos:
                parser.error("inspect --mmap cannot be combined with text, BOS, or EOS options")
            result = run_inspect_mmap(args.mmap)
        else:
            if args.text is None:
                parser.error("inspect --tokenizer requires --text")
            result = run_inspect(
                args.tokenizer,
                args.text,
                add_bos=args.add_bos,
                add_eos=args.add_eos,
            )
    else:  # pragma: no cover - argparse guarantees a known subcommand.
        raise AssertionError(f"unknown tokenizer command {args.command!r}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
