"""Generate text on one device from an exported Nano-Megatron GPT artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from nano_megatron.inference import GenerationResult, generate_from_artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="inference artifact directory")
    parser.add_argument("--prompt", required=True, help="text prompt")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "bfloat16", "float16"),
        default="auto",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--add-bos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="prepend the tokenizer BOS token (default: true)",
    )
    parser.add_argument("--json", action="store_true", help="emit structured generation output")
    return parser


def run_generate(
    model: Path,
    prompt: str,
    *,
    device: str = "auto",
    dtype: str = "auto",
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 0,
    seed: int = 1234,
    add_bos: bool = True,
) -> GenerationResult:
    return generate_from_artifact(
        model,
        prompt,
        device=device,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        add_bos=add_bos,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = run_generate(
        args.model,
        args.prompt,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        add_bos=args.add_bos,
    )
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True))
    else:
        print(result.text)


if __name__ == "__main__":
    main()
