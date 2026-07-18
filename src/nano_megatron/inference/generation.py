"""Single-device autoregressive generation from exported GPT artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from nano_megatron.models.gpt import GPTModel
from nano_megatron.nn.kernels import TorchKernelBackend

from .artifact import InferenceArtifact, load_inference_artifact

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass(frozen=True, slots=True)
class _LocalGroup:
    rank: int = 0
    size: int = 1
    process_group: object | None = None


@dataclass(frozen=True, slots=True)
class _LocalParallel:
    tp: _LocalGroup = _LocalGroup()
    pp: _LocalGroup = _LocalGroup()
    cp: _LocalGroup = _LocalGroup()
    sequence_parallel: bool = False


@dataclass(frozen=True, slots=True)
class GenerationResult:
    prompt: str
    text: str
    completion: str
    token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    stop_reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "completion": self.completion,
            "generated_token_ids": list(self.generated_token_ids),
            "prompt": self.prompt,
            "stop_reason": self.stop_reason,
            "text": self.text,
            "token_ids": list(self.token_ids),
        }


def _resolve_device(value: str | torch.device) -> torch.device:
    if isinstance(value, torch.device):
        device = value
    elif value == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            device = torch.device(value)
        except (TypeError, RuntimeError) as error:
            raise ValueError(f"invalid generation device {value!r}: {error}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA generation was requested but CUDA is unavailable")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"generation device must be CPU or CUDA, got {device}")
    return device


def _resolve_dtype(
    value: str,
    *,
    artifact: InferenceArtifact,
    device: torch.device,
) -> torch.dtype:
    if value == "auto":
        return torch.float32 if device.type == "cpu" else artifact.weight_dtype
    try:
        dtype = _DTYPES[value]
    except KeyError as error:
        raise ValueError(f"generation dtype must be auto or one of {sorted(_DTYPES)}") from error
    if device.type == "cpu" and dtype is torch.float16:
        raise ValueError("float16 generation is not supported on CPU; use float32 or bfloat16")
    return dtype


def load_model_for_generation(
    artifact_path: str | Path,
    *,
    device: str | torch.device = "auto",
    dtype: str = "auto",
) -> tuple[InferenceArtifact, GPTModel, torch.device]:
    """Load a validated artifact into one ordinary, non-distributed GPT model."""

    artifact = load_inference_artifact(artifact_path)
    target_device = _resolve_device(device)
    target_dtype = _resolve_dtype(dtype, artifact=artifact, device=target_device)
    model = GPTModel(
        artifact.model_config,
        parallel=_LocalParallel(),  # type: ignore[arg-type]
        kernels=TorchKernelBackend(),
    )
    model.to(dtype=target_dtype)
    model.load_state_dict(artifact.state_dict, strict=True)
    model.to(device=target_device)
    model.eval()
    if artifact.model_config.tie_embeddings:
        if model.embedding is None or model.lm_head is None:
            raise RuntimeError("loaded tied GPT model is missing embedding or LM head")
        if model.embedding.weight is not model.lm_head.weight:
            raise RuntimeError("loaded GPT model did not preserve tied embedding identity")
    return artifact, model, target_device


def _sample_next_token(
    logits: Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: torch.Generator,
) -> int:
    scores = logits.detach().float().cpu()
    if temperature == 0.0:
        return int(scores.argmax().item())
    scores.div_(temperature)
    if top_k:
        count = min(top_k, scores.numel())
        threshold = torch.topk(scores, count).values[-1]
        scores.masked_fill_(scores < threshold, float("-inf"))
    if top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        sorted_probabilities = torch.softmax(sorted_scores, dim=-1)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative - sorted_probabilities > top_p
        sorted_scores.masked_fill_(remove, float("-inf"))
        filtered = torch.full_like(scores, float("-inf"))
        filtered.scatter_(0, sorted_indices, sorted_scores)
        scores = filtered
    probabilities = torch.softmax(scores, dim=-1)
    if not torch.isfinite(probabilities).all() or float(probabilities.sum()) <= 0.0:
        raise RuntimeError("sampling produced an invalid probability distribution")
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def generate_text(
    artifact: InferenceArtifact,
    model: GPTModel,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 0,
    seed: int = 1234,
    add_bos: bool = True,
) -> GenerationResult:
    """Generate one continuation by recomputing the complete prefix each step."""

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an integer")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a number")
    if not torch.isfinite(torch.tensor(float(temperature))) or temperature < 0.0:
        raise ValueError("temperature must be finite and non-negative")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
        raise TypeError("top_p must be a number")
    if not 0.0 < float(top_p) <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(add_bos, bool):
        raise TypeError("add_bos must be a boolean")

    prompt_ids = artifact.tokenizer.encode(prompt, add_bos=add_bos, add_eos=False)
    if not prompt_ids:
        raise ValueError("prompt produces no tokens; enable BOS or provide non-empty text")
    maximum_length = artifact.model_config.seq_length
    if len(prompt_ids) + max_new_tokens > maximum_length:
        raise ValueError(
            "prompt plus requested generation exceeds model sequence length: "
            f"{len(prompt_ids)} + {max_new_tokens} > {maximum_length}"
        )

    all_ids = list(prompt_ids)
    generated: list[int] = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    stop_reason = "length"
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)
            output = model(input_ids)
            if output.logits is None:
                raise RuntimeError("single-device GPT forward returned no logits")
            next_token = _sample_next_token(
                output.logits[0, -1],
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=top_k,
                generator=generator,
            )
            all_ids.append(next_token)
            generated.append(next_token)
            if next_token == artifact.tokenizer.eos_token_id:
                stop_reason = "eos"
                break

    return GenerationResult(
        prompt=prompt,
        text=artifact.tokenizer.decode(all_ids, skip_special_tokens=True),
        completion=artifact.tokenizer.decode(generated, skip_special_tokens=True),
        token_ids=tuple(all_ids),
        generated_token_ids=tuple(generated),
        stop_reason=stop_reason,
    )


def generate_from_artifact(
    artifact_path: str | Path,
    prompt: str,
    *,
    device: str | torch.device = "auto",
    dtype: str = "auto",
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 0,
    seed: int = 1234,
    add_bos: bool = True,
) -> GenerationResult:
    artifact, model, resolved_device = load_model_for_generation(
        artifact_path,
        device=device,
        dtype=dtype,
    )
    return generate_text(
        artifact,
        model,
        prompt,
        device=resolved_device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        add_bos=add_bos,
    )


__all__ = [
    "GenerationResult",
    "generate_from_artifact",
    "generate_text",
    "load_model_for_generation",
]
