"""Memory-bounded P2P ring context-parallel attention."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.autograd.function import once_differentiable

from nano_megatron.parallel.group import require_parallel_group

from .utils import group_global_rank, group_rank, group_size, unwrap_process_group


def _compute_dtype(tensor: Tensor) -> torch.dtype:
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def _validate_inputs(q: Tensor, k: Tensor, v: Tensor) -> None:
    if (
        q.ndim != 4
        or k.ndim != 4
        or k.shape != v.shape
        or q.shape[0] != k.shape[0]
        or q.shape[-2] != k.shape[-2]
        or q.shape[-1] != k.shape[-1]
    ):
        raise ValueError("q, k, v must have compatible [batch, heads, sequence, dim] shapes")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if q.size(1) % k.size(1):
        raise ValueError("query heads must be divisible by KV heads")


def _expand_kv_heads(k: Tensor, v: Tensor, query_heads: int) -> tuple[Tensor, Tensor, int]:
    repeat = query_heads // k.size(1)
    if repeat == 1:
        return k, v, repeat
    return (
        k.repeat_interleave(repeat, dim=1),
        v.repeat_interleave(repeat, dim=1),
        repeat,
    )


def _collapse_kv_heads(gradient: Tensor, kv_heads: int, repeat: int) -> Tensor:
    if repeat == 1:
        return gradient
    batch, _, sequence, dimension = gradient.shape
    return gradient.reshape(batch, kv_heads, repeat, sequence, dimension).sum(dim=2)


def _causal_mask(
    q: Tensor,
    k: Tensor,
    *,
    query_offset: int,
    block_rank: int,
) -> Tensor:
    query_positions = torch.arange(
        query_offset,
        query_offset + q.size(-2),
        device=q.device,
        dtype=torch.long,
    )
    key_start = block_rank * k.size(-2)
    key_positions = torch.arange(
        key_start,
        key_start + k.size(-2),
        device=q.device,
        dtype=torch.long,
    )
    valid = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    return valid.view(1, 1, q.size(-2), k.size(-2))


def _rotate_forward(tensor: Tensor, group: Any, *, tag: int) -> Tensor:
    """Move one block to the next rank and receive the previous rank's block."""

    size = group_size(group)
    if size == 1:
        return tensor
    rank = group_rank(group)
    send_rank = group_global_rank(group, (rank + 1) % size)
    receive_rank = group_global_rank(group, (rank - 1) % size)
    process_group = unwrap_process_group(group)
    send = tensor.contiguous()
    received = torch.empty_like(send)
    operations = [
        dist.P2POp(dist.isend, send, send_rank, process_group, tag=tag),
        dist.P2POp(dist.irecv, received, receive_rank, process_group, tag=tag),
    ]
    requests = dist.batch_isend_irecv(operations)
    for request in requests:
        request.wait()
    return received


def _ring_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    group: Any,
    *,
    causal: bool,
    sequence_offset: int,
) -> tuple[Tensor, Tensor]:
    compute_dtype = _compute_dtype(q)
    q_compute = q.to(compute_dtype)
    running_max = torch.full(
        (*q.shape[:-1], 1),
        -torch.inf,
        device=q.device,
        dtype=compute_dtype,
    )
    running_sum = torch.zeros_like(running_max)
    running_output = torch.zeros_like(q, dtype=compute_dtype)
    scale = 1.0 / math.sqrt(q.size(-1))
    size = group_size(group)
    rank = group_rank(group)
    current = torch.stack((k, v), dim=0)

    for step in range(size):
        block_rank = (rank - step) % size
        k_block, v_block = current.unbind(dim=0)
        k_expanded, v_expanded, _ = _expand_kv_heads(k_block, v_block, q.size(1))
        scores = torch.matmul(
            q_compute,
            k_expanded.to(compute_dtype).transpose(-1, -2),
        ).mul_(scale)
        if causal:
            valid = _causal_mask(
                q,
                k_block,
                query_offset=sequence_offset,
                block_rank=block_rank,
            )
            scores.masked_fill_(~valid, -torch.inf)

        block_max = scores.amax(dim=-1, keepdim=True)
        new_max = torch.maximum(running_max, block_max)
        old_scale = torch.where(
            torch.isfinite(running_max),
            torch.exp(running_max - new_max),
            torch.zeros_like(running_max),
        )
        probabilities = torch.exp(scores - new_max)
        probabilities = torch.nan_to_num(probabilities, nan=0.0)
        running_output.mul_(old_scale).add_(
            torch.matmul(probabilities, v_expanded.to(compute_dtype))
        )
        running_sum.mul_(old_scale).add_(probabilities.sum(dim=-1, keepdim=True))
        running_max = new_max

        if step + 1 < size:
            current = _rotate_forward(current, group, tag=10_000 + step)

    tiny = torch.finfo(compute_dtype).tiny
    output = running_output / running_sum.clamp_min(tiny)
    logsumexp = running_max + torch.log(running_sum.clamp_min(tiny))
    return output.to(q.dtype), logsumexp


def _ring_backward(
    grad_output: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    output: Tensor,
    logsumexp: Tensor,
    group: Any,
    *,
    causal: bool,
    sequence_offset: int,
) -> tuple[Tensor, Tensor, Tensor]:
    compute_dtype = _compute_dtype(q)
    q_compute = q.to(compute_dtype)
    grad_compute = grad_output.to(compute_dtype)
    output_compute = output.to(compute_dtype)
    delta = (grad_compute * output_compute).sum(dim=-1, keepdim=True)
    grad_q = torch.zeros_like(q_compute)
    size = group_size(group)
    rank = group_rank(group)
    kv_heads = k.size(1)
    current = torch.stack(
        (
            k.to(compute_dtype),
            v.to(compute_dtype),
            torch.zeros_like(k, dtype=compute_dtype),
            torch.zeros_like(v, dtype=compute_dtype),
        ),
        dim=0,
    )
    scale = 1.0 / math.sqrt(q.size(-1))

    # Each K/V block carries its accumulated gradient around one complete ring.
    # After ``size`` rotations it is back at its owner with contributions from
    # every query rank, while dQ is accumulated locally and never communicated.
    for step in range(size):
        block_rank = (rank - step) % size
        k_block, v_block, grad_k_accumulator, grad_v_accumulator = current.unbind(dim=0)
        k_expanded, v_expanded, repeat = _expand_kv_heads(
            k_block,
            v_block,
            q.size(1),
        )
        scores = torch.matmul(q_compute, k_expanded.transpose(-1, -2)).mul_(scale)
        if causal:
            valid = _causal_mask(
                q,
                k_block,
                query_offset=sequence_offset,
                block_rank=block_rank,
            )
            scores.masked_fill_(~valid, -torch.inf)
        probabilities = torch.exp(scores - logsumexp.to(compute_dtype))
        probabilities = torch.nan_to_num(probabilities, nan=0.0)

        grad_probabilities = torch.matmul(grad_compute, v_expanded.transpose(-1, -2))
        grad_scores = probabilities * (grad_probabilities - delta)
        grad_q.add_(torch.matmul(grad_scores, k_expanded).mul_(scale))
        grad_k_expanded = torch.matmul(
            grad_scores.transpose(-1, -2),
            q_compute,
        ).mul_(scale)
        grad_v_expanded = torch.matmul(
            probabilities.transpose(-1, -2),
            grad_compute,
        )
        grad_k_accumulator.add_(
            _collapse_kv_heads(grad_k_expanded, kv_heads, repeat)
        )
        grad_v_accumulator.add_(
            _collapse_kv_heads(grad_v_expanded, kv_heads, repeat)
        )

        current = torch.stack(
            (k_block, v_block, grad_k_accumulator, grad_v_accumulator),
            dim=0,
        )
        if size > 1:
            current = _rotate_forward(current, group, tag=20_000 + step)

    _, _, grad_k, grad_v = current.unbind(dim=0)
    return grad_q.to(q.dtype), grad_k.to(k.dtype), grad_v.to(v.dtype)


class _RingAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        group: Any,
        causal: bool,
        sequence_offset: int,
    ) -> Tensor:
        output, logsumexp = _ring_forward(
            q,
            k,
            v,
            group,
            causal=causal,
            sequence_offset=sequence_offset,
        )
        ctx.save_for_backward(q, k, v, output, logsumexp)
        ctx.group = group
        ctx.causal = causal
        ctx.sequence_offset = sequence_offset
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Any, ...]:
        q, k, v, output, logsumexp = ctx.saved_tensors
        grad_q, grad_k, grad_v = _ring_backward(
            grad_output,
            q,
            k,
            v,
            output,
            logsumexp,
            ctx.group,
            causal=ctx.causal,
            sequence_offset=ctx.sequence_offset,
        )
        return grad_q, grad_k, grad_v, None, None, None


class RingContextParallelAttention(nn.Module):
    """Online-softmax attention with one P2P K/V block resident per ring step."""

    def __init__(self, group: Any, *, dropout_p: float = 0.0) -> None:
        super().__init__()
        if dropout_p != 0.0:
            raise ValueError("ring context-parallel attention requires dropout_p=0 in phase one")
        self.group = require_parallel_group(group, name="context-parallel group")
        self.dropout_p = dropout_p

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        causal: bool = True,
        sequence_offset: int | None = None,
    ) -> Tensor:
        _validate_inputs(q, k, v)
        local_sequence = q.size(-2)
        offset = (
            group_rank(self.group) * local_sequence
            if sequence_offset is None
            else int(sequence_offset)
        )
        return _RingAttention.apply(q, k, v, self.group, causal, offset)
