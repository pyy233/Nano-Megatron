"""Cross entropy over vocabulary-sharded logits."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.distributed as dist
from torch import Tensor, nn

from nano_megatron.tensor_parallel._utils import (
    ParallelGroupLike,
    require_distributed,
    tensor_parallel_group,
)

if TYPE_CHECKING:
    from nano_megatron.parallel import ParallelContext, ParallelGroup


def _all_reduce_in_place(
    tensor: Tensor,
    group: ParallelGroupLike,
    op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM,
) -> None:
    if group.size > 1:
        dist.all_reduce(tensor, op=op, group=require_distributed(group))


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        local_logits: Tensor,
        targets: Tensor,
        group: ParallelGroupLike,
        ignore_index: int,
    ) -> Tensor:
        if local_logits.shape[:-1] != targets.shape:
            raise ValueError(
                "targets must match all non-vocabulary logits dimensions: "
                f"logits={tuple(local_logits.shape)}, targets={tuple(targets.shape)}"
            )
        local_vocab_size = local_logits.shape[-1]
        vocab_start = group.rank * local_vocab_size
        vocab_end = vocab_start + local_vocab_size

        compute_logits = (
            local_logits.float()
            if local_logits.dtype in (torch.float16, torch.bfloat16)
            else local_logits
        )
        logits_max = compute_logits.max(dim=-1).values
        _all_reduce_in_place(logits_max, group, dist.ReduceOp.MAX)
        shifted_logits = compute_logits - logits_max.unsqueeze(-1)

        ignored = targets == ignore_index
        outside = (targets < vocab_start) | (targets >= vocab_end) | ignored
        local_targets = (targets - vocab_start).masked_fill(outside, 0)
        predicted = shifted_logits.gather(-1, local_targets.unsqueeze(-1)).squeeze(-1)
        predicted = predicted.masked_fill(outside, 0.0)
        _all_reduce_in_place(predicted, group)

        exp_logits = shifted_logits.exp()
        sum_exp = exp_logits.sum(dim=-1)
        _all_reduce_in_place(sum_exp, group)
        softmax = exp_logits / sum_exp.unsqueeze(-1)
        loss = (sum_exp.log() - predicted).masked_fill(ignored, 0.0)

        ctx.save_for_backward(softmax, local_targets, outside, ignored)
        ctx.input_dtype = local_logits.dtype
        return loss

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None, None]:
        softmax, local_targets, outside, ignored = ctx.saved_tensors
        grad_logits = softmax.clone()
        grad_2d = grad_logits.reshape(-1, grad_logits.shape[-1])
        targets_1d = local_targets.reshape(-1)
        outside_1d = outside.reshape(-1)
        rows = torch.arange(grad_2d.shape[0], device=grad_2d.device)
        grad_2d[rows, targets_1d] -= (~outside_1d).to(grad_2d.dtype)
        grad_logits = grad_2d.reshape_as(grad_logits)
        grad_logits.masked_fill_(ignored.unsqueeze(-1), 0.0)
        gradient = grad_logits * grad_output.to(grad_logits.dtype).unsqueeze(-1)
        return gradient.to(ctx.input_dtype), None, None, None


class VocabParallelCrossEntropy(nn.Module):
    """Numerically stable cross entropy without gathering vocabulary shards."""

    def __init__(
        self,
        *,
        parallel: ParallelContext | ParallelGroup,
        reduction: Literal["none", "mean", "sum"] = "none",
        ignore_index: int = -100,
        original_vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"unsupported reduction: {reduction}")
        self.group = tensor_parallel_group(parallel)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.original_vocab_size = original_vocab_size

    def forward(self, local_logits: Tensor, targets: Tensor) -> Tensor:
        if self.original_vocab_size is not None:
            invalid = (targets != self.ignore_index) & (
                (targets < 0) | (targets >= self.original_vocab_size)
            )
            if torch.any(invalid):
                raise ValueError("targets contain ids outside the original (unpadded) vocabulary")
        loss = _VocabParallelCrossEntropy.apply(
            local_logits, targets, self.group, self.ignore_index
        )
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        valid_count = (targets != self.ignore_index).sum().clamp_min(1)
        return loss.sum() / valid_count
