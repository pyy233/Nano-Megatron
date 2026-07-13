"""Autograd-aware tensor- and sequence-parallel collectives."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


def _normalize_dim(dim: int, ndim: int) -> int:
    normalized = dim if dim >= 0 else ndim + dim
    if not 0 <= normalized < ndim:
        raise IndexError(f"dimension {dim} is invalid for a {ndim}-D tensor")
    return normalized


def _split(x: Tensor, group: ParallelGroupLike, dim: int) -> Tensor:
    dim = _normalize_dim(dim, x.ndim)
    if x.shape[dim] % group.size:
        raise ValueError(
            f"dimension {dim} with size {x.shape[dim]} is not divisible by group size {group.size}"
        )
    return x.chunk(group.size, dim=dim)[group.rank].contiguous()


def _all_reduce(x: Tensor, group: ParallelGroupLike) -> Tensor:
    if group.size == 1:
        return x
    process_group = require_distributed(group)
    output = x.contiguous().clone()
    dist.all_reduce(output, group=process_group)
    return output


def _all_gather(x: Tensor, group: ParallelGroupLike, dim: int) -> Tensor:
    if group.size == 1:
        return x
    process_group = require_distributed(group)
    gathered = [torch.empty_like(x) for _ in range(group.size)]
    dist.all_gather(gathered, x.contiguous(), group=process_group)
    return torch.cat(gathered, dim=dim).contiguous()


def _reduce_scatter(x: Tensor, group: ParallelGroupLike, dim: int) -> Tensor:
    if group.size == 1:
        return x
    process_group = require_distributed(group)
    dim = _normalize_dim(dim, x.ndim)
    if x.shape[dim] % group.size:
        raise ValueError(
            f"dimension {dim} with size {x.shape[dim]} is not divisible by group size {group.size}"
        )
    moved = x.movedim(dim, 0).contiguous()
    output_shape = list(moved.shape)
    output_shape[0] //= group.size
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    dist.reduce_scatter_tensor(output, moved, group=process_group)
    return output.movedim(0, dim).contiguous()


class _CopyToTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, group: ParallelGroupLike) -> Tensor:
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        return _all_reduce(grad_output, ctx.group), None


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, group: ParallelGroupLike) -> Tensor:
        return _all_reduce(x, group)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        return grad_output, None


class _GatherFromRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, group: ParallelGroupLike, dim: int) -> Tensor:
        ctx.group = group
        ctx.dim = dim
        return _all_gather(x, group, dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None]:
        return _split(grad_output, ctx.group, ctx.dim), None, None


class _ScatterToRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, group: ParallelGroupLike, dim: int) -> Tensor:
        ctx.group = group
        ctx.dim = dim
        return _split(x, group, dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None]:
        return _all_gather(grad_output, ctx.group, ctx.dim), None, None


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, group: ParallelGroupLike, dim: int) -> Tensor:
        ctx.group = group
        ctx.dim = dim
        return _all_gather(x, group, dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None]:
        return _reduce_scatter(grad_output, ctx.group, ctx.dim), None, None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, group: ParallelGroupLike, dim: int) -> Tensor:
        ctx.group = group
        ctx.dim = dim
        return _reduce_scatter(x, group, dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None]:
        return _all_gather(grad_output, ctx.group, ctx.dim), None, None


def copy_to_tensor_parallel_region(
    x: Tensor,
    group: ParallelContext | ParallelGroup | ParallelGroupLike,
) -> Tensor:
    return _CopyToTensorParallelRegion.apply(x, tensor_parallel_group(group))


def reduce_from_tensor_parallel_region(
    x: Tensor,
    group: ParallelContext | ParallelGroup | ParallelGroupLike,
) -> Tensor:
    return _ReduceFromTensorParallelRegion.apply(x, tensor_parallel_group(group))


def gather_from_tensor_parallel_region(
    x: Tensor,
    group: ParallelContext | ParallelGroup | ParallelGroupLike,
    *,
    dim: int = -1,
) -> Tensor:
    return _GatherFromRegion.apply(x, tensor_parallel_group(group), dim)


def scatter_to_tensor_parallel_region(
    x: Tensor,
    group: ParallelContext | ParallelGroup | ParallelGroupLike,
    *,
    dim: int = -1,
) -> Tensor:
    return _ScatterToRegion.apply(x, tensor_parallel_group(group), dim)


def gather_from_sequence_parallel_region(
    x: Tensor,
    group: ParallelContext | ParallelGroup | ParallelGroupLike,
    *,
    sequence_dim: int = 1,
) -> Tensor:
    return _GatherFromSequenceParallelRegion.apply(x, tensor_parallel_group(group), sequence_dim)


def reduce_scatter_to_sequence_parallel_region(
    x: Tensor,
    group: ParallelContext | ParallelGroup | ParallelGroupLike,
    *,
    sequence_dim: int = 1,
) -> Tensor:
    return _ReduceScatterToSequenceParallelRegion.apply(
        x, tensor_parallel_group(group), sequence_dim
    )


def scatter_to_sequence_parallel_region(
    x: Tensor,
    group: ParallelContext | ParallelGroup | ParallelGroupLike,
    *,
    sequence_dim: int = 1,
) -> Tensor:
    """Scatter without reduction; useful when the input is already replicated."""

    return _ScatterToRegion.apply(x, tensor_parallel_group(group), sequence_dim)


def register_sequence_parallel_gradient_hooks(
    module: nn.Module,
    group: ParallelContext | ParallelGroup | ParallelGroupLike,
    *,
    recurse: bool = True,
) -> None:
    """All-reduce gradients for parameters replicated across SP ranks.

    Megatron batches this reduction near the end of backward. Nano-Megatron's
    first implementation uses direct hooks: less overlap, but a much smaller
    and easier-to-follow correctness path.
    """

    resolved = tensor_parallel_group(group)
    if resolved.size == 1:
        return
    for parameter in module.parameters(recurse=recurse):
        registered_groups = getattr(
            parameter,
            "_nano_megatron_sequence_parallel_groups",
            set(),
        )
        identity = id(resolved.process_group)
        if identity in registered_groups:
            continue

        def reduce_gradient(gradient: Tensor, parallel_group=resolved) -> Tensor:
            return _all_reduce(gradient, parallel_group)

        parameter.register_hook(reduce_gradient)
        parameter._nano_megatron_sequence_parallel_groups = {  # type: ignore[attr-defined]
            *registered_groups,
            identity,
        }
