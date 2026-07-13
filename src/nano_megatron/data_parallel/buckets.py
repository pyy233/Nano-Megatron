from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil

import torch
import torch.distributed as dist
from torch import Tensor, nn

from nano_megatron.parallel import ParameterDomain

from ._common import (
    DomainParameter,
    communication_is_active,
    group_rank,
    group_size,
    process_group,
)


@dataclass(frozen=True)
class ParameterSlice:
    name: str
    parameter: nn.Parameter
    start: int
    end: int
    tensor_sharded: bool
    tied_source_rank: int | None

    @property
    def numel(self) -> int:
        return self.end - self.start


class FlatBucket:
    """A flat parameter/gradient bucket with an equal-sized shard on each replica rank."""

    def __init__(
        self,
        *,
        index: int,
        domain: ParameterDomain,
        group: object,
        parameters: list[DomainParameter],
    ) -> None:
        if not parameters:
            raise ValueError("a flat bucket must contain at least one parameter")
        self.index = index
        self.domain = domain
        self.group = group
        self.device = parameters[0].parameter.device
        self.dtype = parameters[0].parameter.dtype
        self.slices: list[ParameterSlice] = []
        cursor = 0
        for item in parameters:
            name = item.name
            parameter = item.parameter
            if parameter.device != self.device or parameter.dtype != self.dtype:
                raise ValueError("all parameters in one flat bucket must share device and dtype")
            end = cursor + parameter.numel()
            self.slices.append(
                ParameterSlice(
                    name,
                    parameter,
                    cursor,
                    end,
                    item.tensor_sharded,
                    item.tied_source_rank,
                )
            )
            cursor = end
        self.numel = cursor
        self.world_size = group_size(group)  # type: ignore[arg-type]
        self.rank = group_rank(group)  # type: ignore[arg-type]
        self.shard_numel = ceil(self.numel / self.world_size)
        self.padded_numel = self.shard_numel * self.world_size
        self.shard_start = self.rank * self.shard_numel
        self.shard_end = min(self.shard_start + self.shard_numel, self.numel)
        self.full_gradient: Tensor | None = None
        self.local_gradient: Tensor | None = None

    def pack_parameters(self, *, dtype: torch.dtype | None = None) -> Tensor:
        result = torch.zeros(
            self.padded_numel,
            dtype=dtype or self.dtype,
            device=self.device,
        )
        for item in self.slices:
            result[item.start : item.end].copy_(item.parameter.detach().reshape(-1))
        return result

    @torch.no_grad()
    def unpack_parameters(self, flat: Tensor) -> None:
        if flat.numel() < self.numel:
            raise ValueError(
                f"flat parameter has {flat.numel()} values, expected at least {self.numel}"
            )
        for item in self.slices:
            source = flat[item.start : item.end].view_as(item.parameter)
            item.parameter.copy_(
                source.to(device=item.parameter.device, dtype=item.parameter.dtype)
            )

    def pack_gradients(self, *, dtype: torch.dtype | None = None) -> Tensor:
        result = torch.zeros(
            self.padded_numel,
            dtype=dtype or self.dtype,
            device=self.device,
        )
        for item in self.slices:
            gradient = item.parameter.grad
            if gradient is not None:
                result[item.start : item.end].copy_(gradient.detach().reshape(-1))
        return result

    @torch.no_grad()
    def unpack_gradients(self, flat: Tensor) -> None:
        for item in self.slices:
            source = flat[item.start : item.end].view_as(item.parameter)
            if item.parameter.grad is None:
                item.parameter.grad = source.to(dtype=item.parameter.dtype).clone()
            else:
                item.parameter.grad.copy_(source)

    @torch.no_grad()
    def all_reduce_gradient(self, *, dtype: torch.dtype | None = None) -> Tensor:
        gradient = self.pack_gradients(dtype=dtype)
        if communication_is_active(self.group):  # type: ignore[arg-type]
            dist.all_reduce(
                gradient,
                op=dist.ReduceOp.SUM,
                group=process_group(self.group),  # type: ignore[arg-type]
            )
            gradient.div_(self.world_size)
        self.full_gradient = gradient
        self.local_gradient = gradient[
            self.shard_start : self.shard_start + self.shard_numel
        ].clone()
        self.unpack_gradients(gradient)
        return gradient

    @torch.no_grad()
    def reduce_scatter_gradient(self, *, dtype: torch.dtype | None = None) -> Tensor:
        gradient = self.pack_gradients(dtype=dtype)
        if communication_is_active(self.group):  # type: ignore[arg-type]
            output = torch.empty(
                self.shard_numel,
                dtype=gradient.dtype,
                device=self.device,
            )
            try:
                reduce_scatter = getattr(
                    dist,
                    "reduce_scatter_single",
                    dist.reduce_scatter_tensor,
                )
                reduce_scatter(
                    output,
                    gradient,
                    op=dist.ReduceOp.SUM,
                    group=process_group(self.group),  # type: ignore[arg-type]
                )
                output.div_(self.world_size)
            except (RuntimeError, NotImplementedError):
                # Older Gloo builds do not implement reduce_scatter_tensor.  This preserves
                # ZeRO-2 ownership semantics while keeping the CPU teaching path testable.
                dist.all_reduce(
                    gradient,
                    op=dist.ReduceOp.SUM,
                    group=process_group(self.group),  # type: ignore[arg-type]
                )
                gradient.div_(self.world_size)
                output = gradient[
                    self.shard_start : self.shard_start + self.shard_numel
                ].clone()
        else:
            output = gradient[: self.shard_numel].clone()
        self.full_gradient = None
        self.local_gradient = output
        for item in self.slices:
            item.parameter.grad = None
        return output

    @torch.no_grad()
    def all_gather_shards(self, local_shard: Tensor) -> Tensor:
        if local_shard.numel() != self.shard_numel:
            raise ValueError(
                f"local shard has {local_shard.numel()} values, expected {self.shard_numel}"
            )
        if communication_is_active(self.group):  # type: ignore[arg-type]
            gathered = [torch.empty_like(local_shard) for _ in range(self.world_size)]
            dist.all_gather(
                gathered,
                local_shard,
                group=process_group(self.group),  # type: ignore[arg-type]
            )
            return torch.cat(gathered)
        return local_shard

    def clear_gradients(self) -> None:
        self.full_gradient = None
        self.local_gradient = None

    def local_squared_norm(self, parallel: object) -> Tensor:
        """Squared norm of the owned replica shard after duplicate filtering."""

        if self.local_gradient is None:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        tp_rank = int(parallel.tp.rank)  # type: ignore[attr-defined]
        global_rank = int(parallel.rank)  # type: ignore[attr-defined]
        mask = torch.zeros(
            self.shard_numel,
            dtype=torch.bool,
            device=self.local_gradient.device,
        )
        local_start = self.shard_start
        local_end = self.shard_start + self.shard_numel
        for item in self.slices:
            include = item.tensor_sharded or tp_rank == 0
            include = include and (
                item.tied_source_rank is None or global_rank == item.tied_source_rank
            )
            if not include:
                continue
            start = max(item.start, local_start) - local_start
            end = min(item.end, local_end) - local_start
            if end > start:
                mask[start:end] = True
        return self.local_gradient[mask].float().square().sum()

    def metadata(self) -> dict[str, object]:
        return {
            "index": self.index,
            "domain": self.domain.value,
            "parameter_names": [item.name for item in self.slices],
            "numel": self.numel,
            "padded_numel": self.padded_numel,
            "shard_numel": self.shard_numel,
        }


def build_flat_buckets(
    parameters: Iterable[DomainParameter],
    *,
    bucket_bytes: int,
) -> list[FlatBucket]:
    if bucket_bytes <= 0:
        raise ValueError("bucket_bytes must be positive")
    BucketKey = tuple[
        ParameterDomain,
        torch.device,
        torch.dtype,
        tuple[int, ...],
        str,
        str,
        int,
    ]
    grouped: dict[BucketKey, list[DomainParameter]] = {}
    for item in parameters:
        key = (
            item.domain,
            item.parameter.device,
            item.parameter.dtype,
            tuple(item.group.ranks),
            str(item.group.key),
            item.group.channel,
            id(item.group.process_group),
        )
        grouped.setdefault(key, []).append(item)

    buckets: list[FlatBucket] = []
    for items in grouped.values():
        current: list[DomainParameter] = []
        current_bytes = 0
        for item in items:
            parameter_bytes = item.parameter.numel() * item.parameter.element_size()
            if current and current_bytes + parameter_bytes > bucket_bytes:
                buckets.append(
                    FlatBucket(
                        index=len(buckets),
                        domain=items[0].domain,
                        group=items[0].group,
                        parameters=current,
                    )
                )
                current = []
                current_bytes = 0
            current.append(item)
            current_bytes += parameter_bytes
        if current:
            buckets.append(
                FlatBucket(
                    index=len(buckets),
                    domain=items[0].domain,
                    group=items[0].group,
                    parameters=current,
                )
            )
    return buckets
