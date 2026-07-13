from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import ceil
from typing import Protocol

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


class _Waitable(Protocol):
    def wait(self) -> object: ...


class GradientReductionRequest:
    """One explicitly started bucket reduction, completed later by ``wait``.

    The request owns its communication buffers until completion.  Keeping the
    collective launch and its post-processing separate is what lets a later
    layer's gradient communication overlap an earlier layer's backward compute.
    """

    def __init__(
        self,
        *,
        bucket: FlatBucket,
        reduction: str,
        input_buffer: Tensor,
        output_buffer: Tensor | None,
        work: _Waitable | None,
    ) -> None:
        if reduction not in {"all_reduce", "reduce_scatter", "all_reduce_slice"}:
            raise ValueError(f"unsupported gradient reduction: {reduction!r}")
        self.bucket = bucket
        self.reduction = reduction
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.work = work
        self._result: Tensor | None = None

    @property
    def completed(self) -> bool:
        return self._result is not None

    @torch.no_grad()
    def wait(self) -> Tensor:
        """Wait once, normalize the result, and publish it to the bucket."""

        if self._result is not None:
            return self._result
        if self.work is not None:
            self.work.wait()

        bucket = self.bucket
        if self.reduction == "all_reduce":
            self.input_buffer.div_(bucket.world_size)
            bucket.full_gradient = self.input_buffer
            bucket.local_gradient = self.input_buffer[
                bucket.shard_start : bucket.shard_start + bucket.shard_numel
            ].clone()
            bucket.unpack_gradients(self.input_buffer)
            result = self.input_buffer
        else:
            if self.reduction == "reduce_scatter":
                if self.output_buffer is None:
                    raise RuntimeError("reduce-scatter request is missing its output buffer")
                self.output_buffer.div_(bucket.world_size)
                result = self.output_buffer
            else:
                self.input_buffer.div_(bucket.world_size)
                result = self.input_buffer[
                    bucket.shard_start : bucket.shard_start + bucket.shard_numel
                ].clone()
            bucket.full_gradient = None
            bucket.local_gradient = result
            for item in bucket.slices:
                item.parameter.grad = None

        self._result = result
        return result


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

    @property
    def parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(item.parameter for item in self.slices)

    @property
    def defer_until_finalize(self) -> bool:
        """Whether model-specific gradient work must run before replica reduction."""

        return any(item.tied_source_rank is not None for item in self.slices)

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
        return self.start_all_reduce_gradient(dtype=dtype, async_op=False).wait()

    @torch.no_grad()
    def start_all_reduce_gradient(
        self,
        *,
        dtype: torch.dtype | None = None,
        async_op: bool,
    ) -> GradientReductionRequest:
        gradient = self.pack_gradients(dtype=dtype)
        work = None
        if communication_is_active(self.group):  # type: ignore[arg-type]
            work = dist.all_reduce(
                gradient,
                op=dist.ReduceOp.SUM,
                group=process_group(self.group),  # type: ignore[arg-type]
                async_op=async_op,
            )
        return GradientReductionRequest(
            bucket=self,
            reduction="all_reduce",
            input_buffer=gradient,
            output_buffer=None,
            work=work,
        )

    @torch.no_grad()
    def reduce_scatter_gradient(self, *, dtype: torch.dtype | None = None) -> Tensor:
        return self.start_reduce_scatter_gradient(dtype=dtype, async_op=False).wait()

    @torch.no_grad()
    def start_reduce_scatter_gradient(
        self,
        *,
        dtype: torch.dtype | None = None,
        async_op: bool,
    ) -> GradientReductionRequest:
        gradient = self.pack_gradients(dtype=dtype)
        output: Tensor | None = None
        work = None
        reduction = "all_reduce_slice"
        if communication_is_active(self.group):  # type: ignore[arg-type]
            backend = str(getattr(self.group, "backend", "")).lower()
            if backend == "gloo":
                # Some Gloo versions report unsupported reduce-scatter only from
                # Work.wait(), which is too late for a safe fallback.  Select the
                # portable all-reduce-and-slice path before launching communication.
                work = dist.all_reduce(
                    gradient,
                    op=dist.ReduceOp.SUM,
                    group=process_group(self.group),  # type: ignore[arg-type]
                    async_op=async_op,
                )
            else:
                output = torch.empty(
                    self.shard_numel,
                    dtype=gradient.dtype,
                    device=self.device,
                )
                reduce_scatter = getattr(dist, "reduce_scatter_tensor", None)
                if reduce_scatter is None:
                    reduce_scatter = dist.reduce_scatter_single
                work = reduce_scatter(
                    output,
                    gradient,
                    op=dist.ReduceOp.SUM,
                    group=process_group(self.group),  # type: ignore[arg-type]
                    async_op=async_op,
                )
                reduction = "reduce_scatter"
        return GradientReductionRequest(
            bucket=self,
            reduction=reduction,
            input_buffer=gradient,
            output_buffer=output,
            work=work,
        )

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


class BucketGradientReducer:
    """Coordinate ordered bucket reductions from final-microbatch grad hooks."""

    def __init__(
        self,
        buckets: Iterable[FlatBucket],
        *,
        partition_gradients: bool,
        reduction_dtype: torch.dtype | None,
        overlap: bool,
        is_final_backward: Callable[[], bool],
    ) -> None:
        self.buckets = tuple(buckets)
        self.partition_gradients = partition_gradients
        self.reduction_dtype = reduction_dtype
        self.overlap = overlap
        self._is_final_backward = is_final_backward
        reversed_buckets = tuple(reversed(self.buckets))
        self._dispatch_order = tuple(
            bucket for bucket in reversed_buckets if not bucket.defer_until_finalize
        )
        self._deferred_order = tuple(
            bucket for bucket in reversed_buckets if bucket.defer_until_finalize
        )
        self._parameter_to_bucket = {
            parameter: bucket for bucket in self.buckets for parameter in bucket.parameters
        }
        self._hook_handles = []
        if self.overlap:
            for parameter in self._parameter_to_bucket:
                self._hook_handles.append(
                    parameter.register_post_accumulate_grad_hook(self._post_accumulate_hook)
                )
        self._reset_state()

    def _reset_state(self) -> None:
        self._ready = {bucket: set() for bucket in self.buckets}
        self._requests: dict[FlatBucket, GradientReductionRequest] = {}
        self._started_order: list[FlatBucket] = []
        self._next_dispatch = 0
        self._finalized = False
        self.start_count = 0
        self.wait_count = 0

    @property
    def requests(self) -> tuple[GradientReductionRequest, ...]:
        return tuple(self._requests[bucket] for bucket in self._started_order)

    @property
    def finalized(self) -> bool:
        return self._finalized

    def _post_accumulate_hook(self, parameter: Tensor) -> None:
        if self._is_final_backward():
            self.mark_parameter_ready(parameter)

    def mark_parameter_ready(self, parameter: Tensor) -> None:
        if self._finalized:
            raise RuntimeError("cannot mark a gradient ready after reduction finalization")
        try:
            bucket = self._parameter_to_bucket[parameter]
        except KeyError as error:
            raise ValueError("parameter does not belong to this bucket reducer") from error
        self._ready[bucket].add(parameter)
        self._start_ready_buckets()

    def _start_ready_buckets(self) -> None:
        while self._next_dispatch < len(self._dispatch_order):
            bucket = self._dispatch_order[self._next_dispatch]
            if len(self._ready[bucket]) != len(bucket.parameters):
                return
            self._start(bucket, async_op=True)
            self._next_dispatch += 1

    def _start(self, bucket: FlatBucket, *, async_op: bool) -> None:
        if bucket in self._requests:
            raise RuntimeError(f"bucket {bucket.index} gradient reduction started twice")
        if self.partition_gradients:
            request = bucket.start_reduce_scatter_gradient(
                dtype=self.reduction_dtype,
                async_op=async_op,
            )
        else:
            request = bucket.start_all_reduce_gradient(
                dtype=self.reduction_dtype,
                async_op=async_op,
            )
        self._requests[bucket] = request
        self._started_order.append(bucket)
        self.start_count += 1

    def finalize(self) -> None:
        """Start unused/deferred buckets, then wait for every reduction exactly once."""

        if self._finalized:
            return
        while self._next_dispatch < len(self._dispatch_order):
            bucket = self._dispatch_order[self._next_dispatch]
            self._start(bucket, async_op=self.overlap)
            self._next_dispatch += 1
        for bucket in self._deferred_order:
            self._start(bucket, async_op=self.overlap)
        for bucket in self._started_order:
            request = self._requests[bucket]
            if not request.completed:
                request.wait()
                self.wait_count += 1
        self._finalized = True

    def reset(self) -> None:
        """Prepare the reducer for a new optimizer step without hiding lost waits."""

        unfinished = [request for request in self.requests if not request.completed]
        if unfinished:
            raise RuntimeError(
                "cannot reset gradient buckets while reductions are still outstanding; "
                "call finalize_gradients() first"
            )
        self._reset_state()
