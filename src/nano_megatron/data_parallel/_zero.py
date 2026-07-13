from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch import Tensor, nn

from nano_megatron.parallel import ParameterDomain

from ._common import (
    collect_domain_parameters,
    communication_is_active,
    config_value,
    process_group,
    reduce_model_parallel_squared_norm,
)
from .buckets import BucketGradientReducer, FlatBucket, build_flat_buckets
from .interface import DataParallelStrategy
from .offload import OptimizerStateStorage

if TYPE_CHECKING:
    from nano_megatron.parallel import ParameterDomainRegistry


@dataclass
class _AdamShard:
    master_parameter: Tensor
    exp_avg: Tensor
    exp_avg_sq: Tensor


class FlatShardZeROStrategy(DataParallelStrategy):
    """Shared readable implementation for optimizer-sharded ZeRO-1 and ZeRO-2."""

    gradient_partitioned: bool

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.buckets: list[FlatBucket] = []
        self._states: list[_AdamShard] = []
        self._storages: list[OptimizerStateStorage] = []
        self._optimizer_config: object | None = None
        self._step = 0
        self._gradients_ready = False
        self.gradient_reducer: BucketGradientReducer | None = None

    def setup(
        self,
        model: nn.Module,
        optimizer_config: object,
        parameter_domains: ParameterDomainRegistry | None = None,
    ) -> nn.Module:
        registry = self._resolve_registry(parameter_domains)
        domain_parameters = collect_domain_parameters(model, registry, self.parallel)
        self.buckets = build_flat_buckets(
            domain_parameters,
            bucket_bytes=int(config_value(self.config, "bucket_bytes", 256 * 1024**2)),
        )
        distributed_buckets = any(bucket.world_size > 1 for bucket in self.buckets)
        if distributed_buckets and not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                f"{self.mode} with replica size > 1 requires initialized torch.distributed"
            )
        self.gradient_reducer = BucketGradientReducer(
            self.buckets,
            partition_gradients=self.gradient_partitioned,
            reduction_dtype=self.grad_reduce_dtype,
            overlap=bool(config_value(self.config, "overlap_grad_reduce", False)),
            is_final_backward=lambda: self._sync_this_backward,
        )
        self._optimizer_config = optimizer_config
        self._model = model
        self._initialize_shards()
        return model

    @contextmanager
    def microbatch_context(
        self,
        *,
        is_last_microbatch: bool,
        unit: nn.Module | None = None,
    ) -> Iterator[None]:
        with super().microbatch_context(
            is_last_microbatch=is_last_microbatch,
            unit=unit,
        ):
            yield

    @torch.no_grad()
    def _initialize_shards(self) -> None:
        self._states.clear()
        self._storages.clear()
        for bucket in self.buckets:
            storage = OptimizerStateStorage(self.offload_policy, bucket.device)
            full_parameter = bucket.pack_parameters(dtype=torch.float32)
            local = full_parameter[
                bucket.shard_start : bucket.shard_start + bucket.shard_numel
            ]
            master = storage.allocate((bucket.shard_numel,))
            master.copy_(local.to(master.device), non_blocking=self.offload_policy.non_blocking)
            self._storages.append(storage)
            self._states.append(
                _AdamShard(
                    master_parameter=master,
                    exp_avg=storage.allocate((bucket.shard_numel,)),
                    exp_avg_sq=storage.allocate((bucket.shard_numel,)),
                )
            )

    def backward(self, loss: Tensor) -> None:
        loss.backward()

    def finalize_gradients(self) -> None:
        if not self._gradients_ready:
            self._reduce_gradients()

    @torch.no_grad()
    def _reduce_gradients(self) -> None:
        if self.gradient_reducer is None:
            raise RuntimeError(f"{type(self).__name__}.setup() must be called first")
        self.gradient_reducer.finalize()
        self._gradients_ready = True

    @torch.no_grad()
    def clip_grad_norm(self, max_norm: float) -> Tensor:
        if not self._gradients_ready:
            self._reduce_gradients()
        if not self.buckets:
            return torch.zeros(())

        squared_norms: list[Tensor] = []
        for bucket in self.buckets:
            gradient = bucket.local_gradient
            if gradient is None:
                continue
            squared = bucket.local_squared_norm(self.parallel)
            if communication_is_active(bucket.group):
                dist.all_reduce(
                    squared,
                    op=dist.ReduceOp.SUM,
                    group=process_group(bucket.group),
                )
            if bucket.domain is ParameterDomain.DENSE and self.parallel.ep.rank != 0:
                squared.zero_()
            squared_norms.append(squared)
        if not squared_norms:
            return torch.zeros((), device=self.buckets[0].device)
        values = [value.to(self.buckets[0].device) for value in squared_norms]
        total_squared = torch.stack(values).sum()
        result_device = total_squared.device
        total_squared = reduce_model_parallel_squared_norm(total_squared, self.parallel)
        total_norm = total_squared.to(result_device).sqrt()
        if max_norm > 0:
            coefficient = torch.clamp(max_norm / (total_norm + 1.0e-6), max=1.0)
            for bucket in self.buckets:
                if bucket.local_gradient is not None:
                    bucket.local_gradient.mul_(coefficient.to(bucket.local_gradient.device))
        return total_norm

    @torch.no_grad()
    def optimizer_step(self) -> None:
        if self._optimizer_config is None:
            raise RuntimeError(f"{type(self).__name__}.setup() must be called first")
        if not self._gradients_ready:
            self._reduce_gradients()

        self._step += 1
        lr = float(config_value(self._optimizer_config, "lr", 3.0e-4))
        beta1, beta2 = tuple(config_value(self._optimizer_config, "betas", (0.9, 0.95)))
        eps = float(config_value(self._optimizer_config, "eps", 1.0e-8))
        weight_decay = float(config_value(self._optimizer_config, "weight_decay", 0.0))
        bias_correction1 = 1.0 - beta1**self._step
        bias_correction2 = 1.0 - beta2**self._step
        step_size = lr / bias_correction1

        for bucket, state, storage in zip(
            self.buckets, self._states, self._storages, strict=True
        ):
            if bucket.local_gradient is None:
                continue
            gradient = storage.move_gradient(bucket.local_gradient)
            state.exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            state.exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
            if weight_decay:
                state.master_parameter.mul_(1.0 - lr * weight_decay)
            denominator = state.exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(eps)
            state.master_parameter.addcdiv_(state.exp_avg, denominator, value=-step_size)

            collective_shard = storage.move_parameter_for_collective(
                state.master_parameter, bucket.dtype
            )
            full_parameter = bucket.all_gather_shards(collective_shard)
            bucket.unpack_parameters(full_parameter)

    def zero_grad(self) -> None:
        if self.gradient_reducer is None:
            raise RuntimeError(f"{type(self).__name__}.setup() must be called first")
        self.gradient_reducer.reset()
        for parameter in self.model.parameters():
            parameter.grad = None
        for bucket in self.buckets:
            bucket.clear_gradients()
        self._gradients_ready = False

    @torch.no_grad()
    def _gather_state_tensor(self, bucket: FlatBucket, tensor: Tensor) -> Tensor:
        local = tensor.to(
            bucket.device,
            dtype=torch.float32,
            non_blocking=self.offload_policy.non_blocking,
        )
        return bucket.all_gather_shards(local)[: bucket.numel].cpu()

    def state_dict(self) -> Mapping[str, Any]:
        bucket_states: dict[str, dict[str, Any]] = {}
        for bucket, state in zip(self.buckets, self._states, strict=True):
            bucket_states[str(bucket.index)] = {
                "metadata": bucket.metadata(),
                "master_parameter": self._gather_state_tensor(
                    bucket, state.master_parameter
                ),
                "exp_avg": self._gather_state_tensor(bucket, state.exp_avg),
                "exp_avg_sq": self._gather_state_tensor(bucket, state.exp_avg_sq),
            }
        return {"mode": self.mode, "step": self._step, "buckets": bucket_states}

    @torch.no_grad()
    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        mode = state_dict.get("mode", self.mode)
        if mode != self.mode:
            raise ValueError(f"cannot load {mode!r} state into {type(self).__name__}")
        raw_buckets = state_dict.get("buckets")
        if not isinstance(raw_buckets, Mapping):
            raise ValueError("ZeRO checkpoint is missing bucket states")
        if len(raw_buckets) != len(self.buckets):
            raise ValueError(
                f"checkpoint has {len(raw_buckets)} buckets, current model has {len(self.buckets)}"
            )

        for bucket, state in zip(self.buckets, self._states, strict=True):
            raw = raw_buckets.get(str(bucket.index))
            if not isinstance(raw, Mapping):
                raise ValueError(f"checkpoint is missing bucket {bucket.index}")
            metadata = raw.get("metadata", {})
            if isinstance(metadata, Mapping):
                expected_names = [item.name for item in bucket.slices]
                saved_names = list(metadata.get("parameter_names", expected_names))
                if saved_names != expected_names:
                    raise ValueError(
                        f"bucket {bucket.index} parameter layout changed: "
                        f"saved={saved_names}, current={expected_names}"
                    )
            for key, target in (
                ("master_parameter", state.master_parameter),
                ("exp_avg", state.exp_avg),
                ("exp_avg_sq", state.exp_avg_sq),
            ):
                full = raw.get(key)
                if not isinstance(full, Tensor) or full.numel() != bucket.numel:
                    raise ValueError(
                        f"bucket {bucket.index} {key} has an incompatible logical shape"
                    )
                padded = torch.zeros(
                    bucket.padded_numel,
                    dtype=torch.float32,
                    device=full.device,
                )
                padded[: bucket.numel].copy_(full.reshape(-1))
                local = padded[
                    bucket.shard_start : bucket.shard_start + bucket.shard_numel
                ]
                target.copy_(
                    local.to(target.device), non_blocking=self.offload_policy.non_blocking
                )
        self._step = int(state_dict.get("step", 0))
