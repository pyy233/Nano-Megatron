from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch import Tensor, nn

from ._common import (
    build_adamw,
    collect_domain_parameters,
    communication_is_active,
    config_value,
    global_parameter_grad_norm,
    process_group,
    scale_parameter_gradients,
    unique_process_groups,
)
from .buckets import FlatBucket, build_flat_buckets
from .interface import DataParallelStrategy

if TYPE_CHECKING:
    from nano_megatron.parallel import ParameterDomainRegistry


class DDPStrategy(DataParallelStrategy):
    """Readable replicated-data parallelism with one explicit final all-reduce.

    Native DDP's ``no_sync`` context must span each matching forward/backward
    graph, which is awkward for GPipe's reverse drain and 1F1B's interleaving.
    A manual flat-bucket path keeps accumulation, pipeline, tied embeddings,
    parameter domains, and reduction dtype correct with one visible lifecycle.
    """

    mode = "ddp"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.optimizer: torch.optim.Optimizer | None = None
        self._buckets: list[FlatBucket] = []
        self._manual_gradients_synchronized = False
        self.gradient_sync_count = 0
        self._domain_parameters = []

    def setup(
        self,
        model: nn.Module,
        optimizer_config: object,
        parameter_domains: ParameterDomainRegistry | None = None,
    ) -> nn.Module:
        registry = self._resolve_registry(parameter_domains)
        parameters = collect_domain_parameters(model, registry, self.parallel)
        self._domain_parameters = parameters
        groups = unique_process_groups(parameters)
        active_groups = [group for group in groups if group.size > 1]
        if active_groups and not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("DDP with replica size > 1 requires initialized torch.distributed")

        self._buckets = build_flat_buckets(
            parameters,
            bucket_bytes=int(config_value(self.config, "bucket_bytes", 256 * 1024**2)),
        )
        self._broadcast_manual_parameters()
        wrapped: nn.Module = model

        self._model = wrapped
        self.optimizer = build_adamw(model.parameters(), optimizer_config)
        return wrapped

    @torch.no_grad()
    def _broadcast_manual_parameters(self) -> None:
        for bucket in self._buckets:
            if not communication_is_active(bucket.group):
                continue
            flat = bucket.pack_parameters()
            dist.broadcast(
                flat,
                src=int(bucket.group.ranks[0]),
                group=process_group(bucket.group),
            )
            bucket.unpack_parameters(flat)

    def backward(self, loss: Tensor) -> None:
        loss.backward()

    def finalize_gradients(self) -> None:
        if not self._manual_gradients_synchronized:
            self._synchronize_manual_gradients()

    def _synchronize_manual_gradients(self) -> None:
        for bucket in self._buckets:
            bucket.all_reduce_gradient(dtype=self.grad_reduce_dtype)
        self._manual_gradients_synchronized = True
        self.gradient_sync_count += 1

    def clip_grad_norm(self, max_norm: float) -> Tensor:
        self.finalize_gradients()
        total_norm = global_parameter_grad_norm(
            self._domain_parameters,
            self.parallel,
        )
        if max_norm > 0:
            coefficient = torch.clamp(max_norm / (total_norm + 1.0e-6), max=1.0)
            scale_parameter_gradients(self._domain_parameters, coefficient)
        return total_norm

    def optimizer_step(self) -> None:
        if self.optimizer is None:
            raise RuntimeError("DDPStrategy.setup() must be called before optimizer_step()")
        self.finalize_gradients()
        self.optimizer.step()

    def zero_grad(self) -> None:
        if self.optimizer is None:
            raise RuntimeError("DDPStrategy.setup() must be called before zero_grad()")
        self.optimizer.zero_grad(set_to_none=True)
        for bucket in self._buckets:
            bucket.clear_gradients()
        self._manual_gradients_synchronized = False
        self.gradient_sync_count = 0

    def state_dict(self) -> Mapping[str, Any]:
        if self.optimizer is None:
            raise RuntimeError("DDPStrategy.setup() must be called before state_dict()")
        return {"mode": self.mode, "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if self.optimizer is None:
            raise RuntimeError("DDPStrategy.setup() must be called before load_state_dict()")
        mode = state_dict.get("mode", self.mode)
        if mode != self.mode:
            raise ValueError(f"cannot load {mode!r} state into DDPStrategy")
        optimizer = state_dict.get("optimizer")
        if not isinstance(optimizer, Mapping):
            raise ValueError("DDP checkpoint is missing optimizer state")
        self.optimizer.load_state_dict(dict(optimizer))


class ReplicatedStrategy(DDPStrategy):
    """The size-one DDP degeneration, named explicitly for tests and examples."""

    def setup(
        self,
        model: nn.Module,
        optimizer_config: object,
        parameter_domains: ParameterDomainRegistry | None = None,
    ) -> nn.Module:
        wrapped = super().setup(model, optimizer_config, parameter_domains)
        if any(bucket.world_size > 1 for bucket in self._buckets):
            raise ValueError("ReplicatedStrategy is only valid when all replica groups have size 1")
        return wrapped
