from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch import Tensor, nn

from nano_megatron.parallel import GroupKey, ParameterDomain

from ._common import (
    DomainParameter,
    build_adamw,
    collect_domain_parameters,
    config_value,
    global_parameter_grad_norm,
    scale_parameter_gradients,
)
from .interface import DataParallelStrategy

if TYPE_CHECKING:
    from nano_megatron.parallel import ParameterDomainRegistry


class Zero3Strategy(DataParallelStrategy):
    """A small FSDP2 adapter; size-one meshes intentionally degenerate to local AdamW."""

    mode = "zero3"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.optimizer: torch.optim.Optimizer | None = None
        self.uses_fsdp2 = False
        self.trace: list[str] = []
        self._domain_parameters: list[DomainParameter] = []
        self._norm_metadata: dict[str, DomainParameter] = {}
        self._fsdp_units: tuple[nn.Module, ...] = ()

    @property
    def checkpoint_domain(self) -> ParameterDomain:
        domains = {item.domain for item in self._domain_parameters}
        if len(domains) != 1:
            raise RuntimeError(
                "ZeRO-3 checkpointing requires exactly one parameter domain per model tree"
            )
        return next(iter(domains))

    @property
    def checkpoint_group(self) -> Any:
        groups = {
            (tuple(item.group.ranks), id(item.group.process_group)): item.group
            for item in self._domain_parameters
        }
        if len(groups) != 1:
            raise RuntimeError(
                "ZeRO-3 checkpointing requires one replica group per model tree"
            )
        return next(iter(groups.values()))

    def setup(
        self,
        model: nn.Module,
        optimizer_config: object,
        parameter_domains: ParameterDomainRegistry | None = None,
    ) -> nn.Module:
        registry = self._resolve_registry(parameter_domains)
        parameters = collect_domain_parameters(model, registry, self.parallel)
        self._domain_parameters = parameters
        self._norm_metadata = {item.name: item for item in parameters}
        domains = {item.domain for item in parameters}
        groups = {item.domain: item.group for item in parameters}
        active_domains = {domain for domain, group in groups.items() if group.size > 1}

        if not active_domains:
            if self.offload_policy.zero3_params_and_grads:
                raise ValueError(
                    "ZeRO-3 parameter/gradient offload requires an active replica mesh; "
                    "size-one ZeRO-3 degenerates to local AdamW"
                )
            self._model = model
            self.optimizer = build_adamw(model.parameters(), optimizer_config)
            self.trace.append("size-one: ZeRO-3 degenerated to replicated AdamW")
            return model
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "ZeRO-3 with replica size > 1 requires initialized torch.distributed"
            )
        if len(domains) > 1:
            raise NotImplementedError(
                "phase-one FSDP2 setup supports one parameter domain per module tree; "
                "MoE must fully_shard dense and expert submodules with their respective meshes"
            )

        try:
            from torch.distributed.fsdp import (
                CPUOffloadPolicy,
                MixedPrecisionPolicy,
                fully_shard,
            )
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "ZeRO-3 requires the FSDP2 fully_shard API from PyTorch 2.6 or newer"
            ) from error

        domain = next(iter(domains), ParameterDomain.DENSE)
        group_key = (
            GroupKey.DENSE_REPLICA
            if domain is ParameterDomain.DENSE
            else GroupKey.EXPERT_REPLICA
        )
        try:
            mesh = self.parallel.mesh(group_key)
        except Exception as error:
            raise RuntimeError(
                f"ParallelContext could not provide the {group_key.value} DeviceMesh for ZeRO-3"
            ) from error

        options: dict[str, Any] = {
            "mesh": mesh,
            "reshard_after_forward": bool(
                config_value(self.config, "reshard_after_forward", True)
            ),
        }
        if self.grad_reduce_dtype is not None:
            options["mp_policy"] = MixedPrecisionPolicy(
                reduce_dtype=self.grad_reduce_dtype,
            )
        if self.offload_policy.zero3_params_and_grads:
            options["offload_policy"] = CPUOffloadPolicy(
                pin_memory=self.offload_policy.pin_memory,
            )
        sharding_units = getattr(model, "sharding_units", None)
        explicit_units: tuple[nn.Module, ...] | None = None
        if callable(sharding_units):
            explicit_units = tuple(sharding_units())
            if not explicit_units:
                raise ValueError("ZeRO-3 sharding_units() must return at least one module")
            if any(not isinstance(unit, nn.Module) for unit in explicit_units):
                raise TypeError("ZeRO-3 sharding_units() must return only nn.Module instances")

        try:
            if explicit_units is None:
                sharded_model = fully_shard(model, **options)
                fsdp_units = (sharded_model,)
            else:
                for unit in explicit_units:
                    fully_shard(unit, **options)
                sharded_model = model
                fsdp_units = explicit_units
        except (RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError(
                "FSDP2 fully_shard setup failed; verify PyTorch >=2.6 and that the "
                "ParallelContext mesh matches the model device"
            ) from error

        self._model = sharded_model
        self._fsdp_units = fsdp_units
        self._domain_parameters = [
            DomainParameter(
                name=name,
                parameter=parameter,
                domain=self._norm_metadata[name].domain,
                group=self._norm_metadata[name].group,
                tensor_sharded=self._norm_metadata[name].tensor_sharded,
                tied_source_rank=self._norm_metadata[name].tied_source_rank,
            )
            for name, parameter in sharded_model.named_parameters()
        ]
        self.optimizer = build_adamw(sharded_model.parameters(), optimizer_config)
        self.uses_fsdp2 = True
        self.trace.append(f"fully_shard:{group_key.value}")
        return sharded_model

    @contextmanager
    def microbatch_context(
        self,
        *,
        is_last_microbatch: bool,
        unit: nn.Module | None = None,
    ) -> Iterator[None]:
        previous = self._sync_this_backward
        self._sync_this_backward = is_last_microbatch
        if (
            self.uses_fsdp2
            and unit is not None
            and all(unit is not candidate for candidate in self._fsdp_units)
        ):
            raise ValueError("ZeRO-3 microbatch unit is not one of the fully-sharded modules")
        sync_unit = self.model if unit is None else unit
        set_sync = getattr(sync_unit, "set_requires_gradient_sync", None)
        if self.uses_fsdp2 and callable(set_sync):
            set_sync(is_last_microbatch, recurse=True)
            self.trace.append(f"gradient_sync:{is_last_microbatch}")
        try:
            with self.activation_context():
                yield
        finally:
            if self.uses_fsdp2 and callable(set_sync) and not is_last_microbatch:
                set_sync(True, recurse=True)
            self._sync_this_backward = previous

    def backward(self, loss: Tensor) -> None:
        loss.backward()
        self.trace.append("backward")

    def clip_grad_norm(self, max_norm: float) -> Tensor:
        total_norm = global_parameter_grad_norm(
            self._domain_parameters,
            self.parallel,
            replica_sharded=self.uses_fsdp2,
        )
        if max_norm > 0:
            coefficient = torch.clamp(max_norm / (total_norm + 1.0e-6), max=1.0)
            scale_parameter_gradients(self._domain_parameters, coefficient)
        return total_norm

    def optimizer_step(self) -> None:
        if self.optimizer is None:
            raise RuntimeError("Zero3Strategy.setup() must be called before optimizer_step()")
        self.optimizer.step()
        self.trace.append("optimizer_step")

    def zero_grad(self) -> None:
        if self.optimizer is None:
            raise RuntimeError("Zero3Strategy.setup() must be called before zero_grad()")
        self.optimizer.zero_grad(set_to_none=True)

    def distributed_checkpoint_state_dict(self) -> dict[str, Any]:
        if self.optimizer is None:
            raise RuntimeError("Zero3Strategy.setup() must be called before checkpointing")
        from torch.distributed.checkpoint.state_dict import get_state_dict

        model_state, optimizer_state = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state,
            "optimizer": optimizer_state,
        }

    def load_distributed_checkpoint_state_dict(self, state: Mapping[str, Any]) -> None:
        if self.optimizer is None:
            raise RuntimeError("Zero3Strategy.setup() must be called before checkpointing")
        model_state = state.get("model")
        optimizer_state = state.get("optimizer")
        if not isinstance(model_state, Mapping) or not isinstance(
            optimizer_state, Mapping
        ):
            raise ValueError("ZeRO-3 DCP state must contain model and optimizer mappings")

        from torch.distributed.checkpoint.state_dict import set_state_dict

        incompatible = set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=dict(model_state),
            optim_state_dict=dict(optimizer_state),
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "ZeRO-3 checkpoint model keys are incompatible: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

    def state_dict(self) -> Mapping[str, Any]:
        if self.optimizer is None:
            raise RuntimeError("Zero3Strategy.setup() must be called before state_dict()")
        return {
            "mode": self.mode,
            "uses_fsdp2": self.uses_fsdp2,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if self.optimizer is None:
            raise RuntimeError("Zero3Strategy.setup() must be called before load_state_dict()")
        mode = state_dict.get("mode", self.mode)
        if mode != self.mode:
            raise ValueError(f"cannot load {mode!r} state into Zero3Strategy")
        optimizer = state_dict.get("optimizer")
        if not isinstance(optimizer, Mapping):
            raise ValueError("ZeRO-3 checkpoint is missing optimizer state")
        self.optimizer.load_state_dict(dict(optimizer))
