from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch import Tensor, nn

from nano_megatron.parallel import GroupKey, ParameterDomain

if TYPE_CHECKING:
    from nano_megatron.parallel import ParallelContext, ParallelGroup, ParameterDomainRegistry


@dataclass(frozen=True)
class DomainParameter:
    name: str
    parameter: nn.Parameter
    domain: ParameterDomain
    group: ParallelGroup
    tensor_sharded: bool
    tied_source_rank: int | None = None


def as_parameter_domain(value: Any) -> ParameterDomain:
    if isinstance(value, ParameterDomain):
        return value
    raw = getattr(value, "value", value)
    return ParameterDomain(str(raw))


def parameter_domain(
    registry: ParameterDomainRegistry,
    parameter: nn.Parameter,
) -> ParameterDomain:
    """Resolve a parameter domain exclusively through the explicit registry."""

    placement = registry.placement(parameter)
    return as_parameter_domain(placement.domain)


def replica_group_key(domain: ParameterDomain) -> GroupKey:
    if domain is ParameterDomain.DENSE:
        return GroupKey.DENSE_REPLICA
    if domain is ParameterDomain.EXPERT:
        return GroupKey.EXPERT_REPLICA
    raise ValueError(f"unsupported parameter domain: {domain!r}")


def collect_domain_parameters(
    model: nn.Module,
    registry: ParameterDomainRegistry,
    parallel: ParallelContext,
) -> list[DomainParameter]:
    result: list[DomainParameter] = []
    missing: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        try:
            placement = registry.placement(parameter)
            domain = as_parameter_domain(placement.domain)
        except (KeyError, LookupError):
            missing.append(name)
            continue
        result.append(
            DomainParameter(
                name=name,
                parameter=parameter,
                domain=domain,
                group=parallel.group(placement.replica_group),
                tensor_sharded=bool(placement.tensor_sharded),
                tied_source_rank=getattr(
                    parameter,
                    "_nano_megatron_tied_source_rank",
                    None,
                ),
            )
        )
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            "every trainable parameter must be registered in ParameterDomainRegistry; "
            f"missing: {joined}"
        )
    return result


def group_size(group: ParallelGroup) -> int:
    return int(group.size)


def group_rank(group: ParallelGroup) -> int:
    return int(group.rank)


def process_group(group: ParallelGroup) -> dist.ProcessGroup | None:
    return group.process_group


def communication_is_active(group: ParallelGroup) -> bool:
    return group_size(group) > 1 and dist.is_available() and dist.is_initialized()


def _scalar_collective_device(
    group: ParallelGroup,
    parallel: ParallelContext,
    fallback: torch.device,
) -> torch.device:
    backend = str(group.backend).lower()
    if backend == "nccl":
        device = torch.device(parallel.runtime.device)
        if device.type != "cuda":
            raise RuntimeError("an NCCL norm reduction requires a CUDA runtime device")
        return device
    if backend == "gloo":
        return torch.device("cpu")
    return fallback


def _all_reduce_squared_norm(
    squared: Tensor,
    group: ParallelGroup,
    parallel: ParallelContext,
) -> Tensor:
    if not communication_is_active(group):
        return squared
    reduced = squared.to(_scalar_collective_device(group, parallel, squared.device))
    dist.all_reduce(
        reduced,
        op=dist.ReduceOp.SUM,
        group=process_group(group),
    )
    return reduced


def parameter_contributes_to_model_norm(
    item: DomainParameter,
    parallel: ParallelContext,
    *,
    replica_sharded: bool = False,
) -> bool:
    """Select one copy of replicated parameters while retaining true shards."""

    if not item.tensor_sharded and parallel.tp.rank != 0:
        return False
    if (
        not replica_sharded
        and item.domain is ParameterDomain.DENSE
        and parallel.ep.rank != 0
    ):
        return False
    return item.tied_source_rank is None or parallel.rank == item.tied_source_rank


def _local_gradient(gradient: Tensor) -> Tensor:
    to_local = getattr(gradient, "to_local", None)
    return to_local() if callable(to_local) else gradient


def reduce_model_parallel_squared_norm(
    squared: Tensor,
    parallel: ParallelContext,
) -> Tensor:
    """Sum unique TP/EP shards and PP stages for one CP/DP replica."""

    for group in (parallel.group(GroupKey.TP_EP), parallel.pp):
        squared = _all_reduce_squared_norm(squared, group, parallel)
    return squared


def global_parameter_grad_norm(
    parameters: Iterable[DomainParameter],
    parallel: ParallelContext,
    *,
    replica_sharded: bool = False,
) -> Tensor:
    """Compute one global norm without counting DP/CP/dense-EP replicas twice."""

    items = tuple(parameters)
    device = items[0].parameter.device if items else torch.device("cpu")
    by_replica_group: dict[
        tuple[tuple[int, ...], int, ParameterDomain],
        tuple[ParallelGroup, ParameterDomain, Tensor],
    ] = {}
    local_total = torch.zeros((), dtype=torch.float32, device=device)
    for item in items:
        gradient = item.parameter.grad
        if gradient is None or not parameter_contributes_to_model_norm(
            item,
            parallel,
            replica_sharded=replica_sharded,
        ):
            continue
        squared = _local_gradient(gradient).float().square().sum()
        if not replica_sharded:
            local_total.add_(squared.to(local_total.device))
            continue
        key = (tuple(item.group.ranks), id(item.group.process_group), item.domain)
        if key not in by_replica_group:
            by_replica_group[key] = (
                item.group,
                item.domain,
                torch.zeros((), dtype=torch.float32, device=squared.device),
            )
        by_replica_group[key][2].add_(squared)

    if replica_sharded:
        for group, domain, group_squared in by_replica_group.values():
            group_squared = _all_reduce_squared_norm(group_squared, group, parallel)
            if domain is ParameterDomain.DENSE and parallel.ep.rank != 0:
                group_squared.zero_()
            local_total.add_(group_squared.to(local_total.device))
    result_device = local_total.device
    local_total = reduce_model_parallel_squared_norm(local_total, parallel)
    return local_total.to(result_device).sqrt()


def scale_parameter_gradients(parameters: Iterable[DomainParameter], coefficient: Tensor) -> None:
    for item in parameters:
        if item.parameter.grad is not None:
            item.parameter.grad.mul_(coefficient.to(item.parameter.grad.device))


def unique_process_groups(parameters: Iterable[DomainParameter]) -> list[ParallelGroup]:
    groups: list[ParallelGroup] = []
    seen: set[tuple[tuple[int, ...], int]] = set()
    for item in parameters:
        key = (tuple(item.group.ranks), id(item.group.process_group))
        if key not in seen:
            seen.add(key)
            groups.append(item.group)
    return groups


def config_value(config: object, name: str, default: Any = None) -> Any:
    return getattr(config, name, default)


def build_adamw(parameters: Iterable[nn.Parameter], config: object) -> torch.optim.AdamW:
    name = str(config_value(config, "name", "adamw")).lower()
    if name != "adamw":
        raise ValueError(f"only AdamW is supported in phase one, got {name!r}")
    return torch.optim.AdamW(
        parameters,
        lr=float(config_value(config, "lr", 3.0e-4)),
        betas=tuple(config_value(config, "betas", (0.9, 0.95))),
        eps=float(config_value(config, "eps", 1.0e-8)),
        weight_decay=float(config_value(config, "weight_decay", 0.0)),
    )


def model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")
