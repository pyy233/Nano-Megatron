"""Explicit parameter ownership domains for dense and expert replication."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .group_plan import GroupKey, GroupKeyLike, normalize_group_key


class ParameterDomain(StrEnum):
    DENSE = "dense"
    EXPERT = "expert"

    @classmethod
    def parse(cls, value: ParameterDomain | str) -> ParameterDomain:
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as error:
            choices = ", ".join(domain.value for domain in cls)
            raise ValueError(
                f"unknown parameter domain {value!r}; expected one of: {choices}"
            ) from error


_DEFAULT_REPLICA_GROUP = {
    ParameterDomain.DENSE: GroupKey.DENSE_REPLICA,
    ParameterDomain.EXPERT: GroupKey.EXPERT_REPLICA,
}


@dataclass(frozen=True, slots=True)
class ParameterPlacement:
    domain: ParameterDomain
    tensor_sharded: bool
    replica_group: GroupKeyLike
    tensor_shard_dim: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", ParameterDomain.parse(self.domain))
        object.__setattr__(self, "replica_group", normalize_group_key(self.replica_group))
        if not isinstance(self.tensor_sharded, bool):
            raise TypeError("tensor_sharded must be a boolean")
        if self.tensor_shard_dim is not None:
            if (
                isinstance(self.tensor_shard_dim, bool)
                or not isinstance(self.tensor_shard_dim, int)
                or self.tensor_shard_dim < 0
            ):
                raise ValueError("tensor_shard_dim must be a non-negative integer or null")
            if not self.tensor_sharded:
                raise ValueError("tensor_shard_dim requires tensor_sharded=True")

    @classmethod
    def for_domain(
        cls,
        domain: ParameterDomain | str,
        *,
        tensor_sharded: bool = False,
        replica_group: GroupKeyLike | None = None,
        tensor_shard_dim: int | None = None,
    ) -> ParameterPlacement:
        parsed = ParameterDomain.parse(domain)
        return cls(
            domain=parsed,
            tensor_sharded=tensor_sharded,
            replica_group=_DEFAULT_REPLICA_GROUP[parsed]
            if replica_group is None
            else replica_group,
            tensor_shard_dim=tensor_shard_dim,
        )


@dataclass(slots=True)
class _Registration:
    parameter: Any
    placement: ParameterPlacement
    name: str | None


class ParameterDomainRegistry:
    """Identity-based registry; never mutates ``torch.nn.Parameter`` objects."""

    def __init__(self) -> None:
        self._registrations: dict[int, _Registration] = {}

    def register(
        self,
        parameter: Any,
        placement: ParameterPlacement | ParameterDomain | str | None = None,
        *,
        domain: ParameterDomain | str | None = None,
        tensor_sharded: bool = False,
        replica_group: GroupKeyLike | None = None,
        tensor_shard_dim: int | None = None,
        name: str | None = None,
    ) -> ParameterPlacement:
        if parameter is None:
            raise TypeError("cannot register None as a parameter")
        if isinstance(placement, ParameterPlacement):
            if (
                domain is not None
                or tensor_sharded
                or replica_group is not None
                or tensor_shard_dim is not None
            ):
                raise ValueError(
                    "domain/tensor_sharded/replica_group cannot accompany an explicit placement"
                )
            resolved = placement
        else:
            if placement is not None and domain is not None:
                raise ValueError("parameter domain was supplied both positionally and by keyword")
            resolved_domain = domain if domain is not None else placement
            if resolved_domain is None:
                resolved_domain = ParameterDomain.DENSE
            resolved = ParameterPlacement.for_domain(
                resolved_domain,
                tensor_sharded=tensor_sharded,
                replica_group=replica_group,
                tensor_shard_dim=tensor_shard_dim,
            )

        identity = id(parameter)
        existing = self._registrations.get(identity)
        if existing is not None:
            if existing.parameter is not parameter:
                # Defensive guard against an id being reused after an external
                # mutation of internal state.  Strong references normally make
                # this impossible.
                raise RuntimeError("parameter identity collision in domain registry")
            if existing.placement != resolved:
                raise ValueError(
                    f"parameter is already registered with {existing.placement}, "
                    f"cannot use {resolved}"
                )
            if name is not None and existing.name not in (None, name):
                raise ValueError(
                    f"parameter is already registered as {existing.name!r}, "
                    f"cannot rename to {name!r}"
                )
            if existing.name is None and name is not None:
                existing.name = name
            return existing.placement

        self._registrations[identity] = _Registration(parameter, resolved, name)
        return resolved

    def register_module(
        self,
        module: Any,
        placement: ParameterPlacement | ParameterDomain | str | None = None,
        *,
        domain: ParameterDomain | str | None = None,
        tensor_sharded: bool = False,
        replica_group: GroupKeyLike | None = None,
        tensor_shard_dim: int | None = None,
        recurse: bool = True,
        name_prefix: str | None = None,
    ) -> tuple[Any, ...]:
        if not hasattr(module, "named_parameters"):
            raise TypeError("register_module expects an object with named_parameters()")
        registered: list[Any] = []
        for name, parameter in module.named_parameters(recurse=recurse):
            qualified_name = f"{name_prefix}.{name}" if name_prefix else name
            self.register(
                parameter,
                placement,
                domain=domain,
                tensor_sharded=tensor_sharded,
                replica_group=replica_group,
                tensor_shard_dim=tensor_shard_dim,
                name=qualified_name,
            )
            registered.append(parameter)
        return tuple(registered)

    def unregister(self, parameter: Any) -> None:
        identity = id(parameter)
        registration = self._registrations.get(identity)
        if registration is None or registration.parameter is not parameter:
            raise KeyError("parameter is not registered")
        del self._registrations[identity]

    def placement(self, parameter: Any) -> ParameterPlacement:
        registration = self._registration(parameter)
        return registration.placement

    def get(self, parameter: Any, default: Any = None) -> ParameterPlacement | Any:
        registration = self._registrations.get(id(parameter))
        if registration is None or registration.parameter is not parameter:
            return default
        return registration.placement

    def domain(self, parameter: Any) -> ParameterDomain:
        return self.placement(parameter).domain

    def replica_group(self, parameter: Any) -> GroupKeyLike:
        return self.placement(parameter).replica_group

    def name(self, parameter: Any) -> str | None:
        return self._registration(parameter).name

    def parameters(self, domain: ParameterDomain | str | None = None) -> tuple[Any, ...]:
        parsed = None if domain is None else ParameterDomain.parse(domain)
        return tuple(
            registration.parameter
            for registration in self._registrations.values()
            if parsed is None or registration.placement.domain is parsed
        )

    def grouped_parameters(self) -> dict[ParameterDomain, tuple[Any, ...]]:
        return {domain: self.parameters(domain) for domain in ParameterDomain}

    def items(self) -> Iterator[tuple[Any, ParameterPlacement]]:
        for registration in self._registrations.values():
            yield registration.parameter, registration.placement

    def validate_complete(self, parameters: Iterable[Any]) -> None:
        missing = [parameter for parameter in parameters if parameter not in self]
        if missing:
            raise ValueError(f"{len(missing)} model parameter(s) have no registered domain")

    def _registration(self, parameter: Any) -> _Registration:
        registration = self._registrations.get(id(parameter))
        if registration is None or registration.parameter is not parameter:
            raise KeyError("parameter is not registered")
        return registration

    def __contains__(self, parameter: object) -> bool:
        registration = self._registrations.get(id(parameter))
        return registration is not None and registration.parameter is parameter

    def __len__(self) -> int:
        return len(self._registrations)


def default_replica_group(domain: ParameterDomain | str) -> GroupKey:
    return _DEFAULT_REPLICA_GROUP[ParameterDomain.parse(domain)]
