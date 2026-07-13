from __future__ import annotations

import pytest

from nano_megatron.parallel import (
    GroupKey,
    ParameterDomain,
    ParameterDomainRegistry,
    ParameterPlacement,
)


def test_parameter_domains_have_distinct_default_replica_groups() -> None:
    registry = ParameterDomainRegistry()
    dense = object()
    expert = object()

    registry.register(dense, ParameterDomain.DENSE, tensor_sharded=True, name="attention.weight")
    registry.register(expert, "expert", name="experts.0.weight")

    assert registry.placement(dense) == ParameterPlacement(
        domain=ParameterDomain.DENSE,
        tensor_sharded=True,
        replica_group=GroupKey.DENSE_REPLICA,
    )
    assert registry.replica_group(expert) is GroupKey.EXPERT_REPLICA
    assert registry.parameters(ParameterDomain.DENSE) == (dense,)
    assert registry.grouped_parameters()[ParameterDomain.EXPERT] == (expert,)


def test_registration_is_identity_based_and_idempotent() -> None:
    registry = ParameterDomainRegistry()
    parameter = object()

    first = registry.register(parameter, "dense")
    second = registry.register(parameter, ParameterDomain.DENSE)

    assert first is second
    assert parameter in registry
    assert len(registry) == 1


def test_conflicting_registration_is_rejected() -> None:
    registry = ParameterDomainRegistry()
    parameter = object()
    registry.register(parameter, "dense")

    with pytest.raises(ValueError, match="already registered"):
        registry.register(parameter, "expert")


class TinyModule:
    def __init__(self) -> None:
        self.weight = object()
        self.bias = object()

    def named_parameters(self, recurse: bool = True):
        assert recurse
        yield "weight", self.weight
        yield "bias", self.bias


def test_register_module_and_complete_validation() -> None:
    module = TinyModule()
    registry = ParameterDomainRegistry()
    registered = registry.register_module(module, domain="dense", name_prefix="linear")

    assert registered == (module.weight, module.bias)
    assert registry.name(module.weight) == "linear.weight"
    registry.validate_complete(registered)

    with pytest.raises(ValueError, match="1 model parameter"):
        registry.validate_complete((*registered, object()))
