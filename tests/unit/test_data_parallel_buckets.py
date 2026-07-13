from __future__ import annotations

import torch
from torch import nn

from nano_megatron.config import DistributedConfig, ParallelConfig
from nano_megatron.data_parallel._common import collect_domain_parameters
from nano_megatron.data_parallel.buckets import build_flat_buckets
from nano_megatron.data_parallel.offload import ActivationOffloader, OffloadPolicy
from nano_megatron.distributed import DistributedRuntime
from nano_megatron.parallel import ParallelContext, ParameterDomain, ParameterDomainRegistry


def test_flat_bucket_round_trip_and_padding() -> None:
    runtime = DistributedRuntime(DistributedConfig(backend="gloo", device="cpu")).initialize()
    parallel = ParallelContext.create(runtime, ParallelConfig(data=1))
    try:
        model = nn.Linear(3, 2)
        registry = ParameterDomainRegistry()
        registry.register_module(model, ParameterDomain.DENSE)
        buckets = build_flat_buckets(
            collect_domain_parameters(model, registry, parallel),
            bucket_bytes=10_000,
        )
        assert len(buckets) == 1
        bucket = buckets[0]
        flat = bucket.pack_parameters()
        assert flat.numel() == bucket.padded_numel
        with torch.no_grad():
            flat[: bucket.numel].add_(2.0)
            bucket.unpack_parameters(flat)
        repacked = bucket.pack_parameters()
        torch.testing.assert_close(repacked[: bucket.numel], flat[: bucket.numel])
    finally:
        parallel.close()
        runtime.close()


def test_activation_offloader_is_identity_safe_on_cpu() -> None:
    offloader = ActivationOffloader(OffloadPolicy(activations=True))
    value = torch.tensor([2.0, -3.0], requires_grad=True)
    with offloader.context():
        loss = value.square().sum()
    loss.backward()
    torch.testing.assert_close(value.grad, 2 * value.detach())
