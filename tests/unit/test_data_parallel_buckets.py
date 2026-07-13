from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nano_megatron.config import DistributedConfig, ParallelConfig
from nano_megatron.data_parallel._common import DomainParameter, collect_domain_parameters
from nano_megatron.data_parallel.buckets import (
    BucketGradientReducer,
    FlatBucket,
    build_flat_buckets,
)
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


class _DelayedAllReduce:
    def __init__(self, tensor: torch.Tensor, world_size: int) -> None:
        self.tensor = tensor
        self.world_size = world_size
        self.wait_calls = 0

    def wait(self) -> None:
        self.wait_calls += 1
        self.tensor.mul_(self.world_size)


def _overlap_buckets(
    parameters: list[nn.Parameter],
    *,
    tied_parameter: nn.Parameter | None = None,
    backend: str = "nccl",
) -> list[FlatBucket]:
    group = SimpleNamespace(
        size=2,
        rank=0,
        ranks=(0, 1),
        process_group=object(),
        backend=backend,
    )
    buckets = []
    for index, parameter in enumerate(parameters):
        buckets.append(
            FlatBucket(
                index=index,
                domain=ParameterDomain.DENSE,
                group=group,
                parameters=[
                    DomainParameter(
                        name=f"parameter_{index}",
                        parameter=parameter,
                        domain=ParameterDomain.DENSE,
                        group=group,
                        tensor_sharded=False,
                        tied_source_rank=0 if parameter is tied_parameter else None,
                    )
                ],
            )
        )
    return buckets


def _fake_delayed_all_reduce(
    monkeypatch: pytest.MonkeyPatch,
) -> list[_DelayedAllReduce]:
    works: list[_DelayedAllReduce] = []

    def all_reduce(tensor, *, op, group, async_op=False):
        del op, group
        assert async_op
        work = _DelayedAllReduce(tensor, world_size=2)
        works.append(work)
        return work

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    return works


def test_bucket_overlap_starts_during_backward_and_waits_only_at_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    works = _fake_delayed_all_reduce(monkeypatch)
    first = nn.Linear(3, 4, bias=False)
    second = nn.Linear(4, 1, bias=False)
    buckets = _overlap_buckets([first.weight, second.weight])
    final_backward = True
    reducer = BucketGradientReducer(
        buckets,
        partition_gradients=False,
        reduction_dtype=torch.float32,
        overlap=True,
        is_final_backward=lambda: final_backward,
    )

    second(torch.tanh(first(torch.randn(2, 3)))).square().mean().backward()
    expected = [parameter.grad.detach().clone() for parameter in (first.weight, second.weight)]

    assert reducer.start_count == 2
    assert reducer.wait_count == 0
    assert [request.bucket.index for request in reducer.requests] == [1, 0]
    assert not any(request.completed for request in reducer.requests)
    assert all(work.wait_calls == 0 for work in works)

    reducer.finalize()
    reducer.finalize()

    assert reducer.wait_count == 2
    assert all(request.completed for request in reducer.requests)
    assert all(work.wait_calls == 1 for work in works)
    for parameter, gradient in zip((first.weight, second.weight), expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient)


def test_bucket_overlap_only_starts_on_the_final_microbatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_delayed_all_reduce(monkeypatch)
    layer = nn.Linear(3, 2, bias=False)
    final_backward = False
    reducer = BucketGradientReducer(
        _overlap_buckets([layer.weight]),
        partition_gradients=False,
        reduction_dtype=torch.float32,
        overlap=True,
        is_final_backward=lambda: final_backward,
    )

    layer(torch.randn(2, 3)).square().mean().backward()
    assert reducer.start_count == 0

    final_backward = True
    layer(torch.randn(2, 3)).square().mean().backward()
    assert reducer.start_count == 1
    assert reducer.wait_count == 0

    reducer.finalize()
    assert reducer.wait_count == 1


def test_tied_bucket_reads_the_post_pipeline_gradient_at_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_delayed_all_reduce(monkeypatch)
    parameter = nn.Parameter(torch.tensor([2.0, -1.0]))
    reducer = BucketGradientReducer(
        _overlap_buckets([parameter], tied_parameter=parameter),
        partition_gradients=False,
        reduction_dtype=torch.float32,
        overlap=True,
        is_final_backward=lambda: True,
    )

    parameter.square().sum().backward()
    assert reducer.start_count == 0
    with torch.no_grad():
        parameter.grad.add_(torch.tensor([3.0, 4.0]))
    expected = parameter.grad.detach().clone()

    reducer.finalize()

    assert reducer.start_count == 1
    torch.testing.assert_close(parameter.grad, expected)


def test_finalize_starts_a_bucket_with_an_unused_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_delayed_all_reduce(monkeypatch)
    unused = nn.Parameter(torch.tensor([1.0]))
    used = nn.Parameter(torch.tensor([2.0]))
    reducer = BucketGradientReducer(
        _overlap_buckets([unused, used]),
        partition_gradients=False,
        reduction_dtype=torch.float32,
        overlap=True,
        is_final_backward=lambda: True,
    )

    used.square().backward()
    assert reducer.start_count == 1
    assert reducer.requests[0].bucket.index == 1

    reducer.finalize()

    assert reducer.start_count == 2
    assert reducer.wait_count == 2
    torch.testing.assert_close(unused.grad, torch.zeros_like(unused))


@pytest.mark.parametrize(
    ("backend", "expected_collective"),
    [("gloo", "all_reduce"), ("nccl", "reduce_scatter")],
)
def test_partitioned_overlap_selects_a_safe_backend_collective(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    expected_collective: str,
) -> None:
    calls: list[str] = []

    class ReduceScatterWork:
        def __init__(self, output: torch.Tensor, input_: torch.Tensor) -> None:
            self.output = output
            self.input = input_

        def wait(self) -> None:
            self.output.copy_(self.input[: self.output.numel()] * 2)

    def all_reduce(tensor, *, op, group, async_op=False):
        del op, group
        assert async_op
        calls.append("all_reduce")
        return _DelayedAllReduce(tensor, world_size=2)

    def reduce_scatter(output, input_, *, op, group, async_op=False):
        del op, group
        assert async_op
        calls.append("reduce_scatter")
        return ReduceScatterWork(output, input_)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(torch.distributed, "reduce_scatter_tensor", reduce_scatter)
    parameter = nn.Parameter(torch.tensor([2.0, -1.0]))
    reducer = BucketGradientReducer(
        _overlap_buckets([parameter], backend=backend),
        partition_gradients=True,
        reduction_dtype=torch.float32,
        overlap=True,
        is_final_backward=lambda: True,
    )

    parameter.square().sum().backward()
    assert calls == [expected_collective]
    assert reducer.wait_count == 0

    reducer.finalize()

    assert reducer.buckets[0].local_gradient is not None
    torch.testing.assert_close(
        reducer.buckets[0].local_gradient,
        torch.tensor([4.0]),
    )
