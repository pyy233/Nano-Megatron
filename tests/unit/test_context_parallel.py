import pytest

torch = pytest.importorskip("torch")

from nano_megatron.context_parallel import (  # noqa: E402
    AllGatherContextParallelAttention,
    RingContextParallelAttention,
    build_context_parallel_attention,
)


class _LocalGroup:
    rank = 0
    size = 1
    process_group = None
    ranks = (0,)


class _MalformedMultiRankGroup:
    rank = 0
    size = 2
    process_group = None
    ranks = (0, 1)


_LOCAL_GROUP = _LocalGroup()


def test_context_parallel_reference_and_blockwise_match_single_rank() -> None:
    generator = torch.Generator().manual_seed(123)
    q = torch.randn(2, 3, 5, 4, generator=generator, requires_grad=True)
    k = torch.randn(2, 3, 5, 4, generator=generator, requires_grad=True)
    v = torch.randn(2, 3, 5, 4, generator=generator, requires_grad=True)

    reference = AllGatherContextParallelAttention(_LOCAL_GROUP)(q, k, v)
    blockwise = RingContextParallelAttention(_LOCAL_GROUP)(q, k, v)

    torch.testing.assert_close(blockwise, reference, atol=2e-5, rtol=2e-5)


def test_context_parallel_rejects_dropout() -> None:
    with pytest.raises(ValueError, match="dropout"):
        RingContextParallelAttention(_LOCAL_GROUP, dropout_p=0.1)


@pytest.mark.parametrize(
    "attention_type",
    [AllGatherContextParallelAttention, RingContextParallelAttention],
)
def test_context_parallel_constructor_requires_explicit_group(attention_type) -> None:
    with pytest.raises(TypeError, match="required positional argument"):
        attention_type()


@pytest.mark.parametrize(
    "attention_type",
    [AllGatherContextParallelAttention, RingContextParallelAttention],
)
def test_context_parallel_rejects_unmaterialized_multi_rank_group(attention_type) -> None:
    with pytest.raises(ValueError, match="materialized process_group"):
        attention_type(_MalformedMultiRankGroup())


def test_context_parallel_factory_is_disabled_for_size_one_group() -> None:
    parallel = type("Parallel", (), {"cp": type("Group", (), {"size": 1})()})()
    config = type("Config", (), {"backend": "ring", "dropout": 0.0})()
    assert build_context_parallel_attention(config, parallel) is None


def test_context_parallel_supports_grouped_query_attention() -> None:
    generator = torch.Generator().manual_seed(7)
    q = torch.randn(1, 4, 3, 8, generator=generator)
    k = torch.randn(1, 2, 3, 8, generator=generator)
    v = torch.randn(1, 2, 3, 8, generator=generator)
    reference = AllGatherContextParallelAttention(_LOCAL_GROUP)(q, k, v)
    blockwise = RingContextParallelAttention(_LOCAL_GROUP)(q, k, v)
    torch.testing.assert_close(blockwise, reference, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    "attention_type",
    [AllGatherContextParallelAttention, RingContextParallelAttention],
)
def test_context_parallel_rejects_mismatched_local_sequence_lengths(attention_type) -> None:
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 3, 8)
    with pytest.raises(ValueError, match="compatible"):
        attention_type(_LOCAL_GROUP)(q, k, v)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(("query_heads", "kv_heads"), [(3, 3), (4, 2)])
def test_context_parallel_blockwise_backward_matches_reference(
    causal: bool,
    query_heads: int,
    kv_heads: int,
) -> None:
    generator = torch.Generator().manual_seed(19)
    q = torch.randn(2, query_heads, 4, 3, generator=generator, requires_grad=True)
    k = torch.randn(2, kv_heads, 4, 3, generator=generator, requires_grad=True)
    v = torch.randn(2, kv_heads, 4, 3, generator=generator, requires_grad=True)
    ring_q, ring_k, ring_v = (
        tensor.detach().clone().requires_grad_() for tensor in (q, k, v)
    )

    reference = AllGatherContextParallelAttention(_LOCAL_GROUP)(q, k, v, causal=causal)
    blockwise = RingContextParallelAttention(_LOCAL_GROUP)(
        ring_q,
        ring_k,
        ring_v,
        causal=causal,
    )
    output_gradient = torch.randn(reference.shape, generator=generator)
    reference.backward(output_gradient)
    blockwise.backward(output_gradient)

    torch.testing.assert_close(blockwise, reference, atol=2e-5, rtol=2e-5)
    for actual, expected in zip(
        (ring_q.grad, ring_k.grad, ring_v.grad),
        (q.grad, k.grad, v.grad),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
