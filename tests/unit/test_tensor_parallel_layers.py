from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from nano_megatron.nn.kernels import TorchKernelBackend
from nano_megatron.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelCrossEntropy,
    VocabParallelEmbedding,
    VocabParallelLinear,
    copy_to_tensor_parallel_region,
    gather_from_sequence_parallel_region,
    reduce_from_tensor_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)


@dataclass(frozen=True)
class FakeGroup:
    rank: int = 0
    size: int = 1
    process_group: object | None = None


@dataclass(frozen=True)
class FakeParallel:
    tp: FakeGroup = FakeGroup()
    sequence_parallel: bool = False


def test_size_one_collectives_are_autograd_identity() -> None:
    group = FakeGroup()
    x = torch.randn(2, 4, 3, requires_grad=True)
    y = copy_to_tensor_parallel_region(x, group)
    y = reduce_from_tensor_parallel_region(y, group)
    y = gather_from_sequence_parallel_region(y, group)
    y = reduce_scatter_to_sequence_parallel_region(y, group)
    y.square().sum().backward()
    torch.testing.assert_close(y, x)
    torch.testing.assert_close(x.grad, 2 * x.detach())


def test_multi_rank_group_requires_initialized_distributed() -> None:
    x = torch.ones(2, requires_grad=True)
    with pytest.raises(RuntimeError, match="torch.distributed"):
        reduce_from_tensor_parallel_region(x, FakeGroup(rank=0, size=2))


def test_column_parallel_size_one_matches_linear() -> None:
    kernels = TorchKernelBackend()
    layer = ColumnParallelLinear(
        4,
        6,
        parallel=FakeParallel(),
        kernels=kernels,
        bias=True,
    )
    reference = torch.nn.Linear(4, 6)
    reference.load_state_dict(layer.linear.state_dict())
    x = torch.randn(2, 3, 4, requires_grad=True)
    actual, returned_bias = layer(x)
    expected = reference(x)
    assert returned_bias is None
    torch.testing.assert_close(actual, expected)


def test_row_parallel_size_one_matches_linear() -> None:
    kernels = TorchKernelBackend()
    layer = RowParallelLinear(
        4,
        6,
        parallel=FakeParallel(),
        kernels=kernels,
        bias=True,
    )
    x = torch.randn(2, 3, 4)
    expected = F.linear(x, layer.weight, layer.bias)
    actual, returned_bias = layer(x)
    assert returned_bias is None
    torch.testing.assert_close(actual, expected)


def test_parallel_dimensions_must_be_divisible() -> None:
    kernels = TorchKernelBackend()
    parallel = FakeParallel(tp=FakeGroup(rank=0, size=2))
    with pytest.raises(ValueError, match="out_features"):
        ColumnParallelLinear(4, 5, parallel=parallel, kernels=kernels)
    with pytest.raises(ValueError, match="in_features"):
        RowParallelLinear(5, 4, parallel=parallel, kernels=kernels)


def test_vocab_parallel_embedding_and_tied_head_size_one() -> None:
    parallel = FakeParallel()
    embedding = VocabParallelEmbedding(8, 4, parallel=parallel)
    head = VocabParallelLinear(4, 8, parallel=parallel, weight=embedding.weight)
    input_ids = torch.tensor([[0, 3, 7]])
    hidden = embedding(input_ids)
    torch.testing.assert_close(hidden, F.embedding(input_ids, embedding.weight))
    torch.testing.assert_close(head(hidden), F.linear(hidden, embedding.weight))
    assert head.weight is embedding.weight


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_vocab_parallel_cross_entropy_matches_torch(reduction: str) -> None:
    torch.manual_seed(7)
    logits = torch.randn(2, 3, 7, dtype=torch.float64, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    targets = torch.tensor([[0, 3, 6], [2, -100, 1]])
    loss_fn = VocabParallelCrossEntropy(
        parallel=FakeParallel(),
        reduction=reduction,
        ignore_index=-100,
        original_vocab_size=7,
    )
    actual = loss_fn(logits, targets)
    expected = F.cross_entropy(
        reference_logits.reshape(-1, 7),
        targets.reshape(-1),
        reduction=reduction,
        ignore_index=-100,
    )
    if reduction == "none":
        expected = expected.view_as(targets)
        grad = torch.randn_like(actual)
        actual.backward(grad)
        expected.backward(grad)
    else:
        actual.backward()
        expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits.grad, reference_logits.grad)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_vocab_parallel_cross_entropy_uses_float32_for_low_precision_logits(
    dtype: torch.dtype,
) -> None:
    vocab_size = 70_000
    logits = torch.zeros(1, vocab_size, dtype=dtype, requires_grad=True)
    reference_logits = logits.detach().float().requires_grad_(True)
    targets = torch.tensor([0])
    loss_fn = VocabParallelCrossEntropy(
        parallel=FakeParallel(),
        reduction="mean",
    )

    actual = loss_fn(logits, targets)
    expected = F.cross_entropy(reference_logits, targets)
    actual.backward()
    expected.backward()

    assert actual.dtype is torch.float32
    assert torch.isfinite(actual)
    torch.testing.assert_close(actual, expected)
    assert logits.grad is not None and logits.grad.dtype is dtype
    torch.testing.assert_close(
        logits.grad.float(),
        reference_logits.grad,
        rtol=5.0e-3,
        atol=1.0e-7,
    )


def test_vocab_parallel_cross_entropy_rejects_padded_target() -> None:
    loss_fn = VocabParallelCrossEntropy(
        parallel=FakeParallel(),
        original_vocab_size=7,
    )
    with pytest.raises(ValueError, match="original"):
        loss_fn(torch.randn(1, 8), torch.tensor([7]))
