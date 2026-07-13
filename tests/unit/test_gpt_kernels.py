from __future__ import annotations

import torch

from nano_megatron.nn import (
    RMSNorm,
    RotaryEmbedding,
    activation_checkpoint,
    parallel_dropout,
)
from nano_megatron.nn.kernels import TorchKernelBackend
from nano_megatron.nn.rotary import apply_rotary_pos_emb
from nano_megatron.parallel import ParallelCoordinate, ParallelRNG


def test_rms_norm_matches_reference() -> None:
    torch.manual_seed(11)
    norm = RMSNorm(8, eps=1.0e-5).double()
    x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    expected = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1.0e-5)
    torch.testing.assert_close(norm(x), expected)


def test_rope_preserves_query_and_key_norms() -> None:
    rope = RotaryEmbedding(8)
    positions = torch.arange(6)
    cos, sin = rope(positions)
    q = torch.randn(2, 4, 6, 8)
    k = torch.randn(2, 2, 6, 8)
    rotated_q, rotated_k = apply_rotary_pos_emb(q, k, cos, sin)
    torch.testing.assert_close(rotated_q.norm(dim=-1), q.norm(dim=-1))
    torch.testing.assert_close(rotated_k.norm(dim=-1), k.norm(dim=-1))


def test_torch_attention_is_causal() -> None:
    kernels = TorchKernelBackend()
    torch.manual_seed(3)
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    changed_v = v.clone()
    changed_v[..., 1:, :] += 100.0
    output = kernels.local_attention(q, k, v, causal=True)
    changed_output = kernels.local_attention(q, k, changed_v, causal=True)
    torch.testing.assert_close(output[..., 0, :], changed_output[..., 0, :])


def test_torch_attention_expands_grouped_query_heads() -> None:
    kernels = TorchKernelBackend()
    q = torch.randn(1, 4, 3, 8)
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 3, 8)
    actual = kernels.local_attention(q, k, v)
    expected = kernels.local_attention(
        q,
        k.repeat_interleave(2, dim=1),
        v.repeat_interleave(2, dim=1),
    )
    torch.testing.assert_close(actual, expected)


def test_non_reentrant_activation_checkpoint_matches_eager_gradient() -> None:
    calls = 0

    def function(x: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.sin(x).square() * x

    x = torch.randn(9, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    actual = activation_checkpoint(function, x).sum()
    expected = (torch.sin(reference_x).square() * reference_x).sum()
    actual.backward()
    expected.backward()
    assert calls >= 2
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, reference_x.grad)


def test_checkpoint_recompute_reuses_explicit_parallel_dropout_mask() -> None:
    calls = 0
    rng = ParallelRNG(41, ParallelCoordinate(0, 0, 0, 0, 0))
    reference_rng = ParallelRNG(41, ParallelCoordinate(0, 0, 0, 0, 0))

    def function(x: torch.Tensor, tracker: ParallelRNG) -> torch.Tensor:
        nonlocal calls
        calls += 1
        dropped = parallel_dropout(
            x,
            0.4,
            training=True,
            rng=tracker,
            layer=2,
            microbatch=7,
            global_token_offset=64,
            op="checkpoint-test",
        )
        return dropped.square()

    x = torch.randn(32, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    actual = activation_checkpoint(lambda value: function(value, rng), x, rng=rng).sum()
    expected = function(reference_x, reference_rng).sum()
    actual.backward()
    expected.backward()

    assert calls >= 3
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, reference_x.grad)
