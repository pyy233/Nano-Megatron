from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from nano_megatron.nn.kernels import TorchKernelBackend
from nano_megatron.parallel import GroupKey, ParallelGroup
from nano_megatron.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelCrossEntropy,
)


def _sp_linear_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        group = ParallelGroup(
            key=GroupKey.TP,
            ranks=tuple(range(world_size)),
            process_group=dist.group.WORLD,
            rank=rank,
            size=world_size,
            backend="gloo",
        )
        kernels = TorchKernelBackend()
        column = ColumnParallelLinear(
            4,
            6,
            parallel=group,
            kernels=kernels,
            bias=True,
            sequence_parallel=True,
        ).double()
        row = RowParallelLinear(
            6,
            5,
            parallel=group,
            kernels=kernels,
            bias=True,
            input_is_parallel=True,
            sequence_parallel=True,
        ).double()

        full_column_weight = torch.arange(24, dtype=torch.float64).view(6, 4) / 31
        full_column_bias = torch.arange(6, dtype=torch.float64) / 17
        full_row_weight = torch.arange(30, dtype=torch.float64).view(5, 6) / 29
        full_row_bias = torch.arange(5, dtype=torch.float64) / 13
        with torch.no_grad():
            column.weight.copy_(full_column_weight.chunk(world_size, dim=0)[rank])
            assert column.bias is not None
            column.bias.copy_(full_column_bias.chunk(world_size, dim=0)[rank])
            row.weight.copy_(full_row_weight.chunk(world_size, dim=1)[rank])
            assert row.bias is not None
            row.bias.copy_(full_row_bias)

        full_input = torch.arange(16, dtype=torch.float64).view(1, 4, 4) / 19
        reference_input = full_input.detach().clone().requires_grad_(True)
        reference_hidden = F.linear(
            reference_input,
            full_column_weight,
            full_column_bias,
        )
        reference_output = F.linear(reference_hidden, full_row_weight, full_row_bias)
        reference_output.square().sum().backward()

        local_input = full_input.chunk(world_size, dim=1)[rank].detach().clone()
        local_input.requires_grad_(True)
        local_hidden, _ = column(local_input)
        local_output, _ = row(local_hidden)
        local_output.square().sum().backward()

        torch.testing.assert_close(
            local_output,
            reference_output.chunk(world_size, dim=1)[rank],
        )
        torch.testing.assert_close(
            local_input.grad,
            reference_input.grad.chunk(world_size, dim=1)[rank],
        )

        # Compare parameter gradients against an equivalent eager module.
        ref_column_weight = full_column_weight.detach().clone().requires_grad_(True)
        ref_column_bias = full_column_bias.detach().clone().requires_grad_(True)
        ref_row_weight = full_row_weight.detach().clone().requires_grad_(True)
        ref_row_bias = full_row_bias.detach().clone().requires_grad_(True)
        eager_hidden = F.linear(full_input, ref_column_weight, ref_column_bias)
        eager_output = F.linear(eager_hidden, ref_row_weight, ref_row_bias)
        eager_output.square().sum().backward()
        torch.testing.assert_close(
            column.weight.grad,
            ref_column_weight.grad.chunk(world_size, dim=0)[rank],
        )
        assert column.bias is not None and column.bias.grad is not None
        torch.testing.assert_close(
            column.bias.grad,
            ref_column_bias.grad.chunk(world_size, dim=0)[rank],
        )
        torch.testing.assert_close(
            row.weight.grad,
            ref_row_weight.grad.chunk(world_size, dim=1)[rank],
        )
        assert row.bias is not None and row.bias.grad is not None
        torch.testing.assert_close(row.bias.grad, ref_row_bias.grad)

        full_logits = torch.arange(64, dtype=torch.float64).view(2, 4, 8) / 23
        reference_logits = full_logits.detach().clone().requires_grad_(True)
        targets = torch.tensor([[0, 3, 7, 2], [6, 4, 1, 5]])
        reference_loss = F.cross_entropy(
            reference_logits.reshape(-1, 8),
            targets.reshape(-1),
        )
        reference_loss.backward()
        local_logits = full_logits.chunk(world_size, dim=-1)[rank].detach().clone()
        local_logits.requires_grad_(True)
        local_loss = VocabParallelCrossEntropy(
            parallel=group,
            reduction="mean",
            original_vocab_size=8,
        )(local_logits, targets)
        local_loss.backward()
        torch.testing.assert_close(local_loss, reference_loss)
        torch.testing.assert_close(
            local_logits.grad,
            reference_logits.grad.chunk(world_size, dim=-1)[rank],
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
def test_tp_sp_linear_pair_matches_eager_on_two_gloo_processes() -> None:
    world_size = 2
    handle, init_file = tempfile.mkstemp(prefix="nano-megatron-gloo-")
    os.close(handle)
    os.unlink(init_file)
    try:
        mp.spawn(
            _sp_linear_worker,
            args=(world_size, init_file),
            nprocs=world_size,
            join=True,
        )
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)
