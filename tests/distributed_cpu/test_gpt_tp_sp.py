from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from nano_megatron.config import GPTConfig
from nano_megatron.models.gpt import GPTModel
from nano_megatron.nn import RMSNorm
from nano_megatron.nn.kernels import TorchKernelBackend
from nano_megatron.parallel import GroupKey, ParallelGroup
from nano_megatron.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    VocabParallelLinear,
)


@dataclass(frozen=True)
class _LocalGroup:
    rank: int = 0
    size: int = 1
    process_group: object | None = None


@dataclass(frozen=True)
class _Parallel:
    tp: object
    sequence_parallel: bool
    cp: object = _LocalGroup()
    pp: object = _LocalGroup()


def _copy_reference_to_tp(
    reference: GPTModel,
    target: GPTModel,
    rank: int,
    world_size: int,
) -> list[tuple[nn.Parameter, nn.Parameter, int | None]]:
    reference_modules = dict(reference.named_modules())
    mappings: list[tuple[nn.Parameter, nn.Parameter, int | None]] = []
    recorded: set[int] = set()

    def copy_parameter(
        target_parameter: nn.Parameter | None,
        reference_parameter: nn.Parameter | None,
        shard_dim: int | None,
    ) -> None:
        if target_parameter is None or reference_parameter is None:
            assert target_parameter is reference_parameter
            return
        if id(target_parameter) in recorded:
            return
        value = reference_parameter
        if shard_dim is not None:
            value = reference_parameter.chunk(world_size, dim=shard_dim)[rank]
        with torch.no_grad():
            target_parameter.copy_(value)
        recorded.add(id(target_parameter))
        mappings.append((target_parameter, reference_parameter, shard_dim))

    for name, module in target.named_modules():
        reference_module = reference_modules[name]
        if isinstance(module, ColumnParallelLinear):
            assert isinstance(reference_module, ColumnParallelLinear)
            copy_parameter(module.weight, reference_module.weight, 0)
            copy_parameter(module.bias, reference_module.bias, 0)
        elif isinstance(module, RowParallelLinear):
            assert isinstance(reference_module, RowParallelLinear)
            copy_parameter(module.weight, reference_module.weight, 1)
            copy_parameter(module.bias, reference_module.bias, None)
        elif isinstance(module, (VocabParallelEmbedding, VocabParallelLinear)):
            assert isinstance(reference_module, type(module))
            copy_parameter(module.weight, reference_module.weight, 0)
            if isinstance(module, VocabParallelLinear):
                copy_parameter(module.bias, reference_module.bias, 0)
        elif isinstance(module, RMSNorm):
            assert isinstance(reference_module, RMSNorm)
            copy_parameter(module.weight, reference_module.weight, None)

    assert len(recorded) == len(tuple(target.parameters()))
    return mappings


def _gpt_tp_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        tp_group = ParallelGroup(
            key=GroupKey.TP,
            ranks=tuple(range(world_size)),
            process_group=dist.group.WORLD,
            rank=rank,
            size=world_size,
            backend="gloo",
        )
        config = GPTConfig(
            layers=1,
            hidden_size=8,
            ffn_hidden_size=12,
            heads=2,
            kv_heads=2,
            seq_length=4,
            vocab_size=16,
            dropout=0.0,
            tie_embeddings=True,
        )
        torch.manual_seed(29)
        reference = GPTModel(
            config,
            parallel=_Parallel(tp=_LocalGroup(), sequence_parallel=False),
            kernels=TorchKernelBackend(),
        ).double()
        target = GPTModel(
            config,
            parallel=_Parallel(tp=tp_group, sequence_parallel=True),
            kernels=TorchKernelBackend(),
        ).double()
        mappings = _copy_reference_to_tp(reference, target, rank, world_size)

        input_ids = torch.tensor([[0, 3, 7, 12], [15, 2, 9, 4]])
        labels = torch.tensor([[3, 7, 12, 1], [2, 9, 4, 6]])
        reference_output = reference(input_ids, labels=labels)
        target_output = target(input_ids, labels=labels)
        assert reference_output.logits is not None and target_output.logits is not None
        assert reference_output.loss is not None and target_output.loss is not None
        torch.testing.assert_close(
            target_output.hidden_states,
            reference_output.hidden_states.chunk(world_size, dim=1)[rank],
            atol=1.0e-9,
            rtol=1.0e-7,
        )
        torch.testing.assert_close(
            target_output.logits,
            reference_output.logits.chunk(world_size, dim=-1)[rank],
            atol=1.0e-9,
            rtol=1.0e-7,
        )
        torch.testing.assert_close(
            target_output.loss,
            reference_output.loss,
            atol=1.0e-9,
            rtol=1.0e-7,
        )

        reference_output.loss.backward()
        target_output.loss.backward()
        for target_parameter, reference_parameter, shard_dim in mappings:
            assert target_parameter.grad is not None
            assert reference_parameter.grad is not None
            expected_gradient = reference_parameter.grad
            if shard_dim is not None:
                expected_gradient = expected_gradient.chunk(world_size, dim=shard_dim)[rank]
            torch.testing.assert_close(
                target_parameter.grad,
                expected_gradient,
                atol=1.0e-8,
                rtol=1.0e-6,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
def test_tp2_sp_gpt_matches_single_rank_reference_on_gloo() -> None:
    world_size = 2
    handle, init_file = tempfile.mkstemp(prefix="nano-megatron-gpt-gloo-")
    os.close(handle)
    os.unlink(init_file)
    try:
        mp.spawn(
            _gpt_tp_worker,
            args=(world_size, init_file),
            nprocs=world_size,
            join=True,
        )
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)
