from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn

from nano_megatron.parallel import GroupKey, ParallelGroup
from nano_megatron.pipeline_parallel import (
    GPipeSchedule,
    OneForwardOneBackwardSchedule,
    P2PCommunicator,
)


@dataclass(frozen=True)
class _Parallel:
    pp: ParallelGroup

    def is_pipeline_first_stage(self) -> bool:
        return self.pp.rank == 0

    def is_pipeline_last_stage(self) -> bool:
        return self.pp.rank == self.pp.size - 1

    def pipeline_prev_rank(self) -> int | None:
        return None if self.is_pipeline_first_stage() else self.pp.global_rank_at(self.pp.rank - 1)

    def pipeline_next_rank(self) -> int | None:
        return None if self.is_pipeline_last_stage() else self.pp.global_rank_at(self.pp.rank + 1)


class _ScalarStage(nn.Module):
    def __init__(self, rank: int, world_size: int, initial_weight: float) -> None:
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.weight = nn.Parameter(torch.tensor(initial_weight, dtype=torch.float64))

    def forward(self, hidden_states: Tensor | None, batch) -> Tensor:
        if self.rank == 0:
            assert hidden_states is None
            return batch["x"] * self.weight
        assert hidden_states is not None
        output = hidden_states * self.weight
        if self.rank != self.world_size - 1:
            return output
        return (output - batch["target"]).square().mean()


def _expected_gradients(microbatches: list[dict[str, Tensor]], weights: tuple[float, ...]):
    parameters = [torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in weights]
    losses = []
    for batch in microbatches:
        output = batch["x"]
        for parameter in parameters:
            output = output * parameter
        losses.append((output - batch["target"]).square().mean())
    (sum(losses) / len(losses)).backward()
    return tuple(float(parameter.grad) for parameter in parameters)


def _schedule_worker(
    rank: int,
    world_size: int,
    init_file: str,
    schedule_name: str,
    num_microbatches: int,
    wire_dtype_name: str,
) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        group = ParallelGroup(
            key=GroupKey.PP,
            ranks=tuple(range(world_size)),
            process_group=dist.group.WORLD,
            rank=rank,
            size=world_size,
            backend="gloo",
        )
        parallel = _Parallel(group)
        communicator = P2PCommunicator(
            parallel,
            activation_shape=(1, 2, 3),
            activation_dtype=getattr(torch, wire_dtype_name),
            device="cpu",
        )
        weights = (1.1, 0.9, 1.3)
        stage = _ScalarStage(rank, world_size, weights[rank])
        microbatches = [
            {
                "x": torch.full((1, 2, 3), 0.25 + index, dtype=torch.float64),
                "target": torch.full((1, 2, 3), -0.5 + index / 3, dtype=torch.float64),
            }
            for index in range(num_microbatches)
        ]
        expected = _expected_gradients(microbatches, weights)
        schedule = (
            GPipeSchedule(parallel, communicator)
            if schedule_name == "gpipe"
            else OneForwardOneBackwardSchedule(parallel, communicator)
        )
        schedule.forward_backward(stage=stage, microbatches=microbatches)
        assert stage.weight.grad is not None
        torch.testing.assert_close(
            stage.weight.grad,
            torch.tensor(expected[rank], dtype=torch.float64),
            atol=2.0e-5 if wire_dtype_name == "float32" else 1.0e-10,
            rtol=2.0e-5 if wire_dtype_name == "float32" else 1.0e-9,
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("schedule_name", "num_microbatches", "wire_dtype_name"),
    [
        ("gpipe", 4, "float64"),
        ("1f1b", 4, "float64"),
        ("1f1b", 1, "float64"),
        ("1f1b", 4, "float32"),
    ],
)
def test_pipeline_schedule_gradients_match_eager_on_three_gloo_processes(
    schedule_name: str,
    num_microbatches: int,
    wire_dtype_name: str,
) -> None:
    world_size = 3
    handle, init_file = tempfile.mkstemp(prefix=f"nano-megatron-{schedule_name}-")
    os.close(handle)
    os.unlink(init_file)
    try:
        mp.spawn(
            _schedule_worker,
            args=(
                world_size,
                init_file,
                schedule_name,
                num_microbatches,
                wire_dtype_name,
            ),
            nprocs=world_size,
            join=True,
        )
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)
