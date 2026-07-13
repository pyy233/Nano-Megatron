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
    InterleavedOneForwardOneBackwardSchedule,
    OneForwardOneBackwardSchedule,
    P2PCommunicator,
    VirtualPipelineLayout,
)


@dataclass(frozen=True)
class _Parallel:
    pp: ParallelGroup

    def is_pipeline_first_stage(self) -> bool:
        return self.pp.rank == 0

    def is_pipeline_last_stage(self) -> bool:
        return self.pp.rank == self.pp.size - 1

    def pipeline_prev_rank(self) -> int | None:
        return None if self.is_pipeline_first_stage() else self.pp.rank - 1

    def pipeline_next_rank(self) -> int | None:
        return None if self.is_pipeline_last_stage() else self.pp.rank + 1


class _Chunk(nn.Module):
    def __init__(
        self,
        *,
        first: bool,
        last: bool,
        initial_weight: float,
    ) -> None:
        super().__init__()
        self.first = first
        self.last = last
        self.weight = nn.Parameter(torch.tensor(initial_weight, dtype=torch.float64))

    def forward(self, hidden_states: Tensor | None, batch) -> Tensor:
        if self.first:
            assert hidden_states is None
            hidden_states = batch["x"]
        else:
            assert hidden_states is not None
        output = hidden_states * self.weight
        if self.last:
            return (output - batch["target"]).square().mean()
        return output


class _Pipeline(nn.Module):
    def __init__(self, rank: int, layout: VirtualPipelineLayout, weights) -> None:
        super().__init__()
        chunks = []
        for chunk_id in range(layout.virtual_stages_per_rank):
            address = layout.address(rank, chunk_id)
            chunks.append(
                _Chunk(
                    first=layout.is_first(address),
                    last=layout.is_last(address),
                    initial_weight=weights[address.logical_stage],
                )
            )
        self.chunks = nn.ModuleList(chunks)

    def chunk(self, chunk_id: int) -> _Chunk:
        return self.chunks[chunk_id]

    def synchronize_tied_embedding_weights(self) -> None:
        return None

    def synchronize_tied_embedding_gradients(self) -> None:
        return None


def _expected_gradients(microbatches, weights):
    parameters = [torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in weights]
    losses = []
    for batch in microbatches:
        output = batch["x"]
        for parameter in parameters:
            output = output * parameter
        losses.append((output - batch["target"]).square().mean())
    (sum(losses) / len(losses)).backward()
    return tuple(parameter.grad.detach().clone() for parameter in parameters)


def _worker(
    rank: int,
    world_size: int,
    init_file: str,
    overlap_p2p: bool,
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
        layout = VirtualPipelineLayout(
            num_layers=4,
            pipeline_size=world_size,
            virtual_stages_per_rank=2,
        )
        communicator = P2PCommunicator(
            parallel,
            activation_shape=None,
            activation_dtype=torch.float64,
            device="cpu",
            dynamic_shapes=True,
        )
        weights = (1.1, 0.9, 1.3, 0.8)
        pipeline = _Pipeline(rank, layout, weights)
        sequence_lengths = (2, 5, 3, 7)
        microbatches = [
            {
                "x": torch.full(
                    (1, sequence, 3),
                    0.25 + index,
                    dtype=torch.float64,
                ),
                "target": torch.full(
                    (1, sequence, 3),
                    -0.5 + index / 3,
                    dtype=torch.float64,
                ),
            }
            for index, sequence in enumerate(sequence_lengths)
        ]
        expected = _expected_gradients(microbatches, weights)
        schedule = InterleavedOneForwardOneBackwardSchedule(
            parallel,
            layout,
            communicator,
            overlap_p2p=overlap_p2p,
        )
        for forward_count in (1, 3, len(microbatches)):
            forward_batches = microbatches[:forward_count]
            forward_only = schedule.forward_backward(
                stage=pipeline,
                microbatches=forward_batches,
                forward_only=True,
            )
            assert len(forward_only.losses) == (forward_count if rank == 1 else 0)
            if rank == 1:
                reference_losses = []
                for batch in forward_batches:
                    output = batch["x"]
                    for weight in weights:
                        output = output * weight
                    reference_losses.append((output - batch["target"]).square().mean())
                torch.testing.assert_close(
                    sum(forward_only.losses),
                    sum(reference_losses) / forward_count,
                    atol=1.0e-10,
                    rtol=1.0e-9,
                )
        schedule.forward_backward(stage=pipeline, microbatches=microbatches)

        for chunk_id, chunk in enumerate(pipeline.chunks):
            logical_stage = layout.address(rank, chunk_id).logical_stage
            assert chunk.weight.grad is not None
            torch.testing.assert_close(
                chunk.weight.grad,
                expected[logical_stage],
                atol=1.0e-10,
                rtol=1.0e-9,
            )
    finally:
        dist.destroy_process_group()


def _dynamic_schedule_worker(
    rank: int,
    world_size: int,
    init_file: str,
    schedule_name: str,
    overlap_p2p: bool,
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
            activation_shape=None,
            activation_dtype=torch.float64,
            device="cpu",
            dynamic_shapes=True,
        )
        weights = (1.1, 0.9)
        stage = _Chunk(
            first=rank == 0,
            last=rank == world_size - 1,
            initial_weight=weights[rank],
        )
        microbatches = [
            {
                "x": torch.full((1, sequence, 3), 0.25 + index, dtype=torch.float64),
                "target": torch.full((1, sequence, 3), -0.5 + index / 3, dtype=torch.float64),
            }
            for index, sequence in enumerate((2, 5, 3, 7))
        ]
        expected = _expected_gradients(microbatches, weights)
        schedule = (
            GPipeSchedule(parallel, communicator, overlap_p2p=overlap_p2p)
            if schedule_name == "gpipe"
            else OneForwardOneBackwardSchedule(
                parallel,
                communicator,
                overlap_p2p=overlap_p2p,
            )
        )
        schedule.forward_backward(stage=stage, microbatches=microbatches)
        assert stage.weight.grad is not None
        torch.testing.assert_close(
            stage.weight.grad,
            expected[rank],
            atol=1.0e-10,
            rtol=1.0e-9,
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.parametrize("overlap_p2p", [False, True])
def test_interleaved_dynamic_pipeline_matches_eager_on_two_gloo_processes(
    overlap_p2p: bool,
) -> None:
    world_size = 2
    handle, init_file = tempfile.mkstemp(prefix="nano-megatron-interleaved-")
    os.close(handle)
    os.unlink(init_file)
    try:
        mp.spawn(
            _worker,
            args=(world_size, init_file, overlap_p2p),
            nprocs=world_size,
            join=True,
        )
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


@pytest.mark.distributed
@pytest.mark.parametrize("schedule_name", ["gpipe", "1f1b"])
@pytest.mark.parametrize("overlap_p2p", [False, True])
def test_dynamic_pipeline_schedules_match_eager_on_two_gloo_processes(
    schedule_name: str,
    overlap_p2p: bool,
) -> None:
    world_size = 2
    handle, init_file = tempfile.mkstemp(prefix="nano-megatron-gpipe-overlap-")
    os.close(handle)
    os.unlink(init_file)
    try:
        mp.spawn(
            _dynamic_schedule_worker,
            args=(world_size, init_file, schedule_name, overlap_p2p),
            nprocs=world_size,
            join=True,
        )
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)
