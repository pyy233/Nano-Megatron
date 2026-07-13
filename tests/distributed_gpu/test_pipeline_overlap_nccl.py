from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.profiler import ProfilerActivity, profile, record_function

from nano_megatron.parallel import GroupKey, ParallelGroup
from nano_megatron.pipeline_parallel import (
    GPipeSchedule,
    InterleavedOneForwardOneBackwardSchedule,
    OneForwardOneBackwardSchedule,
    P2PCommunicator,
    P2PMessageKind,
    P2PMetadata,
    P2PReceive,
    P2PSend,
    VirtualPipelineLayout,
)

pytestmark = [
    pytest.mark.distributed,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="requires two CUDA devices",
    ),
]


@dataclass(frozen=True)
class _Parallel:
    pp: ParallelGroup
    pp_transport_1: ParallelGroup
    pp_transport_2: ParallelGroup

    def group(self, key: GroupKey) -> ParallelGroup:
        if key is GroupKey.PP:
            return self.pp
        if key is GroupKey.PP_TRANSPORT_1:
            return self.pp_transport_1
        if key is GroupKey.PP_TRANSPORT_2:
            return self.pp_transport_2
        raise KeyError(key)

    def is_pipeline_first_stage(self) -> bool:
        return self.pp.rank == 0

    def is_pipeline_last_stage(self) -> bool:
        return self.pp.rank == self.pp.size - 1

    def pipeline_prev_rank(self) -> int | None:
        return None if self.pp.rank == 0 else self.pp.global_rank_at(self.pp.rank - 1)

    def pipeline_next_rank(self) -> int | None:
        if self.pp.rank == self.pp.size - 1:
            return None
        return self.pp.global_rank_at(self.pp.rank + 1)


def _create_parallel(rank: int, world_size: int) -> tuple[_Parallel, list]:
    ranks = tuple(range(world_size))
    transport_process_groups = [dist.new_group(ranks=ranks, backend="nccl") for _ in range(2)]

    def pipeline_group(key: GroupKey, process_group) -> ParallelGroup:
        return ParallelGroup(
            key=key,
            ranks=ranks,
            process_group=process_group,
            rank=rank,
            size=world_size,
            backend="nccl",
        )

    return (
        _Parallel(
            pipeline_group(GroupKey.PP, dist.group.WORLD),
            pipeline_group(GroupKey.PP_TRANSPORT_1, transport_process_groups[0]),
            pipeline_group(GroupKey.PP_TRANSPORT_2, transport_process_groups[1]),
        ),
        transport_process_groups,
    )


class _Chunk(nn.Module):
    def __init__(self, *, first: bool, last: bool, weight: float) -> None:
        super().__init__()
        self.first = first
        self.last = last
        self.weight = nn.Parameter(torch.tensor(weight, dtype=torch.float32))

    def forward(self, hidden_states: Tensor | None, batch) -> Tensor:
        if self.first:
            assert hidden_states is None
            hidden_states = batch["x"]
        else:
            assert hidden_states is not None
        output = hidden_states * self.weight
        return (output - batch["target"]).square().mean() if self.last else output


class _Pipeline(nn.Module):
    def __init__(self, rank: int, layout: VirtualPipelineLayout, weights) -> None:
        super().__init__()
        self.chunks = nn.ModuleList(
            [
                _Chunk(
                    first=layout.is_first(layout.address(rank, chunk)),
                    last=layout.is_last(layout.address(rank, chunk)),
                    weight=weights[layout.address(rank, chunk).logical_stage],
                )
                for chunk in range(layout.virtual_stages_per_rank)
            ]
        )

    def chunk(self, chunk_id: int) -> _Chunk:
        return self.chunks[chunk_id]

    def synchronize_tied_embedding_weights(self) -> None:
        return None

    def synchronize_tied_embedding_gradients(self) -> None:
        return None


def _expected_gradients(microbatches, weights):
    parameters = [
        torch.tensor(
            value,
            dtype=torch.float32,
            device=microbatches[0]["x"].device,
            requires_grad=True,
        )
        for value in weights
    ]
    losses = []
    for batch in microbatches:
        output = batch["x"]
        for parameter in parameters:
            output = output * parameter
        losses.append((output - batch["target"]).square().mean())
    (sum(losses) / len(losses)).backward()
    return tuple(parameter.grad.detach().clone() for parameter in parameters)


def _worker(rank: int, world_size: int, rendezvous: str) -> None:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    transport_process_groups = []
    try:
        parallel, transport_process_groups = _create_parallel(rank, world_size)
        microbatches = [
            {
                "x": torch.full(
                    (1, sequence, 8),
                    0.25 + index,
                    dtype=torch.float32,
                    device=device,
                ),
                "target": torch.full(
                    (1, sequence, 8),
                    -0.5 + index / 3,
                    dtype=torch.float32,
                    device=device,
                ),
            }
            for index, sequence in enumerate((2, 5, 3, 7))
        ]

        # This is intentionally the first P2P use.  Rank 1 enters the lazy
        # transport warmup from its pre-posted receive while rank 0 enters only
        # after first-stage compute; completion proves the phase is safe.
        gpipe_weights = (1.1, 0.9)
        gpipe_expected = _expected_gradients(microbatches, gpipe_weights)
        gpipe_stage = _Chunk(
            first=rank == 0,
            last=rank == world_size - 1,
            weight=gpipe_weights[rank],
        ).to(device)
        gpipe_communicator = P2PCommunicator(
            parallel,
            activation_shape=None,
            activation_dtype=torch.float32,
            device=device,
            dynamic_shapes=True,
        )
        GPipeSchedule(
            parallel,
            gpipe_communicator,
            overlap_p2p=True,
        ).forward_backward(stage=gpipe_stage, microbatches=microbatches)
        assert gpipe_stage.weight.grad is not None
        torch.testing.assert_close(
            gpipe_stage.weight.grad,
            gpipe_expected[rank],
            atol=2.0e-5,
            rtol=2.0e-5,
        )

        one_f_one_b_weights = (1.2, 0.7)
        # MB=1 exercises the all-warmup/all-cooldown edge, while MB=4 reaches
        # the alternating steady state.  Both must agree in sync and overlap
        # modes because overlap changes only request timing.
        for microbatch_count in (1, len(microbatches)):
            selected_microbatches = microbatches[:microbatch_count]
            one_f_one_b_expected = _expected_gradients(
                selected_microbatches,
                one_f_one_b_weights,
            )
            for overlap_p2p in (False, True):
                one_f_one_b_stage = _Chunk(
                    first=rank == 0,
                    last=rank == world_size - 1,
                    weight=one_f_one_b_weights[rank],
                ).to(device)
                one_f_one_b_communicator = P2PCommunicator(
                    parallel,
                    activation_shape=None,
                    activation_dtype=torch.float32,
                    device=device,
                    dynamic_shapes=True,
                )
                OneForwardOneBackwardSchedule(
                    parallel,
                    one_f_one_b_communicator,
                    overlap_p2p=overlap_p2p,
                ).forward_backward(
                    stage=one_f_one_b_stage,
                    microbatches=selected_microbatches,
                )
                assert one_f_one_b_stage.weight.grad is not None
                torch.testing.assert_close(
                    one_f_one_b_stage.weight.grad,
                    one_f_one_b_expected[rank],
                    atol=2.0e-5,
                    rtol=2.0e-5,
                )

        layout = VirtualPipelineLayout(4, world_size, 2)
        communicator = P2PCommunicator(
            parallel,
            activation_shape=None,
            activation_dtype=torch.float32,
            device=device,
            dynamic_shapes=True,
        )
        weights = (1.1, 0.9, 1.3, 0.8)
        pipeline = _Pipeline(rank, layout, weights).to(device)
        expected = _expected_gradients(microbatches, weights)

        schedule = InterleavedOneForwardOneBackwardSchedule(
            parallel,
            layout,
            communicator,
            overlap_p2p=True,
        )
        # MB=1 is the minimal future-header cycle; MB=3 checks the same
        # protocol across a partial microbatch group on real NCCL.
        for forward_count in (1, 3):
            schedule.forward_backward(
                stage=pipeline,
                microbatches=microbatches[:forward_count],
                forward_only=True,
            )
        schedule.forward_backward(stage=pipeline, microbatches=microbatches)

        for chunk_id, chunk in enumerate(pipeline.chunks):
            logical_stage = layout.address(rank, chunk_id).logical_stage
            assert chunk.weight.grad is not None
            torch.testing.assert_close(
                chunk.weight.grad,
                expected[logical_stage],
                atol=2.0e-5,
                rtol=2.0e-5,
            )
    finally:
        for process_group in reversed(transport_process_groups):
            dist.destroy_process_group(process_group)
        dist.destroy_process_group()


def test_pipeline_schedules_dynamic_p2p_overlap_nccl() -> None:
    world_size = 2
    handle, rendezvous = tempfile.mkstemp(prefix="nano-megatron-nccl-pipeline-")
    os.close(handle)
    os.unlink(rendezvous)
    try:
        mp.spawn(
            _worker,
            args=(world_size, rendezvous),
            nprocs=world_size,
            join=True,
        )
    finally:
        if os.path.exists(rendezvous):
            os.unlink(rendezvous)


def _timeline_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    trace_path: str,
) -> None:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    transport_process_groups = []
    try:
        parallel, transport_process_groups = _create_parallel(rank, world_size)
        communicator = P2PCommunicator(
            parallel,
            activation_shape=None,
            activation_dtype=torch.float32,
            device=device,
            dynamic_shapes=True,
        )
        # Keep communicator initialization outside the measured interval.  The
        # parity test above separately covers first-use lazy warmup.
        communicator._ensure_transport_ready()
        dist.barrier()

        huge_elements = 256 * 1024 * 1024  # 1 GiB in FP32.
        outgoing = torch.ones(
            huge_elements if rank == 0 else 1,
            dtype=torch.float32,
            device=device,
        )
        # Prime the caching allocator for the dynamic receive buffer so the
        # measured start_exchange does not call cudaMalloc and synchronize
        # otherwise independent streams.
        receive_reserve = torch.empty(
            1 if rank == 0 else huge_elements,
            dtype=torch.float32,
            device=device,
        )
        del receive_reserve
        send_kind = P2PMessageKind.FORWARD if rank == 0 else P2PMessageKind.BACKWARD
        receive_kind = P2PMessageKind.BACKWARD if rank == 0 else P2PMessageKind.FORWARD
        left = right = ping = pong = None
        if rank == 0:
            left = torch.randn(4096, 4096, dtype=torch.float32, device=device)
            right = torch.randn(4096, 4096, dtype=torch.float32, device=device)
            ping = torch.empty_like(left)
            pong = torch.empty_like(left)
            # Initialize cuBLAS, choose/cache an algorithm, and reserve its
            # workspace before profiling.  All measured mm calls use fixed
            # output buffers and therefore perform no CUDA allocation.
            torch.mm(left, right, out=ping)
            torch.mm(ping, right, out=pong)
        torch.cuda.synchronize(device)

        with profile(activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA)) as profiler:
            pending = communicator.start_exchange(
                sends=(
                    P2PSend(
                        outgoing,
                        1 - rank,
                        P2PMetadata(send_kind, 0, 0),
                    ),
                ),
                receives=(
                    P2PReceive(
                        1 - rank,
                        P2PMetadata(receive_kind, 0, 0),
                    ),
                ),
            )
            pending.wait_receive(receive_kind)
            if rank == 0:
                assert all(tensor is not None for tensor in (left, right, ping, pong))
                with record_function("nano_overlap_compute"):
                    source = left
                    for _ in range(8):
                        destination = ping if source is not ping else pong
                        torch.mm(source, right, out=destination)
                        source = destination
            pending.wait_sends()
            torch.cuda.synchronize(device)

        if rank == 0:
            profiler.export_chrome_trace(trace_path)
    finally:
        for process_group in reversed(transport_process_groups):
            dist.destroy_process_group(process_group)
        dist.destroy_process_group()


def _interval_overlap(left: dict, right: dict) -> float:
    return max(
        0.0,
        min(left["ts"] + left["dur"], right["ts"] + right["dur"]) - max(left["ts"], right["ts"]),
    )


_SCHEDULE_COMPUTE_MARKER = "nano_1f1b_stage_forward_mb1"
_SCHEDULE_SEND_MARKER = "nano_1f1b_send_forward_mb0"


class _ProfiledP2PCommunicator(P2PCommunicator):
    """Add CPU correlation ranges without changing the communicator lifecycle."""

    def start_exchange(
        self,
        *,
        sends=(),
        receives=(),
        causal_turnaround: bool = False,
    ):
        marker = "nano_1f1b_exchange"
        if len(sends) == 1:
            metadata = sends[0].metadata
            marker = f"nano_1f1b_send_{metadata.kind.name.lower()}_mb{metadata.microbatch}"
        with record_function(marker):
            return super().start_exchange(
                sends=sends,
                receives=receives,
                causal_turnaround=causal_turnaround,
            )


class _ScheduleTimelineStage(nn.Module):
    """Synthetic stage with identifiable GEMMs and a large real PP activation."""

    def __init__(
        self,
        *,
        first: bool,
        last: bool,
        device: torch.device,
        matrix_size: int = 4096,
        compute_iterations: int = 8,
    ) -> None:
        super().__init__()
        self.first = first
        self.last = last
        self.compute_iterations = compute_iterations
        self.weight = nn.Parameter(torch.tensor(1.01, dtype=torch.float32, device=device))
        self.register_buffer(
            "left",
            torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)
            if first
            else None,
        )
        self.register_buffer(
            "right",
            torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)
            if first
            else None,
        )
        self.register_buffer(
            "ping",
            torch.empty(matrix_size, matrix_size, dtype=torch.float32, device=device)
            if first
            else None,
        )
        self.register_buffer(
            "pong",
            torch.empty(matrix_size, matrix_size, dtype=torch.float32, device=device)
            if first
            else None,
        )

    def _matrix_compute(self) -> None:
        if not self.first:
            return
        assert self.left is not None
        assert self.right is not None
        assert self.ping is not None
        assert self.pong is not None
        source = self.left
        for _ in range(self.compute_iterations):
            destination = self.ping if source is not self.ping else self.pong
            torch.mm(source, self.right, out=destination)
            source = destination

    def warmup_compute(self) -> None:
        self._matrix_compute()
        self._matrix_compute()

    def forward(self, hidden_states: Tensor | None, batch) -> Tensor:
        microbatch = int(batch["microbatch"])
        with record_function(f"nano_1f1b_stage_forward_mb{microbatch}"):
            if self.first:
                assert hidden_states is None
                self._matrix_compute()
                return batch["x"] * self.weight
            assert hidden_states is not None
            if not self.last:
                return hidden_states * self.weight
            return hidden_states.mean() * self.weight


def _schedule_timeline_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    trace_path: str,
) -> None:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    transport_process_groups = []
    try:
        parallel, transport_process_groups = _create_parallel(rank, world_size)
        payload_elements = 128 * 1024 * 1024  # 512 MiB in FP32.
        communicator = _ProfiledP2PCommunicator(
            parallel,
            activation_shape=(payload_elements,),
            activation_dtype=torch.float32,
            device=device,
        )
        communicator._ensure_transport_ready()
        stage = _ScheduleTimelineStage(
            first=rank == 0,
            last=rank == world_size - 1,
            device=device,
        )
        stage.warmup_compute()

        payload = torch.ones(
            payload_elements if rank == 0 else 1,
            dtype=torch.float32,
            device=device,
        )
        microbatches = [
            {"microbatch": index, "x": payload if rank == 0 else None} for index in range(2)
        ]

        # Prime enough same-sized blocks for the two forward outputs on rank 0
        # and the rotating receive/gradient buffers on rank 1.  cudaMalloc in
        # the measured region would otherwise serialize otherwise independent
        # NCCL and compute streams.
        allocator_reserve = [
            torch.empty(payload_elements, dtype=torch.float32, device=device) for _ in range(2)
        ]
        del allocator_reserve
        torch.cuda.synchronize(device)
        dist.barrier()

        with profile(activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA)) as profiler:
            OneForwardOneBackwardSchedule(
                parallel,
                communicator,
                overlap_p2p=True,
            ).forward_backward(stage=stage, microbatches=microbatches)
            torch.cuda.synchronize(device)

        if rank == 0:
            profiler.export_chrome_trace(trace_path)
        dist.barrier()
    finally:
        for process_group in reversed(transport_process_groups):
            dist.destroy_process_group(process_group)
        dist.destroy_process_group()


def _event_contains(parent: dict, child: dict) -> bool:
    if parent.get("pid") != child.get("pid") or parent.get("tid") != child.get("tid"):
        return False
    parent_start = float(parent.get("ts", 0.0))
    parent_end = parent_start + float(parent.get("dur", 0.0))
    child_start = float(child.get("ts", 0.0))
    child_end = child_start + float(child.get("dur", 0.0))
    return parent_start <= child_start and child_end <= parent_end


def _external_ids_inside(events: list[dict], marker: dict) -> set:
    external_ids = {
        event.get("args", {}).get("External id")
        for event in events
        if _event_contains(marker, event)
    }
    external_ids.discard(None)
    return external_ids


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_RUN_TIMELINE_TESTS") != "1",
    reason="set NANO_MEGATRON_RUN_TIMELINE_TESTS=1 for the schedule profiler test",
)
def test_non_interleaved_1f1b_overlaps_previous_send_with_next_stage_compute() -> None:
    world_size = 2
    handle, rendezvous = tempfile.mkstemp(prefix="nano-megatron-1f1b-timeline-")
    os.close(handle)
    os.unlink(rendezvous)
    trace_handle, trace_path = tempfile.mkstemp(
        prefix="nano-megatron-1f1b-timeline-", suffix=".json"
    )
    os.close(trace_handle)
    os.unlink(trace_path)
    keep_trace = os.environ.get("NANO_MEGATRON_KEEP_TIMELINE_TRACE") == "1"
    try:
        mp.spawn(
            _schedule_timeline_worker,
            args=(world_size, rendezvous, trace_path),
            nprocs=world_size,
            join=True,
        )
        with open(trace_path, encoding="utf-8") as trace_file:
            events = json.load(trace_file)["traceEvents"]

        compute_markers = [
            event
            for event in events
            if event.get("cat") == "user_annotation"
            and event.get("name") == _SCHEDULE_COMPUTE_MARKER
            and float(event.get("dur", 0.0)) > 0.0
        ]
        send_markers = [
            event
            for event in events
            if event.get("cat") == "user_annotation"
            and event.get("name") == _SCHEDULE_SEND_MARKER
            and float(event.get("dur", 0.0)) > 0.0
        ]
        assert len(compute_markers) == 1, (
            f"expected one {_SCHEDULE_COMPUTE_MARKER!r} range, got {len(compute_markers)}"
        )
        assert len(send_markers) == 1, (
            f"expected one {_SCHEDULE_SEND_MARKER!r} range, got {len(send_markers)}"
        )
        compute_marker = compute_markers[0]
        send_marker = send_markers[0]
        assert float(send_marker["ts"]) < float(compute_marker["ts"]), (
            "the forward-microbatch-0 send must be launched before "
            "forward-microbatch-1 stage compute"
        )

        mm_external_ids = {
            event.get("args", {}).get("External id")
            for event in events
            if event.get("name") == "aten::mm" and _event_contains(compute_marker, event)
        }
        mm_external_ids.discard(None)
        compute_kernels = [
            event
            for event in events
            if event.get("cat") == "kernel"
            and event.get("args", {}).get("External id") in mm_external_ids
            and float(event.get("dur", 0.0)) > 0.0
        ]
        all_nccl_kernels = sorted(
            (
                event
                for event in events
                if event.get("cat") == "kernel"
                and "nccl" in str(event.get("name", "")).lower()
                and float(event.get("dur", 0.0)) > 0.0
            ),
            key=lambda event: float(event["ts"]),
        )
        assert mm_external_ids, "F1 marker contains no aten::mm correlation ids"
        assert compute_kernels, "trace contains no F1 stage-compute GEMM kernels"
        assert all_nccl_kernels, "trace contains no NCCL kernels"

        send_external_ids = _external_ids_inside(events, send_marker)
        send_kernels = [
            event
            for event in all_nccl_kernels
            if event.get("args", {}).get("External id") in send_external_ids
        ]
        selection = "external-id correlation"
        if not send_kernels:
            # This rank performs no communication before F0's static-shape send:
            # transport warmup and the barrier are outside the profile, and the
            # first-stage prime has no receive.  Some PyTorch versions do not
            # propagate a record_function external id through ProcessGroupNCCL;
            # there the first NCCL kernel is unambiguously F0's payload send.
            send_kernels = [all_nccl_kernels[0]]
            selection = "first profiled NCCL kernel"

        overlap_us, send_kernel, compute_kernel = max(
            (
                [
                    _interval_overlap(send_kernel, compute_kernel),
                    send_kernel,
                    compute_kernel,
                ]
                for send_kernel in send_kernels
                for compute_kernel in compute_kernels
            ),
            key=lambda candidate: float(candidate[0]),
        )
        print(f"schedule timeline trace: {trace_path}")
        print(f"F0 payload-send kernel selected by: {selection}")
        print(
            "maximum previous-send/F1-compute overlap: "
            f"{overlap_us:.1f}us; send={send_kernel['name']!s}; "
            f"compute={compute_kernel['name']!s}"
        )
        assert overlap_us >= 100.0, (
            "the F0 payload-send kernel did not overlap an F1 stage-compute kernel "
            f"for at least 100us (observed {overlap_us:.1f}us)"
        )
    finally:
        paths = [rendezvous]
        if not keep_trace:
            paths.append(trace_path)
        for path in paths:
            if os.path.exists(path):
                os.unlink(path)


@pytest.mark.skipif(
    os.environ.get("NANO_MEGATRON_RUN_TIMELINE_TESTS") != "1",
    reason="set NANO_MEGATRON_RUN_TIMELINE_TESTS=1 for the 1 GiB profiler test",
)
def test_nccl_payload_send_kernel_overlaps_next_compute_kernel() -> None:
    world_size = 2
    handle, rendezvous = tempfile.mkstemp(prefix="nano-megatron-nccl-timeline-")
    os.close(handle)
    os.unlink(rendezvous)
    trace_handle, trace_path = tempfile.mkstemp(
        prefix="nano-megatron-nccl-timeline-", suffix=".json"
    )
    os.close(trace_handle)
    os.unlink(trace_path)
    keep_trace = os.environ.get("NANO_MEGATRON_KEEP_TIMELINE_TRACE") == "1"
    try:
        mp.spawn(
            _timeline_worker,
            args=(world_size, rendezvous, trace_path),
            nprocs=world_size,
            join=True,
        )
        with open(trace_path, encoding="utf-8") as trace_file:
            trace = json.load(trace_file)
        events = trace["traceEvents"]
        mm_external_ids = {
            event.get("args", {}).get("External id")
            for event in events
            if event.get("name") == "aten::mm"
        }
        mm_external_ids.discard(None)
        compute_kernels = [
            event
            for event in events
            if event.get("cat") == "kernel"
            and event.get("args", {}).get("External id") in mm_external_ids
            and float(event.get("dur", 0.0)) > 0.0
        ]
        nccl_kernels = [
            event
            for event in events
            if event.get("cat") == "kernel"
            and "nccl" in str(event.get("name", "")).lower()
            and float(event.get("dur", 0.0)) > 0.0
        ]
        assert compute_kernels, "profiler trace contains no matrix-multiply kernels"
        assert nccl_kernels, "profiler trace contains no NCCL kernels"
        base_ts = min(float(event["ts"]) for event in (*nccl_kernels, *compute_kernels))
        nccl_summary = [
            (
                str(event["name"]),
                float(event["ts"]) - base_ts,
                float(event["dur"]),
            )
            for event in nccl_kernels
        ]
        compute_summary = [
            (
                str(event["name"]),
                float(event["ts"]) - base_ts,
                float(event["dur"]),
            )
            for event in compute_kernels
        ]
        overlap_us, nccl_event, compute_event = max(
            (
                [
                    _interval_overlap(nccl_event, compute_event),
                    nccl_event,
                    compute_event,
                ]
                for nccl_event in nccl_kernels
                for compute_event in compute_kernels
            ),
            key=lambda candidate: float(candidate[0]),
        )
        print(f"timeline trace: {trace_path}")
        print(f"NCCL kernels (name, relative_us, duration_us): {nccl_summary}")
        print(f"GEMM kernels (name, relative_us, duration_us): {compute_summary}")
        print(
            "maximum NCCL/GEMM overlap: "
            f"{overlap_us:.1f}us; nccl={nccl_event['name']!s}; "
            f"compute={compute_event['name']!s}"
        )
        assert overlap_us >= 100.0, (
            "no NCCL kernel overlapped a compute kernel "
            f"for at least 100us (observed {overlap_us:.1f}us)"
        )
    finally:
        paths = [rendezvous]
        if not keep_trace:
            paths.append(trace_path)
        for path in paths:
            if os.path.exists(path):
                os.unlink(path)
