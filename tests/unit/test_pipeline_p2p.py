from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402

from nano_megatron.pipeline_parallel import (  # noqa: E402
    P2PCommunicator,
    P2PMessageKind,
    P2PMetadata,
    P2PReceive,
    P2PSend,
    PendingP2P,
)
from nano_megatron.pipeline_parallel.p2p import _pipeline_source_color  # noqa: E402


def _source_colored_parallel(
    rank: int = 0,
    size: int = 2,
    *,
    backend: str = "gloo",
):
    raw_groups = (object(), object(), object())
    ranks = tuple(range(size))
    groups = {
        "pp": SimpleNamespace(
            rank=rank,
            size=size,
            ranks=ranks,
            process_group=raw_groups[0],
            backend=backend,
        ),
        "pp_transport_1": SimpleNamespace(
            rank=rank,
            size=size,
            ranks=ranks,
            process_group=raw_groups[1],
            backend=backend,
        ),
        "pp_transport_2": SimpleNamespace(
            rank=rank,
            size=size,
            ranks=ranks,
            process_group=raw_groups[2],
            backend=backend,
        ),
    }
    parallel = SimpleNamespace(
        pp=groups["pp"],
        group=lambda key: groups[str(key)],
        pipeline_next_rank=lambda: 1,
        pipeline_prev_rank=lambda: 1,
    )
    return parallel, raw_groups


def test_individual_p2p_operations_use_the_batched_api(monkeypatch) -> None:
    raw_group = object()
    parallel = SimpleNamespace(
        pp=SimpleNamespace(rank=0, size=2, process_group=raw_group),
        pipeline_next_rank=lambda: 7,
        pipeline_prev_rank=lambda: 3,
    )
    communicator = P2PCommunicator(
        parallel,
        activation_shape=(2, 3),
        activation_dtype=torch.float32,
        device="cpu",
    )

    batch_calls = []
    waited_requests = []

    def fake_p2p_op(op, tensor, peer, group):
        return SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group)

    class Request:
        def wait(self) -> None:
            waited_requests.append(self)

    def fake_batch_isend_irecv(ops):
        batch_calls.append(ops)
        return [Request() for _ in ops]

    def fail_blocking_p2p(*args, **kwargs) -> None:
        del args, kwargs
        pytest.fail("individual pipeline operations must use batch_isend_irecv")

    monkeypatch.setattr(dist, "P2POp", fake_p2p_op)
    monkeypatch.setattr(dist, "batch_isend_irecv", fake_batch_isend_irecv)
    monkeypatch.setattr(dist, "send", fail_blocking_p2p)
    monkeypatch.setattr(dist, "recv", fail_blocking_p2p)

    payload = torch.randn(2, 3)
    communicator.send_forward(payload)
    received_forward = communicator.recv_forward()
    communicator.send_backward(payload)
    received_backward = communicator.recv_backward()

    assert len(batch_calls) == 4
    assert all(len(ops) == 1 for ops in batch_calls)
    assert [ops[0].op for ops in batch_calls] == [
        dist.isend,
        dist.irecv,
        dist.isend,
        dist.irecv,
    ]
    assert [ops[0].peer for ops in batch_calls] == [7, 3, 3, 7]
    assert all(ops[0].group is raw_group for ops in batch_calls)
    assert len(waited_requests) == 4
    assert received_forward is not None and received_forward.requires_grad
    assert received_backward is not None and not received_backward.requires_grad


def test_pending_p2p_waits_receive_without_waiting_send_and_is_idempotent() -> None:
    class Request:
        def __init__(self) -> None:
            self.wait_count = 0

        def wait(self) -> None:
            self.wait_count += 1

    send = Request()
    receive = Request()
    buffer = torch.randn(2, 3)
    pending = PendingP2P(
        send_requests=(send,),
        receive_requests={P2PMessageKind.FORWARD: receive},
        receive_buffers={P2PMessageKind.FORWARD: buffer},
        receive_requires_grad={P2PMessageKind.FORWARD: True},
        keepalive=(buffer,),
    )

    result = pending.wait_receive(P2PMessageKind.FORWARD)
    assert result is not None and result.requires_grad
    assert receive.wait_count == 1
    assert send.wait_count == 0
    assert pending.wait_receive(P2PMessageKind.FORWARD) is result
    assert receive.wait_count == 1

    pending.wait_sends()
    pending.wait_sends()
    assert send.wait_count == 1
    assert pending.done


def test_pending_p2p_deduplicates_a_coalesced_work_handle() -> None:
    class Request:
        wait_count = 0

        def wait(self) -> None:
            self.wait_count += 1

    request = Request()
    forward = torch.randn(2, 3)
    backward = torch.randn(2, 3)
    pending = PendingP2P(
        send_requests=(request, request),
        receive_requests={
            P2PMessageKind.FORWARD: request,
            P2PMessageKind.BACKWARD: request,
        },
        receive_buffers={
            P2PMessageKind.FORWARD: forward,
            P2PMessageKind.BACKWARD: backward,
        },
    )

    assert pending.wait().forward is forward
    assert request.wait_count == 1
    assert pending.done


def test_pipeline_cycle_source_coloring_is_proper_for_even_and_odd_sizes() -> None:
    for size in (2, 3, 4, 5, 7):
        colors = [_pipeline_source_color(rank, size) for rank in range(size)]
        assert all(colors[rank] != colors[(rank + 1) % size] for rank in range(size))
        assert max(colors) == (1 if size % 2 == 0 else 2)


def test_multi_rank_nccl_requires_source_colored_transport_groups() -> None:
    raw_group = object()
    parallel = SimpleNamespace(
        pp=SimpleNamespace(
            rank=0,
            size=2,
            ranks=(0, 1),
            process_group=raw_group,
            backend="nccl",
        ),
        pipeline_next_rank=lambda: 1,
        pipeline_prev_rank=lambda: 1,
    )
    with pytest.raises(ValueError, match="requires pp_transport_1"):
        P2PCommunicator(
            parallel,
            activation_shape=(2, 3),
            activation_dtype=torch.float32,
            device="cpu",
        )


def test_nccl_exchange_rejects_same_source_color_send_and_receive() -> None:
    parallel, _ = _source_colored_parallel(size=4, backend="nccl")
    communicator = P2PCommunicator(
        parallel,
        activation_shape=(2, 3),
        activation_dtype=torch.float32,
        device="cpu",
    )
    with pytest.raises(ValueError, match="same source-color communicator"):
        communicator.start_exchange(
            sends=(
                P2PSend(
                    torch.randn(2, 3),
                    1,
                    P2PMetadata(P2PMessageKind.FORWARD, 0, 0),
                ),
            ),
            # PP rank 2 has the same color as this rank 0 and is not adjacent.
            receives=(
                P2PReceive(
                    2,
                    P2PMetadata(P2PMessageKind.BACKWARD, 0, 0),
                ),
            ),
        )


def test_source_colored_batches_keep_receive_wait_independent(monkeypatch) -> None:
    parallel, raw_groups = _source_colored_parallel()
    communicator = P2PCommunicator(
        parallel,
        activation_shape=(2, 3),
        activation_dtype=torch.float32,
        device="cpu",
    )
    batch_groups = []

    class Request:
        def __init__(self, group) -> None:
            self.group = group
            self.wait_count = 0

        def wait(self) -> None:
            self.wait_count += 1

    monkeypatch.setattr(
        dist,
        "P2POp",
        lambda op, tensor, peer, group: SimpleNamespace(
            op=op, tensor=tensor, peer=peer, group=group
        ),
    )

    def fake_batch(ops):
        assert len({op.group for op in ops}) == 1
        group = ops[0].group
        batch_groups.append(group)
        # Model NCCL's one shared Work per communicator batch.
        return [Request(group)]

    monkeypatch.setattr(dist, "batch_isend_irecv", fake_batch)
    pending = communicator.start_exchange(
        sends=(
            P2PSend(
                torch.randn(2, 3),
                1,
                P2PMetadata(P2PMessageKind.FORWARD, 0, 0),
            ),
        ),
        receives=(
            P2PReceive(
                1,
                P2PMetadata(P2PMessageKind.BACKWARD, 0, 0),
            ),
        ),
    )

    assert batch_groups == [raw_groups[0], raw_groups[1]]
    send_request = pending._send_requests[0]
    receive_request = pending._receive_requests[P2PMessageKind.BACKWARD]
    assert send_request is not receive_request
    pending.wait_receive(P2PMessageKind.BACKWARD)
    assert receive_request.wait_count == 1
    assert send_request.wait_count == 0
    pending.wait_sends()
    assert send_request.wait_count == 1


def test_dynamic_payload_send_launches_before_future_header_wait(monkeypatch) -> None:
    parallel, raw_groups = _source_colored_parallel()
    communicator = P2PCommunicator(
        parallel,
        activation_shape=None,
        activation_dtype=torch.float32,
        device="cpu",
        dynamic_shapes=True,
    )
    incoming_metadata = P2PMetadata(P2PMessageKind.BACKWARD, 7, 1)
    incoming_header = communicator._encode_header(incoming_metadata, (2, 3))
    launched = []
    waits = []

    monkeypatch.setattr(
        dist,
        "P2POp",
        lambda op, tensor, peer, group: SimpleNamespace(
            op=op, tensor=tensor, peer=peer, group=group
        ),
    )

    class Request:
        def __init__(self, ops) -> None:
            self.ops = ops
            self.wait_count = 0

        def wait(self) -> None:
            self.wait_count += 1
            waits.append(self.ops)
            for op in self.ops:
                if op.op is dist.irecv and op.tensor.dtype == torch.int64:
                    assert any(
                        candidate.op is dist.isend and candidate.tensor.dtype == torch.float32
                        for batch in launched
                        for candidate in batch
                    ), "receive header was waited before payload send launch"
                    op.tensor.copy_(incoming_header)

    def fake_batch(ops):
        ops = list(ops)
        launched.append(ops)
        return [Request(ops)]

    monkeypatch.setattr(dist, "batch_isend_irecv", fake_batch)
    pending = communicator.start_exchange(
        sends=(
            P2PSend(
                torch.randn(5, 3),
                1,
                P2PMetadata(P2PMessageKind.FORWARD, 4, 0),
            ),
        ),
        receives=(P2PReceive(1, incoming_metadata),),
    )

    assert [batch[0].group for batch in launched] == [
        raw_groups[0],
        raw_groups[1],
        raw_groups[0],
        raw_groups[1],
    ]
    assert pending._send_requests[0].wait_count == 0
    received = pending.wait_receive(P2PMessageKind.BACKWARD)
    assert received is not None and tuple(received.shape) == (2, 3)
    assert pending._send_requests[0].wait_count == 0
    pending.wait_sends()


def test_dynamic_explicit_turnaround_completes_header_before_launching_payload(
    monkeypatch,
) -> None:
    parallel, raw_groups = _source_colored_parallel()
    communicator = P2PCommunicator(
        parallel,
        activation_shape=None,
        activation_dtype=torch.float32,
        device="cpu",
        dynamic_shapes=True,
    )
    incoming_metadata = P2PMetadata(P2PMessageKind.BACKWARD, 4, 0)
    incoming_header = communicator._encode_header(incoming_metadata, (2, 3))
    launched = []
    requests = []

    monkeypatch.setattr(
        dist,
        "P2POp",
        lambda op, tensor, peer, group: SimpleNamespace(
            op=op, tensor=tensor, peer=peer, group=group
        ),
    )

    class Request:
        def __init__(self, ops) -> None:
            self.ops = ops
            self.wait_count = 0

        def wait(self) -> None:
            self.wait_count += 1
            for op in self.ops:
                if op.op is dist.irecv and op.tensor.dtype == torch.int64:
                    op.tensor.copy_(incoming_header)

    def fake_batch(ops):
        operations = list(ops)
        request = Request(operations)
        launched.append(operations)
        requests.append(request)
        return [request]

    monkeypatch.setattr(dist, "batch_isend_irecv", fake_batch)
    pending = communicator.start_exchange(
        sends=(
            P2PSend(
                torch.randn(5, 3),
                1,
                P2PMetadata(P2PMessageKind.FORWARD, 4, 0),
            ),
        ),
        receives=(P2PReceive(1, incoming_metadata),),
        causal_turnaround=True,
    )

    assert [batch[0].group for batch in launched] == [
        raw_groups[0],
        raw_groups[0],
        raw_groups[1],
        raw_groups[1],
    ]
    assert [request.wait_count for request in requests] == [1, 1, 1, 0]
    assert pending._send_requests == []
    pending.wait_receive(P2PMessageKind.BACKWARD)
    pending.wait_sends()
    assert [request.wait_count for request in requests] == [1, 1, 1, 1]


def test_dynamic_header_round_trip_and_identity_validation() -> None:
    raw_group = object()
    parallel = SimpleNamespace(
        pp=SimpleNamespace(rank=0, size=2, process_group=raw_group),
        pipeline_next_rank=lambda: 7,
        pipeline_prev_rank=lambda: 3,
    )
    communicator = P2PCommunicator(
        parallel,
        activation_shape=None,
        activation_dtype=torch.float32,
        device="cpu",
        dynamic_shapes=True,
    )
    metadata = P2PMetadata(P2PMessageKind.FORWARD, microbatch=5, chunk=2)
    header = communicator._encode_header(metadata, (2, 7, 11))

    assert communicator._decode_header(header, metadata) == (2, 7, 11)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        communicator._decode_header(
            header,
            P2PMetadata(P2PMessageKind.BACKWARD, microbatch=5, chunk=2),
        )


def test_dynamic_send_keeps_header_and_payload_requests_asynchronous(monkeypatch) -> None:
    raw_group = object()
    parallel = SimpleNamespace(
        pp=SimpleNamespace(rank=0, size=2, process_group=raw_group),
        pipeline_next_rank=lambda: 7,
        pipeline_prev_rank=lambda: 3,
    )
    communicator = P2PCommunicator(
        parallel,
        activation_shape=None,
        activation_dtype=torch.float32,
        device="cpu",
        dynamic_shapes=True,
    )
    phase_requests = []

    class Request:
        def __init__(self, phase: int) -> None:
            self.phase = phase
            self.wait_count = 0

        def wait(self) -> None:
            self.wait_count += 1

    monkeypatch.setattr(
        dist,
        "P2POp",
        lambda op, tensor, peer, group: SimpleNamespace(
            op=op, tensor=tensor, peer=peer, group=group
        ),
    )

    def fake_batch(ops):
        requests = [Request(len(phase_requests)) for _ in ops]
        phase_requests.append(requests)
        return requests

    monkeypatch.setattr(dist, "batch_isend_irecv", fake_batch)
    pending = communicator.start_send_forward(
        torch.randn(2, 5, 3),
        microbatch=4,
        chunk=1,
    )

    assert len(phase_requests) == 2
    assert phase_requests[0][0].wait_count == 0
    assert phase_requests[1][0].wait_count == 0
    assert pending._keepalive
    pending.wait_sends()
    assert phase_requests[0][0].wait_count == 1
    assert phase_requests[1][0].wait_count == 1
