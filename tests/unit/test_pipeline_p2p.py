from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402

from nano_megatron.pipeline_parallel import P2PCommunicator  # noqa: E402


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
