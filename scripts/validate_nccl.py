#!/usr/bin/env python3
"""Validate repeated NCCL collectives and ring P2P under ``torchrun``."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from time import perf_counter

import torch
import torch.distributed as dist


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collective-mib", type=int, default=32)
    parser.add_argument("--all-gather-mib", type=int, default=4)
    parser.add_argument("--p2p-mib", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--barrier-iterations", type=int, default=200)
    return parser


def _positive(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _elements(mebibytes: int, dtype: torch.dtype = torch.float32) -> int:
    return _positive("payload MiB", mebibytes) * (1 << 20) // torch.empty((), dtype=dtype).itemsize


def _maximum_elapsed(elapsed: float, device: torch.device) -> float:
    value = torch.tensor(elapsed, dtype=torch.float64, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


def _benchmark(
    operation: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize(device)
    dist.barrier()
    started = perf_counter()
    for _ in range(iterations):
        operation()
    torch.cuda.synchronize(device)
    elapsed = _maximum_elapsed(perf_counter() - started, device)
    dist.barrier()
    return elapsed / iterations


def _bandwidth_gib_s(payload_bytes: int, seconds: float, factor: float = 1.0) -> float:
    return payload_bytes * factor / seconds / (1 << 30)


def _all_gather_single(output: torch.Tensor, value: torch.Tensor) -> None:
    operation = getattr(dist, "all_gather_single", None)
    if operation is None:
        operation = dist.all_gather_into_tensor
    operation(output, value)


def main() -> None:
    args = _parser().parse_args()
    warmup = _positive("warmup", args.warmup)
    iterations = _positive("iterations", args.iterations)
    barrier_iterations = _positive("barrier_iterations", args.barrier_iterations)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        raise RuntimeError("validate_nccl.py requires CUDA and NCCL")
    if world_size < 2:
        raise ValueError("validate_nccl.py requires at least two ranks")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    try:
        identity = torch.tensor(float(rank + 1), device=device)
        dist.all_reduce(identity)
        expected_sum = world_size * (world_size + 1) / 2
        if float(identity.item()) != expected_sum:
            raise RuntimeError(
                f"all-reduce correctness failed on rank {rank}: "
                f"{float(identity.item())} != {expected_sum}"
            )

        gathered_ranks = torch.empty(world_size, dtype=torch.int64, device=device)
        local_rank_tensor = torch.tensor([rank], dtype=torch.int64, device=device)
        _all_gather_single(gathered_ranks, local_rank_tensor)
        if gathered_ranks.tolist() != list(range(world_size)):
            raise RuntimeError(
                f"all-gather correctness failed on rank {rank}: {gathered_ranks.tolist()}"
            )

        collective_bytes = args.collective_mib * (1 << 20)
        collective = torch.zeros(
            _elements(args.collective_mib),
            dtype=torch.float32,
            device=device,
        )
        all_reduce_seconds = _benchmark(
            lambda: dist.all_reduce(collective),
            warmup=warmup,
            iterations=iterations,
            device=device,
        )

        all_gather_bytes = args.all_gather_mib * (1 << 20)
        all_gather_input = torch.full(
            (_elements(args.all_gather_mib),),
            float(rank),
            dtype=torch.float32,
            device=device,
        )
        all_gather_output = torch.empty(
            world_size * all_gather_input.numel(),
            dtype=all_gather_input.dtype,
            device=device,
        )
        all_gather_seconds = _benchmark(
            lambda: _all_gather_single(all_gather_output, all_gather_input),
            warmup=warmup,
            iterations=iterations,
            device=device,
        )
        gathered_view = all_gather_output.view(world_size, -1)
        for source_rank in range(world_size):
            if not torch.all(gathered_view[source_rank] == float(source_rank)):
                raise RuntimeError(
                    f"timed all-gather payload failed on rank {rank} from rank {source_rank}"
                )

        p2p_bytes = args.p2p_mib * (1 << 20)
        send = torch.full(
            (_elements(args.p2p_mib),),
            float(rank),
            dtype=torch.float32,
            device=device,
        )
        receive = torch.empty_like(send)
        previous_rank = (rank - 1) % world_size
        next_rank = (rank + 1) % world_size

        def ring_exchange() -> None:
            requests = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, send, next_rank),
                    dist.P2POp(dist.irecv, receive, previous_rank),
                ]
            )
            for request in requests:
                request.wait()

        p2p_seconds = _benchmark(
            ring_exchange,
            warmup=warmup,
            iterations=iterations,
            device=device,
        )
        if not torch.all(receive == float(previous_rank)):
            raise RuntimeError(
                f"ring P2P payload failed on rank {rank}: expected {previous_rank}"
            )

        dist.barrier()
        barrier_started = perf_counter()
        for _ in range(barrier_iterations):
            dist.barrier()
        barrier_seconds = _maximum_elapsed(perf_counter() - barrier_started, device)

        if rank == 0:
            result = {
                "all_gather": {
                    "algorithm_gib_per_second": _bandwidth_gib_s(
                        all_gather_bytes,
                        all_gather_seconds,
                    ),
                    "milliseconds": all_gather_seconds * 1.0e3,
                    "payload_mib_per_rank": args.all_gather_mib,
                },
                "all_reduce": {
                    "algorithm_gib_per_second": _bandwidth_gib_s(
                        collective_bytes,
                        all_reduce_seconds,
                    ),
                    "bus_gib_per_second": _bandwidth_gib_s(
                        collective_bytes,
                        all_reduce_seconds,
                        2.0 * (world_size - 1) / world_size,
                    ),
                    "milliseconds": all_reduce_seconds * 1.0e3,
                    "payload_mib": args.collective_mib,
                },
                "barrier": {
                    "iterations": barrier_iterations,
                    "microseconds": barrier_seconds * 1.0e6 / barrier_iterations,
                },
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(device),
                "iterations": iterations,
                "nccl": list(torch.cuda.nccl.version()),
                "p2p_ring": {
                    "gib_per_second": _bandwidth_gib_s(p2p_bytes, p2p_seconds),
                    "milliseconds": p2p_seconds * 1.0e3,
                    "payload_mib": args.p2p_mib,
                },
                "torch": torch.__version__,
                "world_size": world_size,
            }
            print(json.dumps(result, sort_keys=True))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
