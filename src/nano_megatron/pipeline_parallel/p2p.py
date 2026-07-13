"""Explicit pipeline point-to-point communication."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from nano_megatron.parallel.group import require_parallel_group


def _pipeline_group(parallel: Any) -> Any:
    group = getattr(parallel, "pp", None)
    if group is None:
        from nano_megatron.parallel import GroupKey

        group_getter = getattr(parallel, "group", None)
        if not callable(group_getter):
            raise TypeError("parallel must expose an explicit PP group")
        try:
            group = group_getter(GroupKey.PP)
        except (KeyError, AttributeError) as error:
            raise TypeError("parallel must expose an explicit PP group") from error
    return require_parallel_group(group, name="pipeline-parallel group")


def _raw_group(group: Any) -> dist.ProcessGroup | None:
    return require_parallel_group(group, name="pipeline-parallel group").process_group


class P2PCommunicator:
    """Static-shape P2P communicator owned by one pipeline schedule."""

    def __init__(
        self,
        parallel: Any,
        *,
        activation_shape: Sequence[int],
        activation_dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        self.parallel = parallel
        self.group = _pipeline_group(parallel)
        self.activation_shape = tuple(activation_shape)
        self.activation_dtype = activation_dtype
        self.device = torch.device(device)

    @property
    def size(self) -> int:
        return int(self.group.size)

    def _prev_rank(self) -> int | None:
        return self.parallel.pipeline_prev_rank()

    def _next_rank(self) -> int | None:
        return self.parallel.pipeline_next_rank()

    def _empty_activation(self, *, requires_grad: bool = False) -> Tensor:
        return torch.empty(
            self.activation_shape,
            dtype=self.activation_dtype,
            device=self.device,
            requires_grad=requires_grad,
        )

    def _wire_tensor(self, tensor: Tensor, *, name: str) -> Tensor:
        if tuple(tensor.shape) != self.activation_shape:
            raise ValueError(
                f"pipeline {name} shape {tuple(tensor.shape)} does not match the configured "
                f"activation shape {self.activation_shape}"
            )
        return tensor.to(
            device=self.device,
            dtype=self.activation_dtype,
        ).contiguous()

    def _run_ops(self, ops: list[dist.P2POp]) -> None:
        if not ops:
            return
        requests = dist.batch_isend_irecv(ops)
        for request in requests:
            request.wait()

    def send_forward(self, tensor: Tensor) -> None:
        peer = self._next_rank()
        if peer is not None:
            dist.send(
                self._wire_tensor(tensor, name="forward tensor"),
                dst=peer,
                group=_raw_group(self.group),
            )

    def recv_forward(self) -> Tensor | None:
        peer = self._prev_rank()
        if peer is None:
            return None
        tensor = self._empty_activation()
        dist.recv(tensor, src=peer, group=_raw_group(self.group))
        return tensor.detach().requires_grad_(True)

    def send_backward(self, gradient: Tensor) -> None:
        peer = self._prev_rank()
        if peer is not None:
            dist.send(
                self._wire_tensor(gradient, name="backward gradient"),
                dst=peer,
                group=_raw_group(self.group),
            )

    def recv_backward(self) -> Tensor | None:
        peer = self._next_rank()
        if peer is None:
            return None
        gradient = self._empty_activation()
        dist.recv(gradient, src=peer, group=_raw_group(self.group))
        return gradient

    def send_forward_recv_backward(self, output: Tensor) -> Tensor | None:
        peer = self._next_rank()
        if peer is None:
            return None
        gradient = self._empty_activation()
        self._run_ops(
            [
                dist.P2POp(
                    dist.isend,
                    self._wire_tensor(output, name="forward tensor"),
                    peer,
                    _raw_group(self.group),
                ),
                dist.P2POp(dist.irecv, gradient, peer, _raw_group(self.group)),
            ]
        )
        # In 1F1B this gradient belongs to the oldest queued output, not necessarily ``output``.
        # The schedule casts it to that exact graph tensor before backward.
        return gradient

    def send_backward_recv_forward(
        self, gradient: Tensor | None, *, receive_forward: bool = True
    ) -> Tensor | None:
        prev = self._prev_rank()
        ops: list[dist.P2POp] = []
        received: Tensor | None = None
        if prev is not None and gradient is not None:
            ops.append(
                dist.P2POp(
                    dist.isend,
                    self._wire_tensor(gradient, name="backward gradient"),
                    prev,
                    _raw_group(self.group),
                )
            )
        if prev is not None and receive_forward:
            received = self._empty_activation()
            ops.append(dist.P2POp(dist.irecv, received, prev, _raw_group(self.group)))
        self._run_ops(ops)
        return received.detach().requires_grad_(True) if received is not None else None
