"""Explicit, lifecycle-safe pipeline point-to-point communication."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import IntEnum
from math import prod
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from nano_megatron.parallel.group import require_parallel_group

_PROTOCOL_VERSION = 1
_MAX_TENSOR_DIMS = 8
_HEADER_SIZE = 5 + _MAX_TENSOR_DIMS


class P2PMessageKind(IntEnum):
    FORWARD = 1
    BACKWARD = 2


@dataclass(frozen=True)
class P2PMetadata:
    """Semantic identity carried by a dynamic-shape message header."""

    kind: P2PMessageKind
    microbatch: int = -1
    chunk: int = -1


@dataclass(frozen=True)
class P2PSend:
    tensor: Tensor
    peer: int
    metadata: P2PMetadata


@dataclass(frozen=True)
class P2PReceive:
    peer: int
    metadata: P2PMetadata
    requires_grad: bool = False


@dataclass(frozen=True)
class P2PResult:
    forward: Tensor | None = None
    backward: Tensor | None = None


@dataclass(frozen=True)
class _WireOperation:
    send: bool
    tensor: Tensor
    peer: int
    group: Any


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


def _pipeline_source_color(rank: int, size: int) -> int:
    """Properly color the physical PP cycle with at most three colors."""

    if size < 1 or not 0 <= rank < size:
        raise ValueError(f"invalid pipeline rank/size: rank={rank}, size={size}")
    if size == 1:
        return 0
    if size % 2 == 0:
        return rank % 2
    return 2 if rank == size - 1 else rank % 2


def _pipeline_transport_groups(parallel: Any, pipeline_group: Any) -> tuple[Any, ...]:
    """Resolve source-colored PP communicators, with a fake/Gloo fallback."""

    from nano_megatron.parallel import GroupKey

    groups = [pipeline_group]
    size = int(pipeline_group.size)
    nccl = "nccl" in str(getattr(pipeline_group, "backend", "")).lower()
    required_colors = 1 if size <= 1 else (2 if size % 2 == 0 else 3)
    getter = getattr(parallel, "group", None)
    for color, key in enumerate((GroupKey.PP_TRANSPORT_1, GroupKey.PP_TRANSPORT_2), start=1):
        transport = None
        if callable(getter):
            with suppress(KeyError, AttributeError):
                transport = getter(key)
        if transport is None:
            if nccl and color < required_colors:
                raise ValueError(
                    f"multi-rank NCCL pipeline communication requires {key.value}; "
                    "materialize the default source-colored PP transport groups"
                )
            groups.append(pipeline_group)
            continue
        transport = require_parallel_group(
            transport,
            name=f"pipeline transport group {key.value}",
        )
        if int(transport.rank) != int(pipeline_group.rank) or int(transport.size) != int(
            pipeline_group.size
        ):
            raise ValueError(
                f"pipeline transport group {key.value} must have the same local "
                "rank and size as the PP group"
            )
        pipeline_ranks = getattr(pipeline_group, "ranks", None)
        transport_ranks = getattr(transport, "ranks", None)
        if (
            pipeline_ranks is not None
            and transport_ranks is not None
            and tuple(transport_ranks) != tuple(pipeline_ranks)
        ):
            raise ValueError(
                f"pipeline transport group {key.value} must contain the same "
                "global ranks as the PP group"
            )
        groups.append(transport)
    if nccl:
        active_groups = groups[:required_colors]
        if len({id(_raw_group(group)) for group in active_groups}) != len(active_groups):
            raise ValueError(
                "source-colored NCCL PP transport groups must use distinct "
                "process-group communicators"
            )
    return tuple(groups)


class PendingP2P:
    """Own asynchronous P2P requests and every tensor they reference.

    Receives may be waited independently from sends.  Schedules use that split
    to wait only for the input needed by the next compute operation while the
    previous output send remains in flight.
    """

    def __init__(
        self,
        *,
        send_requests: Sequence[Any] = (),
        receive_requests: dict[P2PMessageKind, Any] | None = None,
        receive_buffers: dict[P2PMessageKind, Tensor] | None = None,
        receive_requires_grad: dict[P2PMessageKind, bool] | None = None,
        keepalive: Sequence[Tensor] = (),
    ) -> None:
        self._send_requests = list(send_requests)
        self._receive_requests = dict(receive_requests or {})
        self._receive_buffers = dict(receive_buffers or {})
        self._receive_requires_grad = dict(receive_requires_grad or {})
        self._keepalive = tuple(keepalive)
        self._completed_sends = not self._send_requests
        self._completed_receives: set[P2PMessageKind] = set()
        self._waited_requests: set[int] = set()

    @property
    def done(self) -> bool:
        return self._completed_sends and self._completed_receives == set(self._receive_requests)

    def wait_sends(self) -> None:
        if self._completed_sends:
            return
        for request in self._send_requests:
            self._wait_request(request)
        self._completed_sends = True
        if not self._receive_requests or self._completed_receives == set(self._receive_requests):
            self._keepalive = ()

    def wait_receive(self, kind: P2PMessageKind) -> Tensor | None:
        request = self._receive_requests.get(kind)
        if request is None:
            return None
        if kind not in self._completed_receives:
            self._wait_request(request)
            tensor = self._receive_buffers[kind]
            if self._receive_requires_grad.get(kind, False):
                tensor = tensor.detach().requires_grad_(True)
                self._receive_buffers[kind] = tensor
            self._completed_receives.add(kind)
            if self._completed_sends and self._completed_receives == set(self._receive_requests):
                self._keepalive = ()
        return self._receive_buffers[kind]

    def _wait_request(self, request: Any) -> None:
        identity = id(request)
        if identity in self._waited_requests:
            return
        request.wait()
        self._waited_requests.add(identity)

    def wait(self) -> P2PResult:
        forward = self.wait_receive(P2PMessageKind.FORWARD)
        backward = self.wait_receive(P2PMessageKind.BACKWARD)
        self.wait_sends()
        return P2PResult(forward=forward, backward=backward)


class P2PCommunicator:
    """Pipeline communicator with static and metadata-described tensor shapes."""

    def __init__(
        self,
        parallel: Any,
        *,
        activation_shape: Sequence[int] | None,
        activation_dtype: torch.dtype,
        device: torch.device | str,
        dynamic_shapes: bool = False,
    ) -> None:
        self.parallel = parallel
        self.group = _pipeline_group(parallel)
        self._transport_groups = _pipeline_transport_groups(parallel, self.group)
        self._transport_ready = False
        self.dynamic_shapes = bool(dynamic_shapes)
        if activation_shape is None and not self.dynamic_shapes:
            raise ValueError("static pipeline communication requires activation_shape")
        self.activation_shape = (
            None if activation_shape is None else self._validate_shape(activation_shape)
        )
        self.activation_dtype = activation_dtype
        self.device = torch.device(device)

    @property
    def size(self) -> int:
        return int(self.group.size)

    def _prev_rank(self) -> int | None:
        return self.parallel.pipeline_prev_rank()

    def _next_rank(self) -> int | None:
        return self.parallel.pipeline_next_rank()

    def previous_rank(self) -> int | None:
        return self._prev_rank()

    def next_rank(self) -> int | None:
        return self._next_rank()

    @staticmethod
    def _validate_shape(shape: Sequence[int]) -> tuple[int, ...]:
        result = tuple(int(dimension) for dimension in shape)
        if len(result) > _MAX_TENSOR_DIMS:
            raise ValueError(f"pipeline tensors support at most {_MAX_TENSOR_DIMS} dimensions")
        if any(dimension <= 0 for dimension in result):
            raise ValueError(f"pipeline tensor dimensions must be positive, got {result}")
        return result

    def _wire_tensor(self, tensor: Tensor, *, name: str) -> Tensor:
        shape = self._validate_shape(tensor.shape)
        if not self.dynamic_shapes and shape != self.activation_shape:
            raise ValueError(
                f"pipeline {name} shape {shape} does not match the configured "
                f"activation shape {self.activation_shape}"
            )
        return tensor.to(device=self.device, dtype=self.activation_dtype).contiguous()

    def _encode_header(self, metadata: P2PMetadata, shape: Sequence[int]) -> Tensor:
        validated = self._validate_shape(shape)
        dimensions = [*validated, *([-1] * (_MAX_TENSOR_DIMS - len(validated)))]
        return torch.tensor(
            [
                _PROTOCOL_VERSION,
                int(metadata.kind),
                int(metadata.microbatch),
                int(metadata.chunk),
                len(validated),
                *dimensions,
            ],
            dtype=torch.int64,
            device=self.device,
        )

    def _decode_header(self, header: Tensor, expected: P2PMetadata) -> tuple[int, ...]:
        values = [int(value) for value in header.cpu().tolist()]
        if len(values) != _HEADER_SIZE:
            raise RuntimeError(f"invalid pipeline header length {len(values)}")
        version, kind, microbatch, chunk, ndim, *dimensions = values
        if version != _PROTOCOL_VERSION:
            raise RuntimeError(
                f"pipeline protocol version mismatch: {version} != {_PROTOCOL_VERSION}"
            )
        try:
            observed_kind = P2PMessageKind(kind)
        except ValueError as error:
            raise RuntimeError(f"invalid pipeline message kind {kind}") from error
        observed = P2PMetadata(observed_kind, microbatch, chunk)
        if observed != expected:
            raise RuntimeError(
                f"pipeline message identity mismatch: received={observed}, expected={expected}"
            )
        if not 0 <= ndim <= _MAX_TENSOR_DIMS:
            raise RuntimeError(f"invalid pipeline tensor ndim {ndim}")
        if any(value != -1 for value in dimensions[ndim:]):
            raise RuntimeError("unused pipeline header dimensions must be -1")
        try:
            shape = self._validate_shape(dimensions[:ndim])
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        if prod(shape) > torch.iinfo(torch.int64).max:
            raise RuntimeError(f"pipeline tensor shape overflows int64 numel: {shape}")
        return shape

    @staticmethod
    def _ordered_entries(
        sends: Sequence[P2PSend], receives: Sequence[P2PReceive]
    ) -> list[tuple[P2PMessageKind, bool, P2PSend | P2PReceive]]:
        entries = [(item.metadata.kind, True, item) for item in sends] + [
            (item.metadata.kind, False, item) for item in receives
        ]
        entries.sort(key=lambda value: (int(value[0]), not value[1]))
        return entries

    def _peer_pipeline_rank(self, peer: int) -> int | None:
        ranks = getattr(self.group, "ranks", None)
        if ranks is None:
            return None
        try:
            return tuple(ranks).index(peer)
        except ValueError as error:
            raise ValueError(
                f"pipeline peer {peer} is not a member of the PP group {tuple(ranks)}"
            ) from error

    def _transport_group(self, *, send: bool, peer: int) -> Any:
        source_rank = int(self.group.rank) if send else self._peer_pipeline_rank(peer)
        # Lightweight fake groups used by unit tests may not expose their
        # global rank tuple.  They retain the historical single-channel path.
        if source_rank is None:
            return self.group
        color = _pipeline_source_color(source_rank, int(self.group.size))
        return self._transport_groups[color]

    def _wire_operation(self, send: bool, tensor: Tensor, peer: int) -> _WireOperation:
        return _WireOperation(
            send=send,
            tensor=tensor,
            peer=peer,
            group=self._transport_group(send=send, peer=peer),
        )

    def _ensure_transport_ready(self) -> None:
        """Eagerly initialize model and P2P NCCL channels in one order.

        NCCL documents that the first batched P2P use of a communicator must
        involve all its ranks.  A tiny all-reduce satisfies that requirement
        before source-colored ranks begin using different transport channels.
        TP/CP/EP and parameter-replica groups are initialized first because
        otherwise one pipeline stage may enter its first model/FSDP collective
        while the next stage posts a P2P receive on another lazy NCCL
        communicator.  Gloo and test doubles do not need this warmup.
        """

        if self._transport_ready:
            return
        if (
            int(self.group.size) <= 1
            or "nccl" not in str(getattr(self.group, "backend", "")).lower()
        ):
            self._transport_ready = True
            return

        color_count = 2 if int(self.group.size) % 2 == 0 else 3
        seen: set[int] = set()
        token = torch.zeros((), dtype=torch.int32, device=self.device)
        model_groups = []
        for name in ("tp", "cp", "ep", "dense_replica", "expert_replica"):
            group = getattr(self.parallel, name, None)
            if group is None:
                continue
            group = require_parallel_group(group, name=f"{name}-parallel group")
            if int(group.size) > 1 and "nccl" in str(group.backend).lower():
                model_groups.append(group)
        for group in (*model_groups, *self._transport_groups[:color_count]):
            raw_group = _raw_group(group)
            identity = id(raw_group)
            if identity in seen:
                continue
            seen.add(identity)
            dist.all_reduce(token, group=raw_group)
        self._transport_ready = True

    def _launch(self, entries: Sequence[_WireOperation]) -> list[Any]:
        """Launch per-communicator batches and align Work objects to entries."""

        if not entries:
            return []

        batches: dict[int, list[tuple[int, _WireOperation]]] = {}
        raw_groups: dict[int, dist.ProcessGroup | None] = {}
        for index, entry in enumerate(entries):
            raw_group = _raw_group(entry.group)
            identity = id(raw_group)
            batches.setdefault(identity, []).append((index, entry))
            raw_groups[identity] = raw_group

        # Every rank visits source-color communicators in the same order.  A
        # group omitted by this rank simply has no local operations this round.
        group_order: list[int] = []
        for group in self._transport_groups:
            identity = id(_raw_group(group))
            if identity in batches and identity not in group_order:
                group_order.append(identity)
        group_order.extend(identity for identity in batches if identity not in group_order)

        aligned: list[Any | None] = [None] * len(entries)
        for identity in group_order:
            batch = batches[identity]
            operations = [
                dist.P2POp(
                    dist.isend if entry.send else dist.irecv,
                    entry.tensor,
                    entry.peer,
                    raw_groups[identity],
                )
                for _, entry in batch
            ]
            requests = list(dist.batch_isend_irecv(operations))
            if len(requests) == 1 and len(batch) > 1:
                # NCCL commonly returns one coalesced Work for the whole batch.
                requests = requests * len(batch)
            elif len(requests) != len(batch):
                raise RuntimeError(
                    "batch_isend_irecv returned an unexpected number of requests: "
                    f"{len(requests)} for {len(batch)} operations"
                )
            for (index, _), request in zip(batch, requests, strict=True):
                aligned[index] = request
        if any(request is None for request in aligned):
            raise RuntimeError("pipeline P2P request alignment is incomplete")
        return list(aligned)

    @staticmethod
    def _wait_requests(requests: Sequence[Any]) -> None:
        waited: set[int] = set()
        for request in requests:
            identity = id(request)
            if identity in waited:
                continue
            request.wait()
            waited.add(identity)

    def start_exchange(
        self,
        *,
        sends: Sequence[P2PSend] = (),
        receives: Sequence[P2PReceive] = (),
        causal_turnaround: bool = False,
    ) -> PendingP2P:
        """Launch one canonical forward/backward exchange.

        At most one send and receive of each semantic kind is accepted.  That
        constraint makes operation ordering deterministic even when PP=2 and
        the cyclic virtual-pipeline previous/next peer is the same rank.

        ``causal_turnaround`` is set by the schedule executor only when the
        receive cannot be produced until the peer consumes a send in this same
        exchange.  Dynamic-shape transports then complete those sends before
        posting the causally-future receive header.  Direct communicator users
        default to a fully asynchronous, non-causal exchange.
        """

        send_kinds = [item.metadata.kind for item in sends]
        receive_kinds = [item.metadata.kind for item in receives]
        if len(set(send_kinds)) != len(send_kinds):
            raise ValueError("one exchange cannot send the same pipeline message kind twice")
        if len(set(receive_kinds)) != len(receive_kinds):
            raise ValueError("one exchange cannot receive the same pipeline message kind twice")

        if sends and receives and "nccl" in str(getattr(self.group, "backend", "")).lower():
            send_groups = {
                id(_raw_group(self._transport_group(send=True, peer=int(item.peer))))
                for item in sends
            }
            receive_groups = {
                id(_raw_group(self._transport_group(send=False, peer=int(item.peer))))
                for item in receives
            }
            if send_groups.intersection(receive_groups):
                raise ValueError(
                    "an NCCL pipeline exchange cannot send and receive on the same "
                    "source-color communicator; peers must be adjacent in the "
                    "physical linear/cyclic PP topology"
                )

        self._ensure_transport_ready()
        ordered = self._ordered_entries(sends, receives)
        wire_sends = {
            item.metadata.kind: self._wire_tensor(
                item.tensor,
                name=(
                    "forward tensor"
                    if item.metadata.kind is P2PMessageKind.FORWARD
                    else "backward gradient"
                ),
            )
            for item in sends
        }
        receive_shapes: dict[P2PMessageKind, tuple[int, ...]] = {}
        keepalive: list[Tensor] = list(wire_sends.values())
        send_requests: list[Any] = []
        receive_requests: dict[P2PMessageKind, Any] = {}

        if self.dynamic_shapes:
            header_sends = {
                item.metadata.kind: self._encode_header(
                    item.metadata, wire_sends[item.metadata.kind].shape
                )
                for item in sends
            }
            header_receives = {
                item.metadata.kind: torch.empty(_HEADER_SIZE, dtype=torch.int64, device=self.device)
                for item in receives
            }
            header_entries: list[_WireOperation] = []
            header_send_entries: list[_WireOperation] = []
            header_receive_entries: list[_WireOperation] = []
            header_labels: list[tuple[bool, P2PMessageKind]] = []
            for _, send, item in ordered:
                if send:
                    assert isinstance(item, P2PSend)
                    operation = self._wire_operation(
                        True,
                        header_sends[item.metadata.kind],
                        int(item.peer),
                    )
                    header_send_entries.append(operation)
                else:
                    assert isinstance(item, P2PReceive)
                    operation = self._wire_operation(
                        False,
                        header_receives[item.metadata.kind],
                        int(item.peer),
                    )
                    header_receive_entries.append(operation)
                header_entries.append(operation)
                header_labels.append((send, item.metadata.kind))

            # Header and payload sends use the same source-colored communicator
            # and are launched in that order.  Usually both stay asynchronous:
            # an interleaved peer may need another local compute before it
            # posts this receive.  A same-microbatch Forward/Backward exchange
            # is different: the peer cannot produce the gradient until it has
            # consumed this activation.  Completing the tiny activation header
            # first ensures the peer can post its payload receive before a
            # future-gradient receive kernel occupies the other NCCL channel.
            # This avoids a GPU-side F(payload) -> compute -> B(header) cycle at
            # the GPipe/1F1B turnaround without serializing steady F_new/B_old.
            is_turnaround = bool(causal_turnaround and sends and receives)
            if is_turnaround:
                # Do not even enqueue the causally-future receive yet.  A
                # persistent NCCL receive kernel on its independent channel can
                # otherwise occupy the device before the activation payload
                # needed to produce that message has made progress.
                header_send_requests = self._launch(header_send_entries)
                self._wait_requests(header_send_requests)
            else:
                header_requests = self._launch(header_entries)
                header_send_requests = [
                    request
                    for request, (send, _) in zip(
                        header_requests,
                        header_labels,
                        strict=True,
                    )
                    if send
                ]
                send_requests.extend(header_send_requests)
            payload_send_entries: list[_WireOperation] = []
            for kind, send, item in ordered:
                if not send:
                    continue
                assert isinstance(item, P2PSend)
                payload_send_entries.append(
                    self._wire_operation(True, wire_sends[kind], int(item.peer))
                )
            payload_send_requests = self._launch(payload_send_entries)
            if is_turnaround:
                self._wait_requests(payload_send_requests)
                header_receive_requests = self._launch(header_receive_entries)
            else:
                send_requests.extend(payload_send_requests)
                header_receive_requests = [
                    request
                    for request, (send, _) in zip(
                        header_requests,
                        header_labels,
                        strict=True,
                    )
                    if not send
                ]

            # A future receive header may depend on the payload queued above.
            # Source coloring prevents NCCL from coalescing these receive works
            # with our still-live send works.
            self._wait_requests(header_receive_requests)
            for item in receives:
                receive_shapes[item.metadata.kind] = self._decode_header(
                    header_receives[item.metadata.kind], item.metadata
                )
            keepalive.extend(header_sends.values())
            keepalive.extend(header_receives.values())
        else:
            assert self.activation_shape is not None
            receive_shapes = {item.metadata.kind: self.activation_shape for item in receives}

        receive_buffers: dict[P2PMessageKind, Tensor] = {
            item.metadata.kind: torch.empty(
                receive_shapes[item.metadata.kind],
                dtype=self.activation_dtype,
                device=self.device,
            )
            for item in receives
        }
        if self.dynamic_shapes:
            payload_receive_entries: list[_WireOperation] = []
            payload_receive_kinds: list[P2PMessageKind] = []
            for kind, send, item in ordered:
                if send:
                    continue
                assert isinstance(item, P2PReceive)
                payload_receive_entries.append(
                    self._wire_operation(False, receive_buffers[kind], int(item.peer))
                )
                payload_receive_kinds.append(kind)
            for request, kind in zip(
                self._launch(payload_receive_entries),
                payload_receive_kinds,
                strict=True,
            ):
                receive_requests[kind] = request
        else:
            payload_entries: list[_WireOperation] = []
            payload_labels: list[tuple[bool, P2PMessageKind]] = []
            for kind, send, item in ordered:
                if send:
                    assert isinstance(item, P2PSend)
                    tensor = wire_sends[kind]
                else:
                    assert isinstance(item, P2PReceive)
                    tensor = receive_buffers[kind]
                payload_entries.append(self._wire_operation(send, tensor, int(item.peer)))
                payload_labels.append((send, kind))
            for request, (send, kind) in zip(
                self._launch(payload_entries), payload_labels, strict=True
            ):
                if send:
                    send_requests.append(request)
                else:
                    receive_requests[kind] = request
        keepalive.extend(receive_buffers.values())
        return PendingP2P(
            send_requests=send_requests,
            receive_requests=receive_requests,
            receive_buffers=receive_buffers,
            receive_requires_grad={item.metadata.kind: item.requires_grad for item in receives},
            keepalive=keepalive,
        )

    def start_send_forward(
        self,
        tensor: Tensor,
        *,
        peer: int | None = None,
        microbatch: int = -1,
        chunk: int = -1,
    ) -> PendingP2P:
        peer = self._next_rank() if peer is None else peer
        if peer is None:
            return PendingP2P()
        return self.start_exchange(
            sends=(
                P2PSend(
                    tensor,
                    peer,
                    P2PMetadata(P2PMessageKind.FORWARD, microbatch, chunk),
                ),
            )
        )

    def start_recv_forward(
        self,
        *,
        peer: int | None = None,
        microbatch: int = -1,
        chunk: int = -1,
    ) -> PendingP2P:
        peer = self._prev_rank() if peer is None else peer
        if peer is None:
            return PendingP2P()
        return self.start_exchange(
            receives=(
                P2PReceive(
                    peer,
                    P2PMetadata(P2PMessageKind.FORWARD, microbatch, chunk),
                    requires_grad=True,
                ),
            )
        )

    def start_send_backward(
        self,
        gradient: Tensor,
        *,
        peer: int | None = None,
        microbatch: int = -1,
        chunk: int = -1,
    ) -> PendingP2P:
        peer = self._prev_rank() if peer is None else peer
        if peer is None:
            return PendingP2P()
        return self.start_exchange(
            sends=(
                P2PSend(
                    gradient,
                    peer,
                    P2PMetadata(P2PMessageKind.BACKWARD, microbatch, chunk),
                ),
            )
        )

    def start_recv_backward(
        self,
        *,
        peer: int | None = None,
        microbatch: int = -1,
        chunk: int = -1,
    ) -> PendingP2P:
        peer = self._next_rank() if peer is None else peer
        if peer is None:
            return PendingP2P()
        return self.start_exchange(
            receives=(
                P2PReceive(
                    peer,
                    P2PMetadata(P2PMessageKind.BACKWARD, microbatch, chunk),
                ),
            )
        )

    def send_forward(
        self,
        tensor: Tensor,
        *,
        microbatch: int = -1,
        chunk: int = -1,
    ) -> None:
        self.start_send_forward(
            tensor,
            microbatch=microbatch,
            chunk=chunk,
        ).wait_sends()

    def recv_forward(
        self,
        *,
        microbatch: int = -1,
        chunk: int = -1,
    ) -> Tensor | None:
        return self.start_recv_forward(
            microbatch=microbatch,
            chunk=chunk,
        ).wait_receive(P2PMessageKind.FORWARD)

    def send_backward(
        self,
        gradient: Tensor,
        *,
        microbatch: int = -1,
        chunk: int = -1,
    ) -> None:
        self.start_send_backward(
            gradient,
            microbatch=microbatch,
            chunk=chunk,
        ).wait_sends()

    def recv_backward(
        self,
        *,
        microbatch: int = -1,
        chunk: int = -1,
    ) -> Tensor | None:
        return self.start_recv_backward(
            microbatch=microbatch,
            chunk=chunk,
        ).wait_receive(P2PMessageKind.BACKWARD)

    def send_forward_recv_backward(
        self,
        output: Tensor,
        *,
        send_microbatch: int = -1,
        recv_microbatch: int = -1,
        chunk: int = -1,
    ) -> Tensor | None:
        peer = self._next_rank()
        if peer is None:
            return None
        pending = self.start_exchange(
            sends=(
                P2PSend(
                    output,
                    peer,
                    P2PMetadata(P2PMessageKind.FORWARD, send_microbatch, chunk),
                ),
            ),
            receives=(
                P2PReceive(
                    peer,
                    P2PMetadata(P2PMessageKind.BACKWARD, recv_microbatch, chunk),
                ),
            ),
        )
        return pending.wait().backward

    def send_backward_recv_forward(
        self,
        gradient: Tensor | None,
        *,
        receive_forward: bool = True,
        send_microbatch: int = -1,
        recv_microbatch: int = -1,
        chunk: int = -1,
    ) -> Tensor | None:
        peer = self._prev_rank()
        if peer is None:
            return None
        sends = (
            ()
            if gradient is None
            else (
                P2PSend(
                    gradient,
                    peer,
                    P2PMetadata(P2PMessageKind.BACKWARD, send_microbatch, chunk),
                ),
            )
        )
        receives = (
            ()
            if not receive_forward
            else (
                P2PReceive(
                    peer,
                    P2PMetadata(P2PMessageKind.FORWARD, recv_microbatch, chunk),
                    requires_grad=True,
                ),
            )
        )
        return self.start_exchange(sends=sends, receives=receives).wait().forward
