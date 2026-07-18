"""Common event executor for every pipeline schedule."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from torch import Tensor

from ..p2p import (
    P2PCommunicator,
    P2PMessageKind,
    P2PMetadata,
    P2PReceive,
    P2PSend,
    PendingP2P,
)
from ..partition import PipelineStageAddress, VirtualPipelineLayout
from ..stage import PipelineStage, StepOutput
from .base import (
    ComputeContext,
    accumulate_metrics,
    average_metrics,
    backward,
    data_parallel_context,
    extract_loss,
    finalize_pipeline_stage_gradients,
    forward_data_parallel_context,
    match_output_gradient,
    no_compute_context,
    prepare_pipeline_stage,
)


class PipelineEventKind(StrEnum):
    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass(frozen=True, order=True)
class PipelineWork:
    """One microbatch executed by one local model chunk."""

    microbatch: int
    chunk: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.microbatch, bool) or not isinstance(self.microbatch, int):
            raise TypeError("pipeline work microbatch must be an integer")
        if isinstance(self.chunk, bool) or not isinstance(self.chunk, int):
            raise TypeError("pipeline work chunk must be an integer")
        if self.microbatch < 0:
            raise ValueError("pipeline work microbatch cannot be negative")
        if self.chunk < 0:
            raise ValueError("pipeline work chunk cannot be negative")


@dataclass(frozen=True)
class PipelineAction:
    """One atomic Forward or Backward compute operation."""

    kind: PipelineEventKind
    work: PipelineWork

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", PipelineEventKind(self.kind))
        if not isinstance(self.work, PipelineWork):
            raise TypeError("pipeline action work must be a PipelineWork")


@dataclass(frozen=True)
class PipelineEvent:
    """One exchange frame containing a Forward, a Backward, or both."""

    phase: str
    forward: PipelineWork | None = None
    backward: PipelineWork | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, str) or not self.phase.strip():
            raise ValueError("pipeline event phase must be a non-empty string")
        if self.forward is None and self.backward is None:
            raise ValueError("pipeline event must contain at least one compute action")
        if self.forward is not None and not isinstance(self.forward, PipelineWork):
            raise TypeError("pipeline forward work must be a PipelineWork")
        if self.backward is not None and not isinstance(self.backward, PipelineWork):
            raise TypeError("pipeline backward work must be a PipelineWork")
        object.__setattr__(self, "phase", self.phase.strip())

    @property
    def actions(self) -> tuple[PipelineAction, ...]:
        actions: list[PipelineAction] = []
        if self.forward is not None:
            actions.append(PipelineAction(PipelineEventKind.FORWARD, self.forward))
        if self.backward is not None:
            actions.append(PipelineAction(PipelineEventKind.BACKWARD, self.backward))
        return tuple(actions)


@dataclass(frozen=True)
class PipelineEndpoint:
    """The physical peer and destination-local chunk for one logical edge."""

    peer: int
    chunk: int


@dataclass
class GraphRecord:
    input: Tensor | None
    output: Tensor


class PipelineRoute(Protocol):
    """Map schedule work onto local stages and logical pipeline edges."""

    def local_stage(self, root: PipelineStage, work: PipelineWork) -> PipelineStage: ...

    def data_parallel_unit(self, root: PipelineStage, work: PipelineWork) -> Any | None: ...

    def previous(self, work: PipelineWork) -> PipelineEndpoint | None: ...

    def following(self, work: PipelineWork) -> PipelineEndpoint | None: ...


class LinearPipelineRoute:
    """A single contiguous model stage on each physical PP rank."""

    def __init__(self, parallel: Any) -> None:
        self.parallel = parallel

    @staticmethod
    def _validate(work: PipelineWork) -> None:
        if work.chunk != 0:
            raise ValueError("a non-interleaved pipeline work item must use chunk 0")

    def local_stage(self, root: PipelineStage, work: PipelineWork) -> PipelineStage:
        self._validate(work)
        return root

    def data_parallel_unit(self, root: PipelineStage, work: PipelineWork) -> None:
        del root
        self._validate(work)
        return None

    def previous(self, work: PipelineWork) -> PipelineEndpoint | None:
        self._validate(work)
        is_first = getattr(self.parallel, "is_pipeline_first_stage", None)
        if callable(is_first) and is_first():
            return None
        previous_rank = getattr(self.parallel, "pipeline_prev_rank", None)
        if not callable(previous_rank):
            raise TypeError("a non-first pipeline stage must provide pipeline_prev_rank()")
        peer = previous_rank()
        return None if peer is None else PipelineEndpoint(int(peer), 0)

    def following(self, work: PipelineWork) -> PipelineEndpoint | None:
        self._validate(work)
        is_last = getattr(self.parallel, "is_pipeline_last_stage", None)
        if callable(is_last) and is_last():
            return None
        next_rank = getattr(self.parallel, "pipeline_next_rank", None)
        if not callable(next_rank):
            raise TypeError("a non-last pipeline stage must provide pipeline_next_rank()")
        peer = next_rank()
        return None if peer is None else PipelineEndpoint(int(peer), 0)


def _model_chunk(stage: Any, chunk_id: int) -> PipelineStage:
    current = stage
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        getter = getattr(current, "chunk", None)
        if callable(getter):
            return getter(chunk_id)
        current = getattr(current, "module", None)
    raise TypeError("interleaved schedule requires a model with chunk(id)")


class VirtualPipelineRoute:
    """Map local chunks onto an explicit virtual-pipeline layout."""

    def __init__(self, parallel: Any, layout: VirtualPipelineLayout) -> None:
        if layout.pipeline_size != int(parallel.pp.size):
            raise ValueError("virtual layout pipeline size does not match the PP group")
        self.parallel = parallel
        self.layout = layout
        self.rank = int(parallel.pp.rank)

    def _address(self, work: PipelineWork) -> PipelineStageAddress:
        return self.layout.address(self.rank, work.chunk)

    def _endpoint(self, address: PipelineStageAddress | None) -> PipelineEndpoint | None:
        if address is None:
            return None
        return PipelineEndpoint(
            int(self.parallel.pp.global_rank_at(address.pp_rank)),
            address.chunk,
        )

    def local_stage(self, root: PipelineStage, work: PipelineWork) -> PipelineStage:
        self._address(work)
        return _model_chunk(root, work.chunk)

    def data_parallel_unit(
        self,
        root: PipelineStage,
        work: PipelineWork,
    ) -> PipelineStage:
        return self.local_stage(root, work)

    def previous(self, work: PipelineWork) -> PipelineEndpoint | None:
        return self._endpoint(self.layout.previous(self._address(work)))

    def following(self, work: PipelineWork) -> PipelineEndpoint | None:
        return self._endpoint(self.layout.next(self._address(work)))


class _ExchangeSlot:
    """Own exactly one previous-send/current-receive exchange."""

    def __init__(
        self,
        start_exchange: Callable[..., PendingP2P],
        *,
        overlap: bool,
    ) -> None:
        self._start_exchange = start_exchange
        self._overlap = overlap
        self._pending = PendingP2P()

    def prime(self, receives: Sequence[P2PReceive]) -> None:
        self.launch(sends=(), receives=receives)

    def consume(self, kind: PipelineEventKind) -> Tensor | None:
        if not self._overlap:
            self._pending.wait()
        message_kind = (
            P2PMessageKind.FORWARD if kind is PipelineEventKind.FORWARD else P2PMessageKind.BACKWARD
        )
        return self._pending.wait_receive(message_kind)

    def drain_prior_sends(self) -> None:
        self._pending.wait_sends()

    def launch(
        self,
        *,
        sends: Sequence[P2PSend],
        receives: Sequence[P2PReceive],
        causal_turnaround: bool = False,
    ) -> None:
        if not self._pending.done:
            raise RuntimeError(
                "cannot replace a pipeline exchange before its receive and sends finish"
            )
        self._pending = self._start_exchange(
            sends=sends,
            receives=receives,
            causal_turnaround=causal_turnaround,
        )

    def finish(self) -> None:
        self._pending.wait()


class PipelineScheduleExecutor:
    """Execute exchange frames with one shared P2P request lifecycle.

    Every frame follows the same state transition:

    1. consume the receive for each action immediately before that action;
    2. execute the Forward action, then the Backward action when present;
    3. drain sends launched by the previous frame before reusing the send slot;
    4. launch this frame's sends together with the next frame's receives.

    With overlap enabled, the previous sends remain live during step 2.  The
    synchronous reference path uses the same state machine and simply completes
    the whole pending exchange before the frame's compute.
    """

    def __init__(
        self,
        *,
        route: PipelineRoute,
        communicator: P2PCommunicator | None,
        compute_context: ComputeContext | None = None,
        overlap_p2p: bool = False,
    ) -> None:
        self.route = route
        self.communicator = communicator
        self.compute_context = compute_context or no_compute_context
        self.overlap_p2p = bool(overlap_p2p)

    def run(
        self,
        *,
        stage: PipelineStage,
        microbatches: Sequence[Any],
        events: Sequence[PipelineEvent],
        data_parallel: Any = None,
        forward_only: bool = False,
    ) -> StepOutput:
        event_plan = tuple(events)
        self._validate_plan(
            event_plan,
            num_microbatches=len(microbatches),
            forward_only=forward_only,
        )
        synchronize_works = self._gradient_sync_works(stage, event_plan)
        graphs: dict[PipelineWork, GraphRecord] = {}
        losses: list[Tensor] = []
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}

        prepare_pipeline_stage(stage)
        slot = _ExchangeSlot(self._start_exchange, overlap=self.overlap_p2p)
        slot.prime(self._receives_for(event_plan[0]))
        for index, event in enumerate(event_plan):
            sends: list[P2PSend] = []
            for action in event.actions:
                input_tensor = slot.consume(action.kind)
                send: P2PSend | None
                if action.kind is PipelineEventKind.FORWARD:
                    send = self._run_forward(
                        stage=stage,
                        work=action.work,
                        received=input_tensor,
                        batch=microbatches[action.work.microbatch],
                        divisor=len(microbatches),
                        data_parallel=data_parallel,
                        synchronize_gradients=(forward_only or action.work in synchronize_works),
                        record_graph=not forward_only,
                        graphs=graphs,
                        losses=losses,
                        metric_sums=metric_sums,
                        metric_counts=metric_counts,
                    )
                else:
                    send = self._run_backward(
                        stage=stage,
                        work=action.work,
                        received=input_tensor,
                        data_parallel=data_parallel,
                        synchronize_gradients=action.work in synchronize_works,
                        graphs=graphs,
                    )
                if send is not None:
                    sends.append(send)

            # The previous sends have overlapped this frame's compute.  Complete
            # them before occupying the single bounded send slot again.
            slot.drain_prior_sends()
            next_event = event_plan[index + 1] if index + 1 < len(event_plan) else None
            next_receives = self._receives_for(next_event)
            causal_turnaround = bool(
                sends
                and next_receives
                and event.forward is not None
                and next_event is not None
                and next_event.backward == event.forward
            )
            slot.launch(
                sends=sends,
                receives=next_receives,
                causal_turnaround=causal_turnaround,
            )

        slot.finish()
        if graphs:
            raise RuntimeError(f"pipeline schedule left {len(graphs)} forward graphs undrained")
        if not forward_only:
            finalize_pipeline_stage_gradients(stage)
        return StepOutput(tuple(losses), average_metrics(metric_sums, metric_counts))

    def _run_forward(
        self,
        *,
        stage: PipelineStage,
        work: PipelineWork,
        received: Tensor | None,
        batch: Any,
        divisor: int,
        data_parallel: Any,
        synchronize_gradients: bool,
        record_graph: bool,
        graphs: dict[PipelineWork, GraphRecord],
        losses: list[Tensor],
        metric_sums: dict[str, float],
        metric_counts: dict[str, int],
    ) -> P2PSend | None:
        previous = self.route.previous(work)
        following = self.route.following(work)
        if previous is not None and received is None:
            raise RuntimeError(f"forward work {work} did not receive its pipeline input")
        hidden = None if previous is None else received
        local_stage = self.route.local_stage(stage, work)
        unit = self.route.data_parallel_unit(stage, work)
        with (
            forward_data_parallel_context(
                data_parallel,
                synchronize_gradients=synchronize_gradients,
                unit=unit,
            ),
            self.compute_context(),
        ):
            output = local_stage(hidden, batch)

        if following is None:
            output, local_metrics = extract_loss(output, divisor, batch)
            losses.append(output)
            accumulate_metrics(metric_sums, metric_counts, local_metrics)
        elif not isinstance(output, Tensor):
            raise TypeError("non-last pipeline stages must return a Tensor")

        if record_graph:
            if work in graphs:
                raise RuntimeError(f"forward graph {work} was recorded twice")
            graphs[work] = GraphRecord(hidden, output)
        if following is None:
            return None
        return P2PSend(
            output,
            following.peer,
            P2PMetadata(
                P2PMessageKind.FORWARD,
                work.microbatch,
                following.chunk,
            ),
        )

    def _run_backward(
        self,
        *,
        stage: PipelineStage,
        work: PipelineWork,
        received: Tensor | None,
        data_parallel: Any,
        synchronize_gradients: bool,
        graphs: dict[PipelineWork, GraphRecord],
    ) -> P2PSend | None:
        try:
            graph = graphs.pop(work)
        except KeyError as error:
            raise RuntimeError(f"backward work {work} has no matching forward graph") from error
        previous = self.route.previous(work)
        following = self.route.following(work)
        if following is not None and received is None:
            raise RuntimeError(f"backward work {work} did not receive its output gradient")
        output_gradient = match_output_gradient(
            graph.output,
            None if following is None else received,
        )
        unit = self.route.data_parallel_unit(stage, work)
        with data_parallel_context(
            data_parallel,
            is_last_microbatch=synchronize_gradients,
            unit=unit,
        ):
            backward(data_parallel, graph.output, output_gradient)

        if previous is None:
            return None
        if graph.input is None or graph.input.grad is None:
            raise RuntimeError("pipeline input gradient was not produced")
        return P2PSend(
            graph.input.grad,
            previous.peer,
            P2PMetadata(
                P2PMessageKind.BACKWARD,
                work.microbatch,
                previous.chunk,
            ),
        )

    def _receives_for(self, event: PipelineEvent | None) -> tuple[P2PReceive, ...]:
        if event is None:
            return ()
        receives: list[P2PReceive] = []
        for action in event.actions:
            if action.kind is PipelineEventKind.FORWARD:
                previous = self.route.previous(action.work)
                if previous is not None:
                    receives.append(
                        P2PReceive(
                            previous.peer,
                            P2PMetadata(
                                P2PMessageKind.FORWARD,
                                action.work.microbatch,
                                action.work.chunk,
                            ),
                            requires_grad=True,
                        )
                    )
                continue
            following = self.route.following(action.work)
            if following is not None:
                receives.append(
                    P2PReceive(
                        following.peer,
                        P2PMetadata(
                            P2PMessageKind.BACKWARD,
                            action.work.microbatch,
                            action.work.chunk,
                        ),
                    )
                )
        return tuple(receives)

    def _start_exchange(
        self,
        *,
        sends: Sequence[P2PSend] = (),
        receives: Sequence[P2PReceive] = (),
        causal_turnaround: bool = False,
    ) -> PendingP2P:
        if self.communicator is None:
            if sends or receives:
                raise ValueError(
                    "a P2PCommunicator is required for a pipeline plan with remote edges"
                )
            return PendingP2P()
        if causal_turnaround:
            return self.communicator.start_exchange(
                sends=sends,
                receives=receives,
                causal_turnaround=True,
            )
        return self.communicator.start_exchange(sends=sends, receives=receives)

    def _gradient_sync_works(
        self,
        stage: PipelineStage,
        events: Sequence[PipelineEvent],
    ) -> frozenset[PipelineWork]:
        last_by_unit: dict[int | None, PipelineWork] = {}
        for event in events:
            for action in event.actions:
                if action.kind is PipelineEventKind.BACKWARD:
                    unit = self.route.data_parallel_unit(stage, action.work)
                    last_by_unit[None if unit is None else id(unit)] = action.work
        return frozenset(last_by_unit.values())

    @staticmethod
    def _validate_plan(
        events: Sequence[PipelineEvent],
        *,
        num_microbatches: int,
        forward_only: bool,
    ) -> None:
        if num_microbatches < 1:
            raise ValueError("at least one microbatch is required")
        if not events:
            raise ValueError("a pipeline schedule must contain at least one event")
        forwards: set[PipelineWork] = set()
        backwards: set[PipelineWork] = set()
        for event in events:
            for action in event.actions:
                if not 0 <= action.work.microbatch < num_microbatches:
                    raise ValueError(
                        f"pipeline event microbatch {action.work.microbatch} is outside "
                        f"[0, {num_microbatches})"
                    )
                if action.kind is PipelineEventKind.FORWARD:
                    if action.work in forwards:
                        raise ValueError(f"forward work {action.work} appears more than once")
                    forwards.add(action.work)
                    continue
                if forward_only:
                    raise ValueError("a forward-only pipeline plan cannot contain backward events")
                if action.work not in forwards:
                    raise ValueError(
                        f"backward work {action.work} appears before its local forward"
                    )
                if action.work in backwards:
                    raise ValueError(f"backward work {action.work} appears more than once")
                backwards.add(action.work)
        if not forward_only and forwards != backwards:
            missing = sorted(forwards.difference(backwards))
            extra = sorted(backwards.difference(forwards))
            raise ValueError(
                "pipeline training plan must pair every local forward and backward: "
                f"missing_backward={missing}, extra_backward={extra}"
            )
