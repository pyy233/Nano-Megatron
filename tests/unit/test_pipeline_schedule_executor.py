from __future__ import annotations

import ast
import importlib
import inspect
from dataclasses import dataclass
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from nano_megatron.pipeline_parallel import (  # noqa: E402
    GPipeSchedule,
    InterleavedOneForwardOneBackwardSchedule,
    OneForwardOneBackwardSchedule,
    P2PMessageKind,
    P2PMetadata,
    P2PSend,
    PipelineEvent,
    PipelineScheduleExecutor,
    PipelineWork,
    VirtualPipelineLayout,
    build_1f1b_plan,
    build_gpipe_plan,
    build_interleaved_plan,
)
from nano_megatron.pipeline_parallel.schedules.executor import (  # noqa: E402
    LinearPipelineRoute,
    PipelineEndpoint,
    VirtualPipelineRoute,
)


@dataclass(frozen=True)
class _Group:
    rank: int
    size: int
    ranks: tuple[int, ...]
    process_group: object | None

    def global_rank_at(self, rank: int) -> int:
        return self.ranks[rank]


class _Parallel:
    def __init__(self, *, rank: int, ranks: tuple[int, ...]) -> None:
        self.pp = _Group(
            rank=rank,
            size=len(ranks),
            ranks=ranks,
            process_group=None if len(ranks) == 1 else object(),
        )

    def is_pipeline_first_stage(self) -> bool:
        return self.pp.rank == 0

    def is_pipeline_last_stage(self) -> bool:
        return self.pp.rank == self.pp.size - 1

    def pipeline_prev_rank(self) -> int | None:
        if self.is_pipeline_first_stage():
            return None
        return self.pp.global_rank_at(self.pp.rank - 1)

    def pipeline_next_rank(self) -> int | None:
        if self.is_pipeline_last_stage():
            return None
        return self.pp.global_rank_at(self.pp.rank + 1)


class _ChunkContainer:
    def __init__(self, num_chunks: int) -> None:
        self.chunks = tuple(object() for _ in range(num_chunks))

    def chunk(self, chunk_id: int) -> object:
        return self.chunks[chunk_id]


class _RecordingPending:
    def __init__(
        self,
        log: list[tuple[Any, ...]],
        identity: int,
        *,
        sends,
        receives,
    ) -> None:
        self.log = log
        self.identity = identity
        self.sends = tuple(sends)
        self.receives = tuple(receives)
        self._sends_done = not self.sends
        self._receives_done = not self.receives
        self._receive = (
            torch.tensor(float(identity + 1), requires_grad=True) if self.receives else None
        )

    @property
    def done(self) -> bool:
        return self._sends_done and self._receives_done

    def wait_receive(self, kind) -> Any:
        if not self._receives_done:
            assert len(self.receives) == 1
            assert self.receives[0].metadata.kind is kind
            self.log.append(("receive_wait", self.identity))
            self._receives_done = True
        return self._receive

    def wait_sends(self) -> None:
        if not self._sends_done:
            self.log.append(("send_wait", self.identity))
            self._sends_done = True

    def wait(self) -> None:
        self.log.append(("full_wait", self.identity))
        self._receives_done = True
        self._sends_done = True


class _RecordingCommunicator:
    def __init__(self, log: list[tuple[Any, ...]]) -> None:
        self.log = log
        self.calls = 0

    def start_exchange(
        self,
        *,
        sends=(),
        receives=(),
        causal_turnaround: bool = False,
    ) -> _RecordingPending:
        identity = self.calls
        self.calls += 1
        self.log.append(("launch", identity, len(sends), len(receives), causal_turnaround))
        return _RecordingPending(
            self.log,
            identity,
            sends=sends,
            receives=receives,
        )


class _RecordingStage:
    def __init__(self, log: list[tuple[Any, ...]]) -> None:
        self.log = log

    def __call__(self, hidden, batch):
        assert hidden is not None
        self.log.append(("compute", batch["index"]))
        return hidden * batch["scale"]


@pytest.mark.parametrize(
    ("overlap_p2p", "expected"),
    [
        (
            True,
            [
                ("launch", 0, 0, 1, False),
                ("receive_wait", 0),
                ("compute", 0),
                ("launch", 1, 1, 1, False),
                ("receive_wait", 1),
                ("compute", 1),
                ("send_wait", 1),
                ("launch", 2, 1, 0, False),
                ("full_wait", 2),
            ],
        ),
        (
            False,
            [
                ("launch", 0, 0, 1, False),
                ("full_wait", 0),
                ("compute", 0),
                ("launch", 1, 1, 1, False),
                ("full_wait", 1),
                ("compute", 1),
                ("launch", 2, 1, 0, False),
                ("full_wait", 2),
            ],
        ),
    ],
)
def test_executor_owns_one_receive_compute_send_lifecycle(
    overlap_p2p: bool,
    expected: list[tuple[Any, ...]],
) -> None:
    log: list[tuple[Any, ...]] = []
    parallel = _Parallel(rank=1, ranks=(2, 5, 9))
    executor = PipelineScheduleExecutor(
        route=LinearPipelineRoute(parallel),
        communicator=_RecordingCommunicator(log),  # type: ignore[arg-type]
        overlap_p2p=overlap_p2p,
    )

    executor.run(
        stage=_RecordingStage(log),
        microbatches=[
            {"index": 0, "scale": torch.tensor(2.0)},
            {"index": 1, "scale": torch.tensor(3.0)},
        ],
        events=build_gpipe_plan(2, forward_only=True),
        forward_only=True,
    )

    assert log == expected


def test_executor_marks_same_work_forward_to_backward_as_causal_turnaround() -> None:
    log: list[tuple[Any, ...]] = []

    class Route:
        @staticmethod
        def local_stage(root, work):
            del work
            return root

        @staticmethod
        def data_parallel_unit(root, work):
            del root, work
            return None

        @staticmethod
        def previous(work):
            del work
            return None

        @staticmethod
        def following(work):
            del work
            return PipelineEndpoint(9, 0)

    class Executor(PipelineScheduleExecutor):
        def _run_forward(self, **kwargs):
            work = kwargs["work"]
            log.append(("compute", P2PMessageKind.FORWARD))
            return P2PSend(
                torch.ones(()),
                9,
                P2PMetadata(P2PMessageKind.FORWARD, work.microbatch, work.chunk),
            )

        def _run_backward(self, **kwargs):
            del kwargs
            log.append(("compute", P2PMessageKind.BACKWARD))
            return None

    work = PipelineWork(0)
    Executor(
        route=Route(),  # type: ignore[arg-type]
        communicator=_RecordingCommunicator(log),  # type: ignore[arg-type]
        overlap_p2p=True,
    ).run(
        stage=object(),
        microbatches=({},),
        events=(
            PipelineEvent("forward", forward=work),
            PipelineEvent("backward", backward=work),
        ),
    )

    launches = [entry for entry in log if entry[0] == "launch"]
    assert launches == [
        ("launch", 0, 0, 0, False),
        ("launch", 1, 1, 1, True),
        ("launch", 2, 0, 0, False),
    ]


def test_paired_frame_waits_for_each_receive_immediately_before_its_compute() -> None:
    log: list[tuple[Any, ...]] = []

    class Route:
        @staticmethod
        def local_stage(root, work):
            del work
            return root

        @staticmethod
        def data_parallel_unit(root, work):
            del root, work
            return None

        @staticmethod
        def previous(work):
            del work
            return PipelineEndpoint(2, 0)

        @staticmethod
        def following(work):
            del work
            return PipelineEndpoint(9, 0)

    class Pending:
        def __init__(self, receives=()) -> None:
            self.expected = {receive.metadata.kind for receive in receives}
            self.completed: set[P2PMessageKind] = set()

        @property
        def done(self) -> bool:
            return self.completed == self.expected

        def wait_receive(self, kind):
            if kind not in self.expected:
                return None
            if kind not in self.completed:
                log.append(("receive_wait", kind))
                self.completed.add(kind)
            return torch.ones((), requires_grad=True)

        @staticmethod
        def wait_sends() -> None:
            return None

        def wait(self) -> None:
            self.completed.update(self.expected)

    class Communicator:
        @staticmethod
        def start_exchange(*, sends=(), receives=()):
            log.append(("launch", len(sends), len(receives)))
            return Pending(receives)

    class Executor(PipelineScheduleExecutor):
        def _run_forward(self, **kwargs):
            del kwargs
            log.append(("compute", P2PMessageKind.FORWARD))
            return None

        def _run_backward(self, **kwargs):
            del kwargs
            log.append(("compute", P2PMessageKind.BACKWARD))
            return None

    work = PipelineWork(0)
    Executor(
        route=Route(),  # type: ignore[arg-type]
        communicator=Communicator(),  # type: ignore[arg-type]
        overlap_p2p=True,
    ).run(
        stage=object(),
        microbatches=({},),
        events=(PipelineEvent("steady", forward=work, backward=work),),
    )

    assert log[:5] == [
        ("launch", 0, 2),
        ("receive_wait", P2PMessageKind.FORWARD),
        ("compute", P2PMessageKind.FORWARD),
        ("receive_wait", P2PMessageKind.BACKWARD),
        ("compute", P2PMessageKind.BACKWARD),
    ]


def test_gradient_sync_work_is_derived_from_each_units_last_backward() -> None:
    linear_parallel = _Parallel(rank=0, ranks=(0,))
    linear = PipelineScheduleExecutor(
        route=LinearPipelineRoute(linear_parallel),
        communicator=None,
    )
    stage = object()

    assert linear._gradient_sync_works(stage, build_gpipe_plan(4)) == frozenset(  # noqa: SLF001
        {PipelineWork(0)}
    )
    assert linear._gradient_sync_works(  # noqa: SLF001
        stage,
        build_1f1b_plan(
            pipeline_size=4,
            pipeline_rank=1,
            num_microbatches=6,
        ),
    ) == frozenset({PipelineWork(5)})

    virtual_parallel = _Parallel(rank=0, ranks=(2, 5))
    root = _ChunkContainer(2)
    virtual = PipelineScheduleExecutor(
        route=VirtualPipelineRoute(
            virtual_parallel,
            VirtualPipelineLayout(4, 2, 2),
        ),
        communicator=None,
    )
    assert virtual._gradient_sync_works(  # noqa: SLF001
        root,
        build_interleaved_plan(
            pipeline_size=2,
            pipeline_rank=0,
            num_microbatches=4,
            num_chunks=2,
        ),
    ) == frozenset({PipelineWork(3, 0), PipelineWork(3, 1)})


def test_all_schedule_wrappers_delegate_to_the_common_executor() -> None:
    linear_parallel = _Parallel(rank=0, ranks=(0,))
    virtual_parallel = _Parallel(rank=0, ranks=(2, 5))
    layout = VirtualPipelineLayout(4, 2, 2)

    schedules = (
        GPipeSchedule(linear_parallel),
        OneForwardOneBackwardSchedule(linear_parallel),
        InterleavedOneForwardOneBackwardSchedule(
            virtual_parallel,
            layout,
            communicator=object(),  # type: ignore[arg-type]
        ),
    )

    assert all(isinstance(schedule.executor, PipelineScheduleExecutor) for schedule in schedules)


def test_schedule_wrappers_do_not_own_p2p_request_lifecycles() -> None:
    modules = [
        importlib.import_module("nano_megatron.pipeline_parallel.schedules.gpipe"),
        importlib.import_module("nano_megatron.pipeline_parallel.schedules.one_f_one_b"),
        importlib.import_module("nano_megatron.pipeline_parallel.schedules.interleaved"),
    ]
    forbidden = {"PendingP2P", "start_exchange", "wait_receive", "wait_sends"}

    for module in modules:
        tree = ast.parse(inspect.getsource(module))
        referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        referenced.update(node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute))
        assert not forbidden.intersection(referenced), module.__name__

    one_f_one_b_source = inspect.getsource(modules[1])
    assert "GPipeSchedule" not in one_f_one_b_source
