from contextlib import contextmanager

import pytest

torch = pytest.importorskip("torch")

from nano_megatron.pipeline_parallel import (  # noqa: E402
    GPipeSchedule,
    OneForwardOneBackwardSchedule,
    P2PCommunicator,
    PipelineEvent,
    PipelineEventKind,
    PipelineWork,
    StepOutput,
    VirtualPipelineLayout,
    build_gpipe_plan,
    build_interleaved_plan,
    build_interleaved_schedule_table,
    partition_layers,
)
from nano_megatron.pipeline_parallel.schedules import build_1f1b_plan  # noqa: E402
from nano_megatron.pipeline_parallel.schedules.executor import (  # noqa: E402
    VirtualPipelineRoute,
)
from nano_megatron.training.trainer import _compute_context  # noqa: E402


def test_partition_layers_is_balanced_and_complete() -> None:
    partitions = partition_layers(10, 3)
    assert [partition.num_layers for partition in partitions] == [4, 3, 3]
    assert partitions[0].owns_embedding
    assert partitions[-1].owns_lm_head
    assert [(p.start_layer, p.end_layer) for p in partitions] == [(0, 4), (4, 7), (7, 10)]


def test_partition_rejects_empty_stages() -> None:
    with pytest.raises(ValueError, match="empty stages"):
        partition_layers(2, 3)


def test_virtual_pipeline_layout_maps_chunks_without_a_global_virtual_rank() -> None:
    layout = VirtualPipelineLayout(
        num_layers=8,
        pipeline_size=2,
        virtual_stages_per_rank=2,
    )

    assert [
        (partition.start_layer, partition.end_layer) for partition in layout.local_partitions(0)
    ] == [(0, 2), (4, 6)]
    assert [
        (partition.start_layer, partition.end_layer) for partition in layout.local_partitions(1)
    ] == [(2, 4), (6, 8)]
    wrap = layout.next(layout.address(1, 0))
    assert wrap == layout.address(0, 1)
    assert layout.previous(wrap) == layout.address(1, 0)
    assert layout.is_first(layout.address(0, 0))
    assert layout.is_last(layout.address(1, 1))


def test_interleaved_plan_has_one_forward_and_backward_for_every_work_key() -> None:
    table = build_interleaved_schedule_table(4, 2, 2)
    assert [(item.microbatch, item.chunk) for item in table] == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (2, 0),
        (3, 0),
        (2, 1),
        (3, 1),
    ]
    for rank in range(2):
        events = build_interleaved_plan(
            pipeline_size=2,
            pipeline_rank=rank,
            num_microbatches=4,
            num_chunks=2,
        )
        assert all(isinstance(event, PipelineEvent) for event in events)
        actions = [action for event in events for action in event.actions]
        forwards = [action.work for action in actions if action.kind is PipelineEventKind.FORWARD]
        backwards = [action.work for action in actions if action.kind is PipelineEventKind.BACKWARD]
        assert set(forwards) == set(table)
        assert set(backwards) == set(table)
        positions = {work: index for index, work in enumerate(forwards)}
        seen: set[PipelineWork] = set()
        for event in events:
            for action in event.actions:
                if action.kind is PipelineEventKind.FORWARD:
                    seen.add(action.work)
                else:
                    assert action.work in seen
        steady_events = [event for event in events if event.phase == "steady"]
        assert all(event.forward is not None for event in steady_events)
        assert all(event.backward is not None for event in steady_events)
        assert all(len(event.actions) == 2 for event in steady_events)
        assert len(positions) == 8


@pytest.mark.parametrize("num_microbatches", [1, 3])
def test_interleaved_forward_only_accepts_small_or_partial_microbatch_groups(
    num_microbatches: int,
) -> None:
    events = build_interleaved_plan(
        pipeline_size=2,
        pipeline_rank=0,
        num_microbatches=num_microbatches,
        num_chunks=2,
        forward_only=True,
    )

    actions = [action for event in events for action in event.actions]
    work = [action.work for action in actions]
    assert len(work) == num_microbatches * 2
    assert all(action.kind is PipelineEventKind.FORWARD for action in actions)
    assert all(len(event.actions) == 1 for event in events)
    assert {item.microbatch for item in work} == set(range(num_microbatches))


def test_virtual_pipeline_route_translates_pp_local_peers_and_chunks() -> None:
    group = type(
        "Group",
        (),
        {
            "rank": 0,
            "size": 2,
            "global_rank_at": staticmethod(lambda rank: (2, 5)[rank]),
        },
    )()
    parallel = type("Parallel", (), {"pp": group})()
    layout = VirtualPipelineLayout(4, 2, 2)
    route = VirtualPipelineRoute(parallel, layout)

    first = PipelineWork(0, 0)
    later = PipelineWork(0, 1)
    assert route.previous(first) is None
    assert (route.following(first).peer, route.following(first).chunk) == (5, 0)
    assert (route.previous(later).peer, route.previous(later).chunk) == (5, 0)
    assert (route.following(later).peer, route.following(later).chunk) == (5, 1)


def test_pipeline_objects_require_explicit_pp_group() -> None:
    parallel = object()
    with pytest.raises(TypeError, match="explicit PP group"):
        P2PCommunicator(
            parallel,
            activation_shape=(1, 2, 3),
            activation_dtype=torch.float32,
            device="cpu",
        )
    with pytest.raises(TypeError, match="explicit PP group"):
        OneForwardOneBackwardSchedule(parallel)


def test_pipeline_communicator_rejects_unmaterialized_multi_rank_group() -> None:
    group = type(
        "Group",
        (),
        {"rank": 0, "size": 2, "process_group": None},
    )()
    parallel = type("Parallel", (), {"pp": group})()
    with pytest.raises(ValueError, match="materialized process_group"):
        P2PCommunicator(
            parallel,
            activation_shape=(1, 2, 3),
            activation_dtype=torch.float32,
            device="cpu",
        )


def test_1f1b_plan_has_one_forward_and_backward_per_microbatch() -> None:
    events = build_1f1b_plan(pipeline_size=4, pipeline_rank=1, num_microbatches=6)
    assert all(len(event.actions) == 1 for event in events)
    assert [
        (event.phase, action.kind, action.work) for event in events for action in event.actions
    ] == [
        ("warmup", PipelineEventKind.FORWARD, PipelineWork(0)),
        ("warmup", PipelineEventKind.FORWARD, PipelineWork(1)),
        ("steady", PipelineEventKind.FORWARD, PipelineWork(2)),
        ("steady", PipelineEventKind.BACKWARD, PipelineWork(0)),
        ("steady", PipelineEventKind.FORWARD, PipelineWork(3)),
        ("steady", PipelineEventKind.BACKWARD, PipelineWork(1)),
        ("steady", PipelineEventKind.FORWARD, PipelineWork(4)),
        ("steady", PipelineEventKind.BACKWARD, PipelineWork(2)),
        ("steady", PipelineEventKind.FORWARD, PipelineWork(5)),
        ("steady", PipelineEventKind.BACKWARD, PipelineWork(3)),
        ("cooldown", PipelineEventKind.BACKWARD, PipelineWork(4)),
        ("cooldown", PipelineEventKind.BACKWARD, PipelineWork(5)),
    ]


def test_gpipe_plan_uses_single_action_frames_in_forward_then_reverse_backward_order() -> None:
    events = build_gpipe_plan(3)

    assert all(len(event.actions) == 1 for event in events)
    assert [(action.kind, action.work) for event in events for action in event.actions] == [
        (PipelineEventKind.FORWARD, PipelineWork(0)),
        (PipelineEventKind.FORWARD, PipelineWork(1)),
        (PipelineEventKind.FORWARD, PipelineWork(2)),
        (PipelineEventKind.BACKWARD, PipelineWork(2)),
        (PipelineEventKind.BACKWARD, PipelineWork(1)),
        (PipelineEventKind.BACKWARD, PipelineWork(0)),
    ]


def test_gpipe_wraps_forward_in_activation_offload_context() -> None:
    class Parallel:
        @staticmethod
        def is_pipeline_first_stage() -> bool:
            return True

        @staticmethod
        def is_pipeline_last_stage() -> bool:
            return True

        @staticmethod
        def pipeline_prev_rank() -> None:
            return None

        @staticmethod
        def pipeline_next_rank() -> None:
            return None

    class Stage(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, hidden_states, batch):
            del hidden_states
            return self.weight * batch["value"]

    class Strategy:
        def __init__(self) -> None:
            self.activation_entries = 0

        @contextmanager
        def activation_context(self):
            self.activation_entries += 1
            yield

        @contextmanager
        def microbatch_context(self, *, is_last_microbatch: bool):
            del is_last_microbatch
            yield

        @staticmethod
        def backward(loss):
            loss.backward()

    strategy = Strategy()
    stage = Stage()
    GPipeSchedule(Parallel()).forward_backward(
        stage=stage,
        microbatches=[{"value": torch.tensor(1.0)}, {"value": torch.tensor(3.0)}],
        data_parallel=strategy,
    )
    assert strategy.activation_entries == 2
    assert stage.weight.grad.item() == pytest.approx(2.0)


def test_step_output_sums_already_scaled_microbatch_losses() -> None:
    output = StepOutput(
        losses=(torch.tensor(1.0), torch.tensor(2.0)),
        metrics={},
    )
    assert output.loss is not None
    assert output.loss.item() == pytest.approx(3.0)


def test_gpipe_reports_average_loss_metric_over_microbatches() -> None:
    class Parallel:
        @staticmethod
        def is_pipeline_first_stage() -> bool:
            return True

        @staticmethod
        def is_pipeline_last_stage() -> bool:
            return True

        @staticmethod
        def pipeline_prev_rank() -> None:
            return None

        @staticmethod
        def pipeline_next_rank() -> None:
            return None

    class Stage:
        def __call__(self, hidden_states, batch):
            del hidden_states
            return batch["value"]

    output = GPipeSchedule(Parallel()).forward_backward(
        stage=Stage(),
        microbatches=[{"value": torch.tensor(1.0)}, {"value": torch.tensor(3.0)}],
        forward_only=True,
    )
    assert output.metrics["loss"] == pytest.approx(2.0)


def test_gpipe_rejects_structured_outputs_at_the_sharding_boundary() -> None:
    class Parallel:
        @staticmethod
        def is_pipeline_first_stage() -> bool:
            return True

        @staticmethod
        def is_pipeline_last_stage() -> bool:
            return True

        @staticmethod
        def pipeline_prev_rank() -> None:
            return None

        @staticmethod
        def pipeline_next_rank() -> None:
            return None

    class StructuredOutput:
        def __init__(self, loss):
            self.loss = loss

    class Stage:
        def __call__(self, hidden_states, batch):
            del hidden_states
            return StructuredOutput(batch["value"])

    with pytest.raises(TypeError, match="pipeline stages must return a Tensor"):
        GPipeSchedule(Parallel()).forward_backward(
            stage=Stage(),
            microbatches=[{"value": torch.tensor(1.0)}],
            forward_only=True,
        )


def test_gpipe_requires_a_scalar_loss_from_the_last_stage() -> None:
    class Parallel:
        @staticmethod
        def is_pipeline_first_stage() -> bool:
            return True

        @staticmethod
        def is_pipeline_last_stage() -> bool:
            return True

        @staticmethod
        def pipeline_prev_rank() -> None:
            return None

        @staticmethod
        def pipeline_next_rank() -> None:
            return None

    class Stage:
        def __call__(self, hidden_states, batch):
            del hidden_states
            return batch["value"]

    with pytest.raises(TypeError, match="last pipeline stage must return a scalar loss Tensor"):
        GPipeSchedule(Parallel()).forward_backward(
            stage=Stage(),
            microbatches=[{"value": torch.ones(2)}],
            forward_only=True,
        )


def test_cpu_bfloat16_compute_context_autocasts_linear_output() -> None:
    linear = torch.nn.Linear(4, 3).float()
    inputs = torch.randn(2, 4)
    with _compute_context("cpu", torch.bfloat16):
        output = linear(inputs)
    assert output.dtype is torch.bfloat16


def test_float32_compute_context_is_a_noop() -> None:
    linear = torch.nn.Linear(4, 3).float()
    with _compute_context("cpu", torch.float32):
        output = linear(torch.randn(2, 4))
    assert output.dtype is torch.float32
