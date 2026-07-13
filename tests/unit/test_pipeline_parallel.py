from contextlib import contextmanager

import pytest

torch = pytest.importorskip("torch")

from nano_megatron.pipeline_parallel import (  # noqa: E402
    GPipeSchedule,
    OneForwardOneBackwardSchedule,
    P2PCommunicator,
    StepOutput,
    partition_layers,
)
from nano_megatron.pipeline_parallel.schedules import build_1f1b_plan  # noqa: E402
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
    assert sum(event.kind == "forward" for event in events) == 6
    assert sum(event.kind == "backward" for event in events) == 6
    assert [event.phase for event in events[:2]] == ["warmup", "warmup"]


def test_gpipe_wraps_forward_in_activation_offload_context() -> None:
    class Parallel:
        @staticmethod
        def is_pipeline_first_stage() -> bool:
            return True

        @staticmethod
        def is_pipeline_last_stage() -> bool:
            return True

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
