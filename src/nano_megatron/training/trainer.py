"""Small training engine that keeps parallel and DP dependencies explicit."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nano_megatron.pipeline_parallel import (
    GPipeSchedule,
    InterleavedOneForwardOneBackwardSchedule,
    OneForwardOneBackwardSchedule,
    P2PCommunicator,
    StepOutput,
    prepare_pipeline_stage,
)

from .batch_router import BatchRouter
from .microbatches import split_microbatches


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def _torch_dtype(value: Any) -> torch.dtype:
    name = _enum_value(value)
    try:
        return {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[name]
    except KeyError as error:
        raise ValueError(f"unsupported activation dtype: {name}") from error


def _compute_context(
    device_type: str,
    dtype: torch.dtype,
) -> AbstractContextManager[Any]:
    if dtype is torch.float32:
        return nullcontext()
    if device_type not in {"cpu", "cuda"}:
        raise ValueError(f"autocast is not supported for training device type {device_type!r}")
    return torch.autocast(device_type=device_type, dtype=dtype)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _place_unwrapped_model(
    model: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """Apply the Trainer precision contract without flattening custom layouts."""

    parameters = tuple(model.parameters())
    floating_parameters = tuple(
        parameter for parameter in parameters if parameter.is_floating_point()
    )
    parameter_dtypes = {parameter.dtype for parameter in floating_parameters}
    tensor_devices = {parameter.device for parameter in parameters}
    tensor_devices.update(buffer.device for buffer in model.buffers())
    if len(parameter_dtypes) > 1:
        raise ValueError(
            "Trainer automatic placement requires homogeneous floating-point parameter "
            "dtypes; place an intentionally mixed-dtype model before data-parallel setup"
        )
    if len(tensor_devices) > 1:
        raise ValueError(
            "Trainer automatic placement requires model parameters and buffers on one "
            "device; place an intentionally heterogeneous model before data-parallel setup"
        )
    already_placed = (not tensor_devices or tensor_devices == {device}) and (
        not parameter_dtypes or parameter_dtypes == {dtype}
    )
    return model if already_placed else model.to(device=device, dtype=dtype)


def _number_microbatches(
    microbatches: tuple[Any, ...],
    *,
    start_index: int = 0,
) -> tuple[Any, ...]:
    numbered: list[Any] = []
    for index, microbatch in enumerate(microbatches, start=start_index):
        if isinstance(microbatch, dict):
            microbatch = dict(microbatch)
            microbatch["_microbatch_index"] = index
        numbered.append(microbatch)
    return tuple(numbered)


@dataclass
class TrainerState:
    step: int = 0
    consumed_samples: int = 0
    consumed_tokens: int = 0

    def state_dict(self) -> dict[str, int]:
        return {
            "step": self.step,
            "consumed_samples": self.consumed_samples,
            "consumed_tokens": self.consumed_tokens,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.step = int(state.get("step", 0))
        self.consumed_samples = int(state.get("consumed_samples", 0))
        self.consumed_tokens = int(state.get("consumed_tokens", 0))


class Trainer:
    def __init__(
        self,
        *,
        config: Any,
        model: nn.Module,
        parallel: Any,
        data_parallel: Any,
        checkpoint: Any | None = None,
        data_iterator: Iterator[Any] | None = None,
        rng: Any | None = None,
    ) -> None:
        self.config = config
        self.parallel = parallel
        self.data_parallel = data_parallel
        self.checkpoint = checkpoint
        self.data_iterator = data_iterator
        self.rng = rng
        self._data_batches_consumed = 0
        self.batch_router = BatchRouter(parallel)
        self.state = TrainerState()

        self.data_parallel.configure_precision(config.precision)
        if getattr(data_parallel, "_model", None) is None:
            model = _place_unwrapped_model(
                model,
                device=parallel.runtime.device,
                dtype=_torch_dtype(config.precision.params),
            )
            prepare_pipeline_stage(model)
            model = data_parallel.setup(
                model,
                config.optimizer,
                getattr(data_parallel, "parameter_domains", None),
            )
        else:
            model = data_parallel.model
            prepare_pipeline_stage(model)
        self.model = model
        self.schedule = self._build_schedule()

    def _pipeline_size(self) -> int:
        group = self.parallel.pp
        return group.size

    def _build_schedule(self) -> Any:
        communicator = None
        parameter = next(self.model.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        compute_dtype = _torch_dtype(self.config.precision.compute)

        def compute_context() -> AbstractContextManager[Any]:
            return _compute_context(device.type, compute_dtype)

        if self._pipeline_size() > 1:
            sequence = self.config.model.seq_length // self.config.parallel.context
            if self.config.parallel.sequence_parallel:
                sequence //= self.config.parallel.tensor
            dtype_config = self.config.pipeline.activation_dtype or self.config.precision.compute
            communicator = P2PCommunicator(
                self.parallel,
                activation_shape=(
                    self.config.training.micro_batch_size,
                    sequence,
                    self.config.model.hidden_size,
                ),
                activation_dtype=_torch_dtype(dtype_config),
                device=device,
                dynamic_shapes=self.config.pipeline.dynamic_activation_shapes,
            )
        schedule = _enum_value(self.config.pipeline.schedule)
        if schedule == "gpipe":
            return GPipeSchedule(
                self.parallel,
                communicator,
                compute_context=compute_context,
                overlap_p2p=self.config.pipeline.overlap_p2p,
            )
        if schedule == "1f1b":
            return OneForwardOneBackwardSchedule(
                self.parallel,
                communicator,
                compute_context=compute_context,
                overlap_p2p=self.config.pipeline.overlap_p2p,
            )
        if schedule == "interleaved_1f1b":
            if communicator is None:
                raise ValueError("interleaved_1f1b requires pipeline size greater than one")
            layout = getattr(self.model, "layout", None)
            if layout is None:
                raise TypeError(
                    "interleaved_1f1b requires a model built as an explicit GPTPipeline"
                )
            return InterleavedOneForwardOneBackwardSchedule(
                self.parallel,
                layout,
                communicator,
                compute_context=compute_context,
                overlap_p2p=self.config.pipeline.overlap_p2p,
            )
        raise ValueError(f"unknown pipeline schedule: {schedule}")

    def train_step(self, batch: Any) -> StepOutput:
        self.model.train()
        routed_batch = self.batch_router.route(batch if self.batch_router.is_source else None)
        parameter = next(self.model.parameters(), None)
        if parameter is not None:
            routed_batch = _move_to_device(routed_batch, parameter.device)
        actual_sequence_length = self._validate_runtime_sequence(routed_batch)
        microbatches = _number_microbatches(
            split_microbatches(routed_batch, self.config.training.micro_batch_size),
            start_index=(self.state.step * self.config.training.gradient_accumulation_steps),
        )
        expected = self.config.training.gradient_accumulation_steps
        if len(microbatches) != expected:
            raise ValueError(
                "local batch must contain exactly gradient_accumulation_steps microbatches: "
                f"got {len(microbatches)}, expected {expected}"
            )

        self.data_parallel.zero_grad()
        output = self.schedule.forward_backward(
            stage=self.model,
            microbatches=microbatches,
            data_parallel=self.data_parallel,
        )
        self.data_parallel.finalize_gradients()
        max_norm = self.config.optimizer.clip_grad_norm
        if max_norm is not None:
            grad_norm = self.data_parallel.clip_grad_norm(max_norm)
            output.metrics["grad_norm"] = float(grad_norm.detach().cpu())
        self.data_parallel.optimizer_step()

        self.state.step += 1
        self.state.consumed_samples += self.config.global_batch_size(
            world_size=self.parallel.topology.world_size
        )
        self.state.consumed_tokens += (
            self.config.global_batch_size(world_size=self.parallel.topology.world_size)
            * actual_sequence_length
        )
        return output

    def _validate_runtime_sequence(self, batch: Any) -> int:
        input_ids = batch.get("input_ids") if isinstance(batch, dict) else None
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim < 2:
            return int(self.config.model.seq_length)
        local_sequence = int(input_ids.shape[-1])
        global_sequence = local_sequence * int(self.parallel.cp.size)
        if (
            not self.config.pipeline.dynamic_activation_shapes
            and global_sequence != self.config.model.seq_length
        ):
            raise ValueError(
                "runtime sequence length differs from model.seq_length while "
                "pipeline.dynamic_activation_shapes is disabled: "
                f"{global_sequence} != {self.config.model.seq_length}"
            )
        if self.config.parallel.sequence_parallel and local_sequence % int(self.parallel.tp.size):
            raise ValueError(
                "sequence parallelism requires the CP-local runtime sequence length "
                "to be divisible by TP"
            )
        return global_sequence

    @torch.no_grad()
    def evaluate_step(self, batch: Any) -> StepOutput:
        self.model.eval()
        routed_batch = self.batch_router.route(batch if self.batch_router.is_source else None)
        parameter = next(self.model.parameters(), None)
        if parameter is not None:
            routed_batch = _move_to_device(routed_batch, parameter.device)
        self._validate_runtime_sequence(routed_batch)
        microbatches = split_microbatches(routed_batch, self.config.training.micro_batch_size)
        return self.schedule.forward_backward(
            stage=self.model,
            microbatches=microbatches,
            data_parallel=self.data_parallel,
            forward_only=True,
        )

    def fit(
        self,
        data: Iterable[Any] | Iterator[Any] | None = None,
        *,
        max_steps: int | None = None,
    ) -> TrainerState:
        iterator = iter(data) if data is not None else self.data_iterator
        if iterator is None:
            raise ValueError("Trainer.fit requires a data iterator")
        target = self.config.training.max_steps if max_steps is None else max_steps
        try:
            while self.state.step < target:
                if self.batch_router.is_source:
                    batch = next(iterator)
                    self._data_batches_consumed += 1
                else:
                    batch = None
                self.train_step(batch)
                if self.checkpoint is not None and self._should_save():
                    self.save_checkpoint()
        finally:
            if self.checkpoint is not None:
                flush = getattr(self.checkpoint, "flush", None)
                if callable(flush):
                    flush()
        return self.state

    def close(self) -> None:
        if self.checkpoint is None:
            return
        close = getattr(self.checkpoint, "close", None)
        if callable(close):
            close()

    def restore_data_position(self) -> None:
        """Advance a deterministic source iterator to the restored optimizer step.

        The first-phase loader consumes exactly one dataloader batch per optimizer step.  Its
        ``DistributedSampler`` order is deterministic for a fixed config seed, so replaying and
        discarding these batches restores the next sample without adding sampler internals to the
        checkpoint schema.  Non-source TP/PP/CP ranks never own a data iterator.
        """

        target = self.state.step
        if self.state.consumed_samples:
            local_batch_size = (
                self.config.training.micro_batch_size
                * self.config.training.gradient_accumulation_steps
            )
            samples_per_data_batch = self.batch_router.data_replica_count * local_batch_size
            if self.state.consumed_samples % samples_per_data_batch:
                raise RuntimeError(
                    "restored consumed_samples is incompatible with the current DP/EP "
                    "replica count and local batch size"
                )
            target = self.state.consumed_samples // samples_per_data_batch
        if self._data_batches_consumed > target:
            raise RuntimeError(
                "the current data iterator is already past the restored checkpoint position"
            )
        if not self.batch_router.is_source:
            self._data_batches_consumed = target
            return
        if self.data_iterator is None:
            raise ValueError("restoring data position requires a configured data iterator")
        try:
            for _ in range(target - self._data_batches_consumed):
                next(self.data_iterator)
        except StopIteration as error:
            raise RuntimeError(
                "the training data iterator ended before the restored checkpoint step; "
                "use a dataset with enough batches or restore its sampler state explicitly"
            ) from error
        self._data_batches_consumed = target

    def _should_save(self) -> bool:
        interval = self.config.checkpoint.save_interval
        return interval > 0 and self.state.step % interval == 0

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        if self.checkpoint is None:
            raise RuntimeError("no CheckpointManager was configured")
        return self.checkpoint.save(
            self.state.step,
            model=self.model,
            data_parallel=self.data_parallel,
            trainer_state=self.state.state_dict(),
            path=path,
            rng=self.rng,
        )

    def load_checkpoint(self, path: str | Path) -> TrainerState:
        if self.checkpoint is None:
            raise RuntimeError("no CheckpointManager was configured")
        flush = getattr(self.checkpoint, "flush", None)
        if callable(flush):
            flush()
        loaded = self.checkpoint.load(
            path,
            model=self.model,
            data_parallel=self.data_parallel,
            rng=self.rng,
        )
        if isinstance(loaded, TrainerState):
            self.state = loaded
        elif isinstance(loaded, dict):
            self.state.load_state_dict(loaded)
        else:
            self.state.load_state_dict(loaded.state_dict())
        return self.state
