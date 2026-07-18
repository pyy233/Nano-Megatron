"""Small training engine that keeps parallel and DP dependencies explicit."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.distributed as dist
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
from .lr_scheduler import LearningRateScheduler
from .metrics import MetricStore, aggregate_loss_metrics
from .microbatches import split_microbatches
from .wandb_logger import WandbLogger


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
    data_fingerprint: str | None = None
    data_epoch: int = 0
    data_shuffle_seed: int | None = None
    data_sample_offset: int = 0
    data_samples_per_epoch: int | None = None
    lr_scheduler: dict[str, Any] | None = None
    wandb_run_id: str | None = None
    metrics_history: dict[str, int | None] | None = None
    best_validation_loss: float | None = None
    best_validation_step: int | None = None
    best_validation_checkpoint: str | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "consumed_samples": self.consumed_samples,
            "consumed_tokens": self.consumed_tokens,
            "data_fingerprint": self.data_fingerprint,
            "data_epoch": self.data_epoch,
            "data_shuffle_seed": self.data_shuffle_seed,
            "data_sample_offset": self.data_sample_offset,
            "data_samples_per_epoch": self.data_samples_per_epoch,
            "lr_scheduler": None if self.lr_scheduler is None else dict(self.lr_scheduler),
            "wandb_run_id": self.wandb_run_id,
            "metrics_history": (
                None if self.metrics_history is None else dict(self.metrics_history)
            ),
            "best_validation_loss": self.best_validation_loss,
            "best_validation_step": self.best_validation_step,
            "best_validation_checkpoint": self.best_validation_checkpoint,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.step = int(state.get("step", 0))
        self.consumed_samples = int(state.get("consumed_samples", 0))
        self.consumed_tokens = int(state.get("consumed_tokens", 0))
        self.data_epoch = int(state.get("data_epoch", 0))
        shuffle_seed = state.get("data_shuffle_seed")
        self.data_shuffle_seed = None if shuffle_seed is None else int(shuffle_seed)
        self.data_sample_offset = int(state.get("data_sample_offset", 0))
        samples_per_epoch = state.get("data_samples_per_epoch")
        self.data_samples_per_epoch = (
            None if samples_per_epoch is None else int(samples_per_epoch)
        )
        for name in (
            "step",
            "consumed_samples",
            "consumed_tokens",
            "data_epoch",
            "data_sample_offset",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"trainer {name} must be non-negative")
        if self.data_shuffle_seed is not None and self.data_shuffle_seed < 0:
            raise ValueError("trainer data_shuffle_seed must be non-negative or null")
        if self.data_samples_per_epoch is not None and self.data_samples_per_epoch < 1:
            raise ValueError("trainer data_samples_per_epoch must be positive or null")
        fingerprint = state.get("data_fingerprint")
        if fingerprint is not None and (not isinstance(fingerprint, str) or not fingerprint):
            raise ValueError("trainer data_fingerprint must be a non-empty string or null")
        self.data_fingerprint = fingerprint
        scheduler = state.get("lr_scheduler")
        if scheduler is not None and not isinstance(scheduler, dict):
            raise ValueError("trainer lr_scheduler state must be a mapping or null")
        self.lr_scheduler = None if scheduler is None else dict(scheduler)
        wandb_run_id = state.get("wandb_run_id")
        if wandb_run_id is not None and (
            not isinstance(wandb_run_id, str) or not wandb_run_id
        ):
            raise ValueError("trainer wandb_run_id must be a non-empty string or null")
        self.wandb_run_id = wandb_run_id
        metrics_history = state.get("metrics_history")
        if metrics_history is not None and not isinstance(metrics_history, dict):
            raise ValueError("trainer metrics_history state must be a mapping or null")
        self.metrics_history = (
            None if metrics_history is None else dict(metrics_history)
        )
        best_loss = state.get("best_validation_loss")
        best_step = state.get("best_validation_step")
        best_checkpoint = state.get("best_validation_checkpoint")
        if best_loss is None and best_step is None:
            if best_checkpoint is not None:
                raise ValueError(
                    "trainer best_validation_checkpoint requires best validation metrics"
                )
            self.best_validation_loss = None
            self.best_validation_step = None
            self.best_validation_checkpoint = None
            return
        if (
            isinstance(best_loss, bool)
            or not isinstance(best_loss, (int, float))
            or not isfinite(float(best_loss))
            or float(best_loss) < 0.0
        ):
            raise ValueError("trainer best_validation_loss must be finite and non-negative")
        if isinstance(best_step, bool) or not isinstance(best_step, int) or best_step < 0:
            raise ValueError("trainer best_validation_step must be a non-negative integer")
        if best_checkpoint is not None and (
            not isinstance(best_checkpoint, str) or not best_checkpoint
        ):
            raise ValueError(
                "trainer best_validation_checkpoint must be a non-empty string or null"
            )
        self.best_validation_loss = float(best_loss)
        self.best_validation_step = best_step
        self.best_validation_checkpoint = best_checkpoint


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
        validation_data: Iterable[Any] | None = None,
        tracker: Any | None = None,
        metrics_path: Path | None = None,
        resume_metrics_path: Path | None = None,
        rng: Any | None = None,
    ) -> None:
        self.config = config
        self.parallel = parallel
        self.data_parallel = data_parallel
        self.checkpoint = checkpoint
        self.data_iterator = data_iterator
        self.validation_data = validation_data
        self.rng = rng
        self._data_batches_consumed = 0
        self._stateful_data: Any | None = None
        self._checkpoint_loaded = False
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
        self.lr_scheduler = LearningRateScheduler(
            config=config.lr_scheduler,
            optimizer_config=config.optimizer,
            data_parallel=data_parallel,
            max_steps=config.training.max_steps,
        )
        self.state.lr_scheduler = self.lr_scheduler.state_dict()
        self.last_train_metrics: dict[str, float] = {}
        self.last_validation_metrics: dict[str, float] = {}
        runtime = getattr(parallel, "runtime", None)
        is_primary = getattr(runtime, "is_primary", None)
        if is_primary is None:
            is_primary = int(getattr(parallel, "rank", 0)) == 0
        self.tracker = tracker or WandbLogger(
            getattr(config, "wandb", None),
            run_config=config,
            is_primary=bool(is_primary),
            metrics_path=metrics_path,
            resume_metrics_path=resume_metrics_path,
        )

    def bind_data_iterator(
        self,
        data: Iterable[Any] | Iterator[Any],
        *,
        data_fingerprint: str | None = None,
    ) -> None:
        """Bind deterministic data and reject silent corpus/order changes on resume."""

        if data_fingerprint is not None and (
            not isinstance(data_fingerprint, str) or not data_fingerprint
        ):
            raise ValueError("data_fingerprint must be a non-empty string or null")
        checkpoint_fingerprint = self.state.data_fingerprint
        if checkpoint_fingerprint is not None and data_fingerprint is None:
            raise RuntimeError(
                "checkpoint contains a training data fingerprint; rebinding requires the "
                "current data_fingerprint"
            )
        if (
            checkpoint_fingerprint is not None
            and data_fingerprint is not None
            and checkpoint_fingerprint != data_fingerprint
        ):
            raise RuntimeError(
                "training data fingerprint does not match the checkpoint: "
                f"{data_fingerprint} != {checkpoint_fingerprint}"
            )
        if data_fingerprint is not None:
            self.state.data_fingerprint = data_fingerprint
        state_dict = getattr(data, "state_dict", None)
        load_state_dict = getattr(data, "load_state_dict", None)
        if callable(state_dict) and callable(load_state_dict):
            if self.state.data_shuffle_seed is None:
                seek = getattr(data, "seek_consumed_samples", None)
                if self.state.consumed_samples and callable(seek):
                    seek(self.state.consumed_samples)
            else:
                loader_state: dict[str, Any] = {
                    "epoch": self.state.data_epoch,
                    "shuffle_seed": self.state.data_shuffle_seed,
                    "epoch_seed": self.state.data_shuffle_seed + self.state.data_epoch,
                    "sample_offset": self.state.data_sample_offset,
                }
                if self.state.data_samples_per_epoch is not None:
                    loader_state["samples_per_epoch"] = self.state.data_samples_per_epoch
                load_state_dict(loader_state)
            self._stateful_data = data
            self._sync_data_state()
        else:
            self._stateful_data = None
        self.data_iterator = iter(data)

    def _sync_data_state(self) -> None:
        if self._stateful_data is None:
            return
        loader_state = self._stateful_data.state_dict()
        self.state.data_epoch = int(loader_state["epoch"])
        self.state.data_shuffle_seed = int(loader_state["shuffle_seed"])
        self.state.data_sample_offset = int(loader_state["sample_offset"])
        self.state.data_samples_per_epoch = int(loader_state["samples_per_epoch"])

    def bind_validation_data(self, data: Iterable[Any]) -> None:
        self.validation_data = data

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
        learning_rate = self.lr_scheduler.learning_rate
        self.data_parallel.optimizer_step()
        output.metrics["learning_rate"] = learning_rate

        self.state.step += 1
        self.lr_scheduler.set_completed_steps(self.state.step)
        self.state.lr_scheduler = self.lr_scheduler.state_dict()
        self.state.consumed_samples += self.config.global_batch_size(
            world_size=self.parallel.topology.world_size
        )
        self.state.consumed_tokens += (
            self.config.global_batch_size(world_size=self.parallel.topology.world_size)
            * actual_sequence_length
        )
        parameter = next(self.model.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        output.metrics = aggregate_loss_metrics(
            output.metrics,
            self.parallel,
            device=device,
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
        output = self.schedule.forward_backward(
            stage=self.model,
            microbatches=microbatches,
            data_parallel=self.data_parallel,
            forward_only=True,
        )
        parameter = next(self.model.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        output.metrics = aggregate_loss_metrics(
            output.metrics,
            self.parallel,
            device=device,
        )
        return output

    def evaluate(
        self,
        data: Iterable[Any] | None = None,
        *,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        source_data = self.validation_data if data is None else data
        if self.batch_router.is_source and source_data is None:
            raise ValueError("validation requires a data iterable on batch-source ranks")
        batches = self.config.validation.batches if max_batches is None else max_batches
        if isinstance(batches, bool) or not isinstance(batches, int) or batches < 1:
            raise ValueError("validation max_batches must be a positive integer")
        iterator = iter(source_data) if self.batch_router.is_source else None
        metrics = MetricStore()
        for _ in range(batches):
            if self.batch_router.is_source:
                assert iterator is not None
                try:
                    batch = next(iterator)
                except StopIteration:
                    raise RuntimeError(
                        "validation data ended before validation.batches; build a loader "
                        "with enough complete global batches"
                    ) from None
            else:
                batch = None
            metrics.update(self.evaluate_step(batch).metrics)
        self.last_validation_metrics = metrics.compute()
        return dict(self.last_validation_metrics)

    def fit(
        self,
        data: Iterable[Any] | Iterator[Any] | None = None,
        *,
        max_steps: int | None = None,
    ) -> TrainerState:
        if data is not None and self.state.data_fingerprint is not None:
            raise RuntimeError(
                "checkpointed training data must be rebound with bind_data_iterator() "
                "so its fingerprint can be verified"
            )
        iterator = iter(data) if data is not None else self.data_iterator
        if iterator is None:
            raise ValueError("Trainer.fit requires a data iterator")
        target = self.config.training.max_steps if max_steps is None else max_steps
        run_id = self.tracker.start(
            checkpoint_run_id=self.state.wandb_run_id,
            checkpoint_step=self.state.step if self._checkpoint_loaded else None,
            checkpoint_metrics_state=(
                self.state.metrics_history if self._checkpoint_loaded else None
            ),
            model=self.model,
        )
        if run_id is not None:
            self.state.wandb_run_id = run_id
        train_metrics = MetricStore()
        interval_started = perf_counter()
        interval_samples = self.state.consumed_samples
        interval_tokens = self.state.consumed_tokens
        initial_step = self.state.step
        last_checkpoint_step: int | None = None
        parameter = next(self.model.parameters(), None)
        if parameter is not None and parameter.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(parameter.device)
        try:
            while self.state.step < target:
                if self.batch_router.is_source:
                    loader_state = (
                        None
                        if self._stateful_data is None
                        else self._stateful_data.state_dict()
                    )
                    batch = next(iterator)
                    self._data_batches_consumed += 1
                else:
                    loader_state = None
                    batch = None
                try:
                    output = self.train_step(batch)
                except BaseException:
                    if loader_state is not None:
                        self._stateful_data.load_state_dict(loader_state)
                        self.data_iterator = iter(self._stateful_data)
                    raise
                self._sync_data_state()
                train_metrics.update(output.metrics)
                logged = self._should_log() or self.state.step == target
                if logged:
                    now = perf_counter()
                    elapsed = max(now - interval_started, 1.0e-12)
                    self.last_train_metrics = train_metrics.compute()
                    self.last_train_metrics.update(
                        {
                            "samples_per_second": (
                                self.state.consumed_samples - interval_samples
                            )
                            / elapsed,
                            "tokens_per_second": (
                                self.state.consumed_tokens - interval_tokens
                            )
                            / elapsed,
                        }
                    )
                    self.last_train_metrics.update(self._device_memory_metrics())
                    self._print_metrics("train", self.last_train_metrics)
                    self.tracker.log("train", self.last_train_metrics, state=self.state)
                    train_metrics.reset()
                    interval_started = now
                    interval_samples = self.state.consumed_samples
                    interval_tokens = self.state.consumed_tokens
                if self._should_validate():
                    validation_metrics = self.evaluate()
                    self._print_metrics("validation", validation_metrics)
                    self.tracker.log("validation", validation_metrics, state=self.state)
                    improved = self._record_best_validation(validation_metrics)
                    if (
                        improved
                        and self._checkpointing_enabled()
                        and bool(
                            getattr(
                                self.config.checkpoint,
                                "save_best_validation",
                                True,
                            )
                        )
                    ):
                        expected = (
                            Path(self.checkpoint.directory)
                            / f"step_{self.state.step:08d}"
                        )
                        self.state.best_validation_checkpoint = str(expected)
                        checkpoint_path = self.save_checkpoint()
                        pin = getattr(self.checkpoint, "pin_best_validation", None)
                        if not callable(pin):
                            raise RuntimeError(
                                "checkpoint manager does not support best-validation pinning"
                            )
                        pinned_path = pin(
                            self.state.step,
                            self.state.best_validation_loss,
                        )
                        if Path(pinned_path) != Path(checkpoint_path):
                            raise RuntimeError(
                                "best-validation pin returned a different checkpoint path"
                            )
                        last_checkpoint_step = self.state.step
                        self.tracker.update_summary(
                            {
                                "checkpoint/best_validation": str(checkpoint_path),
                                "checkpoint/best_validation_loss": (
                                    self.state.best_validation_loss
                                ),
                                "checkpoint/best_validation_step": self.state.step,
                                "checkpoint/latest": str(checkpoint_path),
                                "checkpoint/step": self.state.step,
                            }
                        )
                    if logged:
                        interval_started = perf_counter()
                if (
                    self.checkpoint is not None
                    and self._should_save()
                    and last_checkpoint_step != self.state.step
                ):
                    checkpoint_path = self.save_checkpoint()
                    last_checkpoint_step = self.state.step
                    self.tracker.update_summary(
                        {
                            "checkpoint/latest": str(checkpoint_path),
                            "checkpoint/step": self.state.step,
                        }
                    )
            if (
                self._checkpointing_enabled()
                and bool(getattr(self.config.checkpoint, "save_final", True))
                and self.state.step > initial_step
                and last_checkpoint_step != self.state.step
            ):
                checkpoint_path = self.save_checkpoint()
                last_checkpoint_step = self.state.step
                self.tracker.update_summary(
                    {
                        "checkpoint/final": str(checkpoint_path),
                        "checkpoint/latest": str(checkpoint_path),
                        "checkpoint/step": self.state.step,
                    }
                )
        finally:
            if self.checkpoint is not None:
                flush = getattr(self.checkpoint, "flush", None)
                if callable(flush):
                    flush()
        self._sync_tracker_state()
        return self.state

    def _should_log(self) -> bool:
        interval = int(self.config.training.log_interval)
        return interval > 0 and self.state.step % interval == 0

    def _should_validate(self) -> bool:
        interval = int(self.config.validation.interval)
        return interval > 0 and self.state.step % interval == 0

    def _checkpointing_enabled(self) -> bool:
        return self.checkpoint is not None and int(self.config.checkpoint.save_interval) > 0

    def _record_best_validation(self, metrics: Mapping[str, float]) -> bool:
        loss = metrics.get("loss")
        if (
            isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not isfinite(float(loss))
            or float(loss) < 0.0
        ):
            raise RuntimeError(f"validation returned invalid loss: {loss!r}")
        value = float(loss)
        current = self.state.best_validation_loss
        if current is not None and value >= current:
            return False
        self.state.best_validation_loss = value
        self.state.best_validation_step = self.state.step
        self.state.best_validation_checkpoint = None
        self.tracker.update_summary(
            {
                "validation/best_loss": value,
                "validation/best_step": self.state.step,
            }
        )
        return True

    def _device_memory_metrics(self) -> dict[str, float]:
        parameter = next(self.model.parameters(), None)
        if parameter is None or parameter.device.type != "cuda":
            return {}
        device = parameter.device
        gibibyte = float(1 << 30)
        values = torch.tensor(
            [
                torch.cuda.memory_allocated(device) / gibibyte,
                torch.cuda.max_memory_allocated(device) / gibibyte,
                torch.cuda.memory_reserved(device) / gibibyte,
                torch.cuda.max_memory_reserved(device) / gibibyte,
            ],
            dtype=torch.float64,
            device=device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.MAX)
        names = (
            "gpu_memory_allocated_gib",
            "gpu_peak_memory_allocated_gib",
            "gpu_memory_reserved_gib",
            "gpu_peak_memory_reserved_gib",
        )
        return dict(zip(names, (float(value) for value in values.cpu()), strict=True))

    def _print_metrics(self, prefix: str, metrics: Mapping[str, float]) -> None:
        runtime = getattr(self.parallel, "runtime", None)
        is_primary = getattr(runtime, "is_primary", None)
        if is_primary is None:
            is_primary = int(getattr(self.parallel, "rank", 0)) == 0
        if not bool(is_primary):
            return
        values = " ".join(f"{name}={value:.6g}" for name, value in sorted(metrics.items()))
        print(f"{prefix} step={self.state.step} {values}")

    def close(self) -> None:
        try:
            if self.checkpoint is not None:
                close = getattr(self.checkpoint, "close", None)
                if callable(close):
                    close()
        finally:
            self.tracker.finish()

    def restore_data_position(self) -> None:
        """Restore a legacy plain iterator, or verify a stateful loader is bound.

        New checkpoints load epoch/seed/sample offset directly in
        :meth:`bind_data_iterator`, avoiding replay and worker-prefetch state.
        This replay path remains only for programmatic callers with old plain iterators.
        """

        if getattr(self, "_stateful_data", None) is not None:
            self._sync_data_state()
            return

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
        tracker_flush = getattr(self.tracker, "flush", None)
        if callable(tracker_flush):
            tracker_flush()
        self._sync_tracker_state()
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
        if self.state.lr_scheduler is None:
            self.lr_scheduler.set_completed_steps(self.state.step)
        else:
            self.lr_scheduler.load_state_dict(
                self.state.lr_scheduler,
                expected_completed_steps=self.state.step,
            )
        self.state.lr_scheduler = self.lr_scheduler.state_dict()
        self._checkpoint_loaded = True
        return self.state

    def _sync_tracker_state(self) -> None:
        history_state = getattr(self.tracker, "history_state", None)
        if callable(history_state):
            value = history_state()
            self.state.metrics_history = None if value is None else dict(value)
