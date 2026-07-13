from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from .offload import ActivationOffloader, OffloadPolicy

if TYPE_CHECKING:
    from nano_megatron.parallel import ParallelContext, ParameterDomainRegistry


class DataParallelStrategy(ABC):
    """The trainer-facing contract shared by DDP and all ZeRO stages."""

    mode: str

    def __init__(
        self,
        *,
        config: object,
        offload: object | None,
        parallel: ParallelContext,
        parameter_domains: ParameterDomainRegistry | None = None,
    ) -> None:
        self.config = config
        self.parallel = parallel
        self.parameter_domains = parameter_domains
        self.offload_policy = OffloadPolicy.from_config(offload)
        self.offload_policy.validate(self.mode)
        self._activation_offloader = ActivationOffloader(self.offload_policy)
        self._model: nn.Module | None = None
        self._sync_this_backward = True
        self.param_dtype: torch.dtype | None = None
        self.compute_dtype: torch.dtype | None = None
        self.grad_reduce_dtype: torch.dtype | None = None

    @staticmethod
    def _torch_dtype(value: Any) -> torch.dtype:
        name = str(getattr(value, "value", value)).lower()
        try:
            return {
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
            }[name]
        except KeyError as error:
            raise ValueError(f"unsupported precision dtype: {name!r}") from error

    def configure_precision(self, precision: object) -> None:
        """Inject cross-cutting precision policy before ``setup`` materializes wrappers."""

        self.param_dtype = self._torch_dtype(precision.params)  # type: ignore[attr-defined]
        self.compute_dtype = self._torch_dtype(precision.compute)  # type: ignore[attr-defined]
        self.grad_reduce_dtype = self._torch_dtype(  # type: ignore[attr-defined]
            precision.grad_reduce
        )

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("data-parallel strategy has not been set up")
        return self._model

    def _resolve_registry(
        self, parameter_domains: ParameterDomainRegistry | None
    ) -> ParameterDomainRegistry:
        registry = parameter_domains or self.parameter_domains
        if registry is None:
            raise ValueError("ParameterDomainRegistry must be passed explicitly")
        self.parameter_domains = registry
        return registry

    @abstractmethod
    def setup(
        self,
        model: nn.Module,
        optimizer_config: object,
        parameter_domains: ParameterDomainRegistry | None = None,
    ) -> nn.Module:
        raise NotImplementedError

    @contextmanager
    def activation_context(self) -> Iterator[None]:
        """Wrap forward so tensors saved by autograd may be moved to CPU."""

        with self._activation_offloader.context():
            yield

    @contextmanager
    def forward_microbatch_context(
        self, *, synchronize_gradients: bool
    ) -> Iterator[None]:
        """Wrap a microbatch forward.

        ``synchronize_gradients`` identifies the graph whose backward is last
        in the schedule.  DDP consumes this during forward when it prepares
        its reducer; the reference ZeRO strategies only need the activation
        offload portion of this context.
        """

        del synchronize_gradients
        with self.activation_context():
            yield

    @contextmanager
    def microbatch_context(self, *, is_last_microbatch: bool) -> Iterator[None]:
        """Wrap backward for one microbatch.

        The name is kept for the public strategy contract and direct callers.
        Pipeline schedules use ``forward_microbatch_context`` separately so a
        DDP ``no_sync`` decision is made before the matching forward.
        """

        previous = self._sync_this_backward
        self._sync_this_backward = is_last_microbatch
        try:
            # Non-reentrant checkpointing may recompute forward operators
            # during backward; keep saved-tensor offload active there too.
            with self.activation_context():
                yield
        finally:
            self._sync_this_backward = previous

    @abstractmethod
    def backward(self, loss: Tensor) -> None:
        raise NotImplementedError

    def finalize_gradients(self) -> None:
        """Finish deferred communication after model-specific gradient hooks."""

        return None

    @abstractmethod
    def clip_grad_norm(self, max_norm: float) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def optimizer_step(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def zero_grad(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        raise NotImplementedError
