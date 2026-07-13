"""Dropout driven by an explicitly injected parallel RNG stream."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any

import torch.nn.functional as F
from torch import Tensor

from nano_megatron.parallel.rng import RNGStream


@contextmanager
def activation_rng_context(
    rng: Any | None,
    tensor: Tensor,
    *,
    layer: int,
    microbatch: int,
    global_token_offset: int,
    op: str,
):
    """Fork a deterministic activation stream for one logical operation."""

    if rng is None or not hasattr(rng, "fork"):
        with nullcontext():
            yield
        return
    devices: tuple[int, ...] = ()
    if tensor.device.type == "cuda":
        index = tensor.device.index
        if index is None:
            import torch

            index = torch.cuda.current_device()
        devices = (index,)
    with rng.fork(
        RNGStream.ACTIVATION,
        components=(layer, microbatch, global_token_offset, op),
        cuda_devices=devices,
        advance=False,
    ):
        yield


def parallel_dropout(
    tensor: Tensor,
    p: float,
    *,
    training: bool,
    rng: Any | None,
    layer: int,
    microbatch: int,
    global_token_offset: int,
    op: str,
) -> Tensor:
    """Apply topology-stable dropout without consuming the default RNG stream."""

    if not training or p == 0.0:
        return tensor
    with activation_rng_context(
        rng,
        tensor,
        layer=layer,
        microbatch=microbatch,
        global_token_offset=global_token_offset,
        op=op,
    ):
        return F.dropout(tensor, p=p, training=True)


__all__ = ["activation_rng_context", "parallel_dropout"]
