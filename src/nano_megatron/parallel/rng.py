"""Explicit deterministic RNG streams derived from parallel coordinates."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from enum import StrEnum
from threading import Lock
from typing import TYPE_CHECKING, Any

from .axes import ParallelCoordinate

if TYPE_CHECKING:
    from .context import ParallelContext


class RNGStream(StrEnum):
    DATA = "data"
    DENSE_INIT = "dense_init"
    EXPERT_INIT = "expert_init"
    ACTIVATION = "activation"


class ParallelRNG:
    """Own independent, checkpointable logical RNG streams.

    Seeds are stable across Python processes and intentionally include only
    coordinates which identify distinct data or parameter shards.  In
    particular, dense initialization ignores DP/EP/CP replica coordinates,
    while expert initialization additionally includes EP.
    """

    _MAX_SEED = 2**63 - 1

    def __init__(self, base_seed: int, coordinate: ParallelCoordinate) -> None:
        if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
            raise ValueError("base_seed must be a non-negative integer")
        if not isinstance(coordinate, ParallelCoordinate):
            raise TypeError("coordinate must be a ParallelCoordinate")
        self.base_seed = base_seed
        self.coordinate = coordinate
        self._counters = {stream: 0 for stream in RNGStream}
        self._lock = Lock()

    @classmethod
    def from_context(cls, base_seed: int, parallel: ParallelContext) -> ParallelRNG:
        return cls(base_seed, parallel.coordinate)

    def seed(self, stream: RNGStream | str, *components: int | str) -> int:
        parsed = RNGStream(stream)
        coordinate_components = self._coordinate_components(parsed)
        payload = "|".join(
            [
                str(self.base_seed),
                parsed.value,
                *coordinate_components,
                *(str(component) for component in components),
            ]
        ).encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8, person=b"nano-mgt").digest()
        return int.from_bytes(digest, byteorder="little", signed=False) % self._MAX_SEED

    def next_seed(self, stream: RNGStream | str, *components: int | str) -> int:
        parsed = RNGStream(stream)
        with self._lock:
            counter = self._counters[parsed]
            self._counters[parsed] += 1
        return self.seed(parsed, counter, *components)

    def activation_seed(
        self,
        *,
        layer: int,
        microbatch: int,
        global_token_offset: int = 0,
        op: str = "dropout",
    ) -> int:
        for name, value in (
            ("layer", layer),
            ("microbatch", microbatch),
            ("global_token_offset", global_token_offset),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        return self.seed(
            RNGStream.ACTIVATION,
            layer,
            microbatch,
            global_token_offset,
            op,
        )

    def generator(
        self,
        stream: RNGStream | str,
        *,
        device: str = "cpu",
        components: Sequence[int | str] = (),
        advance: bool = True,
    ) -> Any:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("creating a torch.Generator requires PyTorch") from error
        seed = self.next_seed(stream, *components) if advance else self.seed(stream, *components)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator

    @contextmanager
    def fork(
        self,
        stream: RNGStream | str,
        *,
        components: Sequence[int | str] = (),
        cuda_devices: Sequence[int] | None = None,
        advance: bool = True,
    ) -> Iterator[int]:
        """Temporarily seed PyTorch's CPU/CUDA default generators and restore them."""

        try:
            import torch
        except ImportError as error:
            raise RuntimeError("forking a PyTorch RNG stream requires PyTorch") from error
        seed = self.next_seed(stream, *components) if advance else self.seed(stream, *components)
        devices = list(cuda_devices or ())
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.random.default_generator.manual_seed(seed)
            for device in devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            yield seed

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            counters = {stream.value: value for stream, value in self._counters.items()}
        return {
            "base_seed": self.base_seed,
            "coordinate": {
                "tp": self.coordinate.tp,
                "cp": self.coordinate.cp,
                "ep": self.coordinate.ep,
                "dp": self.coordinate.dp,
                "pp": self.coordinate.pp,
            },
            "counters": counters,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("base_seed") != self.base_seed:
            raise ValueError(
                f"RNG base seed mismatch: state has {state.get('base_seed')}, "
                f"expected {self.base_seed}"
            )
        expected_coordinate = self.state_dict()["coordinate"]
        if state.get("coordinate") != expected_coordinate:
            raise ValueError(
                f"RNG coordinate mismatch: state has {state.get('coordinate')}, "
                f"expected {expected_coordinate}"
            )
        counters = state.get("counters")
        if not isinstance(counters, Mapping):
            raise ValueError("RNG state must contain a counters mapping")
        parsed: dict[RNGStream, int] = {}
        for stream in RNGStream:
            value = counters.get(stream.value)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid counter for RNG stream {stream.value!r}: {value!r}")
            parsed[stream] = value
        with self._lock:
            self._counters.update(parsed)

    def _coordinate_components(self, stream: RNGStream) -> tuple[str, ...]:
        coordinate = self.coordinate
        if stream is RNGStream.DATA:
            return (f"dp={coordinate.dp}", f"ep={coordinate.ep}")
        if stream is RNGStream.DENSE_INIT:
            return (f"tp={coordinate.tp}", f"pp={coordinate.pp}")
        if stream is RNGStream.EXPERT_INIT:
            return (
                f"tp={coordinate.tp}",
                f"pp={coordinate.pp}",
                f"ep={coordinate.ep}",
            )
        return (f"dp={coordinate.dp}", f"ep={coordinate.ep}")
