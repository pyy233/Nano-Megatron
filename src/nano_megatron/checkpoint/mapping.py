from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from torch import Tensor, nn

from nano_megatron.parallel import ParallelAxis, ParameterDomain


@dataclass(frozen=True)
class ShardMetadata:
    logical_key: str
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    global_offset: tuple[int, ...]
    sharded_axes: tuple[ParallelAxis, ...]
    parameter_domain: ParameterDomain
    replica_coordinate: tuple[int, ...]

    def __post_init__(self) -> None:
        ndim = len(self.global_shape)
        if len(self.local_shape) != ndim or len(self.global_offset) != ndim:
            raise ValueError("global_shape, local_shape, and global_offset must have equal rank")
        if any(size < 0 for size in (*self.global_shape, *self.local_shape)):
            raise ValueError("tensor shapes cannot contain negative values")
        for offset, local, global_ in zip(
            self.global_offset, self.local_shape, self.global_shape, strict=True
        ):
            if offset < 0 or offset + local > global_:
                raise ValueError(
                    f"local shard [{offset}, {offset + local}) exceeds global size {global_}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_key": self.logical_key,
            "global_shape": list(self.global_shape),
            "local_shape": list(self.local_shape),
            "global_offset": list(self.global_offset),
            "sharded_axes": [axis.value for axis in self.sharded_axes],
            "parameter_domain": self.parameter_domain.value,
            "replica_coordinate": list(self.replica_coordinate),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ShardMetadata:
        return cls(
            logical_key=str(value["logical_key"]),
            global_shape=tuple(int(item) for item in value["global_shape"]),
            local_shape=tuple(int(item) for item in value["local_shape"]),
            global_offset=tuple(int(item) for item in value["global_offset"]),
            sharded_axes=tuple(ParallelAxis(item) for item in value["sharded_axes"]),
            parameter_domain=ParameterDomain(value["parameter_domain"]),
            replica_coordinate=tuple(int(item) for item in value["replica_coordinate"]),
        )


@dataclass
class ShardedState(Mapping[str, Any]):
    """Logical tensor state kept separate from its physical checkpoint backend."""

    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, ShardMetadata] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.state[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.state)

    def __len__(self) -> int:
        return len(self.state)

    def validate(self) -> None:
        tensor_keys = {key for key, _ in iter_tensors(self.state)}
        unknown = set(self.metadata) - tensor_keys
        if unknown:
            raise ValueError(f"metadata refers to missing tensor keys: {sorted(unknown)}")
        for key, metadata in self.metadata.items():
            tensor = dict(iter_tensors(self.state))[key]
            local = _local_tensor(tensor)
            if tuple(local.shape) != metadata.local_shape:
                raise ValueError(
                    f"metadata local shape for {key!r} is {metadata.local_shape}, "
                    f"tensor shape is {tuple(local.shape)}"
                )

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        parameter_domains: object,
        parallel: object,
    ) -> ShardedState:
        state = dict(model.state_dict())
        try:
            named_parameters = dict(model.named_parameters(remove_duplicate=False))
        except TypeError:
            named_parameters = dict(model.named_parameters())
        metadata: dict[str, ShardMetadata] = {}
        for key, value in state.items():
            if not isinstance(value, Tensor):
                continue
            parameter = named_parameters.get(key)
            if parameter is None and key.startswith("module."):
                parameter = named_parameters.get(key.removeprefix("module."))
            domain = ParameterDomain.DENSE
            tensor_sharded = False
            tensor_shard_dim: int | None = None
            if parameter is not None:
                placement = parameter_domains.placement(parameter)
                domain = ParameterDomain(getattr(placement.domain, "value", placement.domain))
                tensor_sharded = bool(placement.tensor_sharded)
                tensor_shard_dim = getattr(placement, "tensor_shard_dim", None)
            metadata[key] = infer_shard_metadata(
                key,
                value,
                domain=domain,
                parallel=parallel,
                tensor_sharded=tensor_sharded,
                tensor_shard_dim=tensor_shard_dim,
            )
        result = cls(state=state, metadata=metadata)
        result.validate()
        return result


def iter_tensors(value: Any, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
    if isinstance(value, Tensor):
        yield prefix, value
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_tensors(child, child_prefix)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            yield from iter_tensors(child, child_prefix)


def _local_tensor(tensor: Tensor) -> Tensor:
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        return to_local()
    return tensor


def _replica_coordinate(parallel: object, domain: ParameterDomain) -> tuple[int, ...]:
    coordinate = getattr(parallel, "coordinate", None)
    if coordinate is None:
        return ()
    if domain is ParameterDomain.DENSE:
        return (
            int(getattr(coordinate, "dp", 0)),
            int(getattr(coordinate, "ep", 0)),
            int(getattr(coordinate, "cp", 0)),
        )
    return (
        int(getattr(coordinate, "dp", 0)),
        int(getattr(coordinate, "cp", 0)),
    )


def infer_shard_metadata(
    logical_key: str,
    tensor: Tensor,
    *,
    domain: ParameterDomain,
    parallel: object,
    tensor_sharded: bool = False,
    tensor_shard_dim: int | None = None,
) -> ShardMetadata:
    local = _local_tensor(tensor)
    global_shape = [int(size) for size in tensor.shape]
    local_shape = tuple(int(size) for size in local.shape)
    offsets = [0] * len(global_shape)
    axes: list[ParallelAxis] = []

    if tensor_shard_dim is not None:
        if tensor_shard_dim >= len(local_shape):
            raise ValueError(
                f"tensor shard dim {tensor_shard_dim} is outside {logical_key!r} "
                f"with local shape {local_shape}"
            )
        tp_group = parallel.tp
        global_shape[tensor_shard_dim] = local_shape[tensor_shard_dim] * int(tp_group.size)
        offsets[tensor_shard_dim] = local_shape[tensor_shard_dim] * int(tp_group.rank)
        axes.append(ParallelAxis.TP)

    placements = getattr(tensor, "placements", ())
    mesh = getattr(tensor, "device_mesh", None)
    mesh_names = tuple(getattr(mesh, "mesh_dim_names", ()) or ())
    mesh_coordinate = getattr(mesh, "get_coordinate", lambda: None)()
    for mesh_dimension, placement in enumerate(placements):
        shard_dimension = getattr(placement, "dim", None)
        if shard_dimension is None or mesh_coordinate is None:
            continue
        offsets[int(shard_dimension)] = int(mesh_coordinate[mesh_dimension]) * local_shape[
            int(shard_dimension)
        ]
        if mesh_dimension < len(mesh_names):
            with suppress(ValueError):
                axes.append(ParallelAxis(str(mesh_names[mesh_dimension])))
    if tensor_sharded and ParallelAxis.TP not in axes:
        axes.append(ParallelAxis.TP)

    return ShardMetadata(
        logical_key=logical_key,
        global_shape=tuple(global_shape),
        local_shape=local_shape,
        global_offset=tuple(offsets),
        sharded_axes=tuple(axes),
        parameter_domain=domain,
        replica_coordinate=_replica_coordinate(parallel, domain),
    )
