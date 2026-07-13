"""Strict YAML/dictionary loading for configuration dataclasses."""

from __future__ import annotations

import ast
import json
import os
import types
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from .schema import TrainConfig
from .validation import ValidationResult, validate_config


class ConfigLoadError(ValueError):
    """Raised for malformed files, unknown keys, or uncoercible values."""


class ConfigWarning(UserWarning):
    """A valid but surprising or inefficient configuration combination."""


DataclassT = TypeVar("DataclassT")


def config_from_dict(data: Mapping[str, Any]) -> TrainConfig:
    if not isinstance(data, Mapping):
        raise ConfigLoadError(
            f"top-level configuration must be a mapping, got {type(data).__name__}"
        )
    try:
        return _construct_dataclass(TrainConfig, data, path="config")
    except ConfigLoadError:
        raise
    except (TypeError, ValueError) as error:
        raise ConfigLoadError(str(error)) from error


def load_config(
    path: str | Path,
    *,
    overrides: Sequence[str] = (),
    world_size: int | None = None,
    emit_warnings: bool = True,
) -> TrainConfig:
    """Load YAML, apply dotted ``key=value`` overrides, and validate.

    PyYAML is imported lazily so pure topology/config tests can import the
    package without optional runtime dependencies installed.
    """

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigLoadError(f"could not read configuration {source}: {error}") from error

    raw = _load_yaml_mapping(text, source)
    mutable = _deep_copy_mapping(raw)
    for override in overrides:
        _apply_override(mutable, override)
    config = config_from_dict(mutable)

    effective_world_size = world_size
    if effective_world_size is None and "WORLD_SIZE" in os.environ:
        try:
            effective_world_size = int(os.environ["WORLD_SIZE"])
        except ValueError as error:
            raise ConfigLoadError("WORLD_SIZE must be an integer") from error
    result = validate_config(config, world_size=effective_world_size)
    if emit_warnings:
        for message in result.warnings:
            warnings.warn(message, ConfigWarning, stacklevel=2)
    return config


def load_config_with_result(
    path: str | Path,
    *,
    overrides: Sequence[str] = (),
    world_size: int | None = None,
) -> tuple[TrainConfig, ValidationResult]:
    config = load_config(
        path,
        overrides=overrides,
        world_size=world_size,
        emit_warnings=False,
    )
    effective_world_size = world_size
    if effective_world_size is None and "WORLD_SIZE" in os.environ:
        effective_world_size = int(os.environ["WORLD_SIZE"])
    return config, validate_config(config, world_size=effective_world_size)


def _load_yaml_mapping(text: str, source: Path) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        # JSON is a strict YAML subset and provides a useful dependency-free
        # path for generated configs and unit tests.
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as error:
            raise ConfigLoadError(
                "loading YAML requires PyYAML; install the project dependencies "
                f"or provide JSON-compatible YAML ({source})"
            ) from error
    else:
        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise ConfigLoadError(f"invalid YAML in {source}: {error}") from error

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, Mapping):
        raise ConfigLoadError(f"top-level configuration in {source} must be a mapping")
    return loaded


def _construct_dataclass(
    cls: type[DataclassT],
    values: Mapping[str, Any],
    *,
    path: str,
) -> DataclassT:
    if not isinstance(values, Mapping):
        raise ConfigLoadError(f"{path} must be a mapping, got {type(values).__name__}")
    dataclass_fields = {field.name: field for field in fields(cls)}
    unknown = sorted(set(values).difference(dataclass_fields))
    if unknown:
        raise ConfigLoadError(f"{path} contains unsupported field(s): {', '.join(unknown)}")

    type_hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        annotation = type_hints[name]
        kwargs[name] = _coerce(value, annotation, path=f"{path}.{name}")
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as error:
        raise ConfigLoadError(f"{path}: {error}") from error


def _coerce(value: Any, annotation: Any, *, path: str) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        candidates = tuple(candidate for candidate in args if candidate is not type(None))
        errors: list[str] = []
        for candidate in candidates:
            try:
                return _coerce(value, candidate, path=path)
            except ConfigLoadError as error:
                errors.append(str(error))
        raise ConfigLoadError(f"{path} does not match any supported type: {'; '.join(errors)}")

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigLoadError(f"{path} must be a sequence")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(item, args[0], path=f"{path}[]") for item in value)
        if len(value) != len(args):
            raise ConfigLoadError(f"{path} must contain exactly {len(args)} values")
        return tuple(
            _coerce(item, item_type, path=f"{path}[{index}]")
            for index, (item, item_type) in enumerate(zip(value, args, strict=True))
        )

    if origin is list:
        if not isinstance(value, (list, tuple)):
            raise ConfigLoadError(f"{path} must be a sequence")
        item_type = args[0] if args else Any
        return [_coerce(item, item_type, path=f"{path}[]") for item in value]

    if isinstance(annotation, type) and is_dataclass(annotation):
        return _construct_dataclass(annotation, value, path=path)

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if isinstance(value, annotation):
            return value
        try:
            return annotation(value)
        except (TypeError, ValueError) as error:
            choices = ", ".join(str(member.value) for member in annotation)
            raise ConfigLoadError(f"{path} must be one of: {choices}; got {value!r}") from error

    if annotation is Path:
        if not isinstance(value, (str, Path)):
            raise ConfigLoadError(f"{path} must be a filesystem path string")
        return Path(value)
    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigLoadError(f"{path} must be a boolean, got {value!r}")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigLoadError(f"{path} must be an integer, got {value!r}")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigLoadError(f"{path} must be a number, got {value!r}")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigLoadError(f"{path} must be a string, got {value!r}")
        return value
    if annotation is Any:
        return value
    if isinstance(value, annotation):
        return value
    raise ConfigLoadError(f"{path} has unsupported schema type {annotation!r}")


def _apply_override(config: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ConfigLoadError(f"override must use dotted.path=value syntax, got {override!r}")
    dotted_path, raw_value = override.split("=", 1)
    keys = [key.strip() for key in dotted_path.split(".")]
    if not keys or any(not key for key in keys):
        raise ConfigLoadError(f"invalid override path {dotted_path!r}")

    cursor: dict[str, Any] = config
    for key in keys[:-1]:
        existing = cursor.get(key)
        if existing is None:
            existing = {}
            cursor[key] = existing
        if not isinstance(existing, dict):
            raise ConfigLoadError(
                f"override path {dotted_path!r} crosses non-mapping field {key!r}"
            )
        cursor = existing
    cursor[keys[-1]] = _parse_override_value(raw_value)


def _parse_override_value(value: str) -> Any:
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return stripped


def _deep_copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ConfigLoadError(f"configuration keys must be strings, got {key!r}")
        if isinstance(item, Mapping):
            result[key] = _deep_copy_mapping(item)
        elif isinstance(item, list):
            result[key] = list(item)
        else:
            result[key] = item
    return result
