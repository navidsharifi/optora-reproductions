"""Loading experiment settings from TOML files into frozen dataclasses."""

from __future__ import annotations

import dataclasses
import tomllib
import typing
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

ConfigT = TypeVar("ConfigT", bound="DataclassInstance")


def load_config(path: str | Path, config_type: type[ConfigT]) -> ConfigT:
    """Read a flat TOML file into a dataclass.

    Every key of the file must name a field of `config_type`, and every field
    without a default must appear in the file. Values are checked against the
    field annotations. Supported annotations are `bool`, `int`, `float`,
    `str` and variadic tuples `tuple[X, ...]` of those. A TOML integer is
    accepted for a `float` field and a TOML array becomes a tuple, so the
    result can be a frozen dataclass.

    Args:
        path: Path of the TOML file.
        config_type: Dataclass describing the settings, usually frozen.

    Returns:
        An instance of `config_type` holding the file's values.

    Raises:
        TypeError: If `config_type` is not a dataclass or one of its fields
            has an unsupported annotation.
        ValueError: If the file has an unknown key, misses a required field,
            or holds a value of the wrong type.
    """
    if not (isinstance(config_type, type) and dataclasses.is_dataclass(config_type)):
        raise TypeError(f"config_type must be a dataclass type, got {config_type!r}")
    with Path(path).open("rb") as file:
        data = tomllib.load(file)

    fields = {field.name: field for field in dataclasses.fields(config_type)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}")
    missing = sorted(
        name
        for name, field in fields.items()
        if name not in data
        and field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
    )
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")

    hints = typing.get_type_hints(config_type)
    values = {name: _coerce(name, value, hints[name]) for name, value in data.items()}
    return config_type(**values)


def _coerce(name: str, value: Any, annotation: Any) -> Any:
    """Check one TOML value against a field annotation and convert it.

    Args:
        name: Field name, used in error messages.
        value: Value parsed from TOML.
        annotation: Resolved annotation of the field.

    Returns:
        The value, with integers widened to `float` and arrays turned into
        tuples where the annotation asks for it.

    Raises:
        TypeError: If the annotation is not supported.
        ValueError: If the value does not match the annotation.
    """
    if typing.get_origin(annotation) is tuple:
        args = typing.get_args(annotation)
        if len(args) != 2 or args[1] is not Ellipsis:
            raise TypeError(
                f"{name}: only tuple[X, ...] is supported, got {annotation}"
            )
        if not isinstance(value, list):
            raise ValueError(f"{name}: expected an array, got {value!r}")
        return tuple(
            _coerce(f"{name}[{index}]", item, args[0])
            for index, item in enumerate(value)
        )
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{name}: expected a number, got {value!r}")
        return float(value)
    if annotation in (bool, int, str):
        if type(value) is not annotation:
            raise ValueError(f"{name}: expected {annotation.__name__}, got {value!r}")
        return value
    raise TypeError(f"{name}: unsupported annotation {annotation}")
