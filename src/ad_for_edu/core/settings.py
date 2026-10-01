"""Settings files: YAML with closed key sets and values taken from the environment.

Three rules hold for every settings file:

* **Closed keys.** A file is bound to a dataclass. A key the dataclass does not
  declare is an error that lists the keys that exist, so a misspelled key cannot
  silently fall back to a default.
* **Environment, not defaults.** `${NAME}` in a value is replaced by the environment
  variable `NAME`. An unset variable is an error that names the variable.
* **No absolute paths.** Locations are written relative to a directory that comes
  from the environment, so a settings file never encodes one machine's layout.
"""

from __future__ import annotations

import dataclasses
import os
import re
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

from .errors import SettingsError

S = TypeVar("S")

CONFIG_DIR_VARIABLE = "AD_FOR_EDU_CONFIG_DIR"
DATA_DIR_VARIABLE = "AD_FOR_EDU_DATA_DIR"
WORK_DIR_VARIABLE = "AD_FOR_EDU_WORK_DIR"

_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ABSOLUTE = re.compile(r"^(?:[A-Za-z]:[\\/]|/|~)")


def require_env(name: str) -> str:
    """The value of environment variable `name`; an error naming it when unset or empty."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise SettingsError(
            f"environment variable {name} is not set; it has no default. "
            "See .env.example for the variables a command may need."
        )
    return value


def data_dir() -> Path:
    return Path(require_env(DATA_DIR_VARIABLE))


def work_dir() -> Path:
    return Path(require_env(WORK_DIR_VARIABLE))


def config_dir() -> Path:
    """Where the settings files live.

    `AD_FOR_EDU_CONFIG_DIR` when set; otherwise the `configs` directory of a source
    checkout; otherwise `configs` under the current directory.
    """
    named = os.environ.get(CONFIG_DIR_VARIABLE, "").strip()
    if named:
        return Path(named)
    checkout = Path(__file__).resolve().parents[3] / "configs"
    if checkout.is_dir():
        return checkout
    return Path.cwd() / "configs"


def set_config_dir(path: str | Path) -> Path:
    """Point every later settings read at `path`.

    Set here, in the environment, rather than passed down through every call, so that
    one run cannot read half its settings from one directory and half from another.
    """
    directory = Path(path)
    if not directory.is_dir():
        raise SettingsError(f"{directory}: not a directory, so it holds no settings files")
    os.environ[CONFIG_DIR_VARIABLE] = str(directory)
    return directory


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Parse one YAML file into a mapping. An empty file is an empty mapping."""
    import yaml

    source = Path(path)
    if not source.is_file():
        raise SettingsError(f"settings file not found: {source}")
    parsed = yaml.safe_load(source.read_text(encoding="utf-8"))
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise SettingsError(f"{source}: expected a mapping at the top level")
    return parsed


def reject_absolute_paths(value: Any, *, where: str) -> None:
    """Refuse a raw settings value that is an absolute path."""
    if isinstance(value, str):
        if _ABSOLUTE.match(value.strip()):
            raise SettingsError(
                f"{where}: {value!r} is an absolute path. Write locations relative to a "
                "directory named by an environment variable."
            )
    elif isinstance(value, Mapping):
        for key, item in value.items():
            reject_absolute_paths(item, where=f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_absolute_paths(item, where=f"{where}[{index}]")


def resolve_env(value: Any, *, where: str) -> Any:
    """Replace every `${NAME}` in strings, recursing through mappings and lists."""
    if isinstance(value, str):
        return _VARIABLE.sub(lambda match: require_env(match.group(1)), value)
    if isinstance(value, Mapping):
        return {key: resolve_env(item, where=f"{where}.{key}") for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_env(item, where=f"{where}[{i}]") for i, item in enumerate(value)]
    return value


def check_keys(
    mapping: Mapping[str, Any],
    allowed: typing.Iterable[str],
    *,
    required: typing.Iterable[str] = (),
    where: str,
) -> None:
    """Fail on a key outside `allowed` or on a missing `required` key."""
    allowed = set(allowed)
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise SettingsError(f"{where}: unknown keys {unknown}; known keys: {sorted(allowed)}")
    missing = sorted(set(required) - set(mapping))
    if missing:
        raise SettingsError(f"{where}: missing keys {missing}")


def bind(cls: type[S], mapping: Mapping[str, Any], *, where: str) -> S:
    """Build the settings dataclass `cls` from `mapping`, enforcing its key set.

    A field whose declared type is itself a dataclass is bound recursively. Tuples
    declared on the dataclass accept YAML lists.
    """
    if not dataclasses.is_dataclass(cls):
        raise SettingsError(f"{where}: {cls!r} is not a settings dataclass")
    if not isinstance(mapping, Mapping):
        raise SettingsError(f"{where}: expected a mapping, got {type(mapping).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    required = [
        name
        for name, f in fields.items()
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    ]
    check_keys(mapping, fields, required=required, where=where)
    hints = typing.get_type_hints(cls)
    values: dict[str, Any] = {}
    for name, raw in mapping.items():
        declared = hints.get(name)
        if dataclasses.is_dataclass(declared) and isinstance(raw, Mapping):
            values[name] = bind(declared, raw, where=f"{where}.{name}")
        elif typing.get_origin(declared) is tuple and isinstance(raw, list):
            values[name] = tuple(raw)
        else:
            values[name] = raw
    return cls(**values)


def load_settings(name: str | Path, cls: type[S], *, section: str | None = None) -> S:
    """Load `<config dir>/<name>` (or an explicit path) and bind it to `cls`.

    `section` selects one top-level key of the file before binding.
    """
    path = Path(name)
    if not path.is_absolute() and not path.is_file():
        path = config_dir() / path
    raw = load_yaml(path)
    where = path.name
    if section is not None:
        if section not in raw:
            raise SettingsError(f"{where}: no section {section!r}; sections: {sorted(raw)}")
        raw = raw[section]
        where = f"{where}:{section}"
    reject_absolute_paths(raw, where=where)
    return bind(cls, resolve_env(raw, where=where), where=where)


def merge(*layers: Mapping[str, Any]) -> dict[str, Any]:
    """Overlay mappings left to right; a nested mapping is merged, anything else replaced."""
    merged: dict[str, Any] = {}
    for layer in layers:
        for key, value in layer.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = merge(merged[key], value)
            else:
                merged[key] = value
    return merged
