"""The files the pipeline reads, by what they are.

The settings file names every file once, under the dataset or the directory it
belongs to. Code asks for `reference_rows` or `rated_pairs` and never spells a
file name or a dataset name.

A value such as `${AD_FOR_EDU_REFERENCES_DATASET}` is resolved when the file is
asked for, not when the settings are loaded, so a command needs only the
variables of the data it actually touches.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import SettingsError
from ..core.settings import (
    check_keys,
    config_dir,
    data_dir,
    load_yaml,
    reject_absolute_paths,
    resolve_env,
    work_dir,
)
from ..core.strategy import StrategySpec
from .sources import DATASET_SOURCES, DatasetSource

SETTINGS_FILE = "data.yaml"


@dataclass(frozen=True)
class DatasetEntry:
    """One dataset: how it is named, and its files by role."""

    name: str
    identifier: str
    files: Mapping[str, str] = field(default_factory=dict)


class DataCatalog:
    """Resolves the role of a file to a local path."""

    def __init__(
        self,
        datasets: Mapping[str, DatasetEntry],
        local: Mapping[str, str],
        outputs: Mapping[str, str],
        source: DatasetSource,
    ) -> None:
        self.datasets = dict(datasets)
        self.local = dict(local)
        self.outputs = dict(outputs)
        self.source = source

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        source: DatasetSource | None = None,
    ) -> DataCatalog:
        settings = Path(path) if path else config_dir() / SETTINGS_FILE
        raw = load_yaml(settings)
        check_keys(
            raw,
            ("source", "datasets", "local", "outputs"),
            required=("datasets", "local"),
            where=settings.name,
        )
        reject_absolute_paths(raw, where=settings.name)
        datasets = {}
        for name, entry in (raw["datasets"] or {}).items():
            where = f"{settings.name}:datasets.{name}"
            check_keys(entry, ("id", "files"), required=("id", "files"), where=where)
            datasets[name] = DatasetEntry(name, str(entry["id"]), dict(entry["files"] or {}))
        chosen = source or DATASET_SOURCES.create_from(
            StrategySpec.parse(raw.get("source") or "hub")
        )
        return cls(datasets, raw["local"] or {}, raw.get("outputs") or {}, chosen)

    # ------------------------------------------------------------ dataset files
    def _find(self, role: str) -> tuple[DatasetEntry, str]:
        owners = [entry for entry in self.datasets.values() if role in entry.files]
        if not owners:
            known = sorted(role for entry in self.datasets.values() for role in entry.files)
            raise SettingsError(f"no dataset file is registered as {role!r}; known: {known}")
        if len(owners) > 1:
            raise SettingsError(
                f"{role!r} is registered in more than one dataset: "
                f"{sorted(entry.name for entry in owners)}"
            )
        return owners[0], owners[0].files[role]

    def dataset_file(self, role: str, **values: Any) -> Path:
        """The local path of the dataset file registered as `role`.

        `values` fill placeholders of the file name, such as the annotator of a label file.
        """
        entry, filename = self._find(role)
        try:
            filename = filename.format(**values)
        except KeyError as error:
            raise SettingsError(
                f"the file name of {role!r} needs a value for {error.args[0]!r}"
            ) from None
        identifier = resolve_env(entry.identifier, where=f"datasets.{entry.name}.id")
        return self.source.fetch(identifier, filename)

    def roles(self) -> tuple[str, ...]:
        return tuple(sorted(role for entry in self.datasets.values() for role in entry.files))

    # ------------------------------------------------------------ local material
    def local_path(self, role: str) -> Path:
        """A location under the data directory, such as the keyframes of all lectures."""
        if role not in self.local:
            raise SettingsError(f"no local location {role!r}; known: {sorted(self.local)}")
        return data_dir() / self.local[role]

    def output_path(self, role: str) -> Path:
        """A location under the work directory, where a stage writes."""
        if role not in self.outputs:
            raise SettingsError(f"no output location {role!r}; known: {sorted(self.outputs)}")
        return work_dir() / self.outputs[role]
