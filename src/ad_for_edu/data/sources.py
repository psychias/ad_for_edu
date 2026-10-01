"""Where dataset files come from.

A source turns a dataset name and a file name into a local path. Two sources
exist: the dataset hub, and a directory that mirrors the layout of a dataset.
The name of a dataset is never written in this package. It is supplied by the
environment and passed in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..core.errors import MissingSourceError, SettingsError
from ..core.registry import Registry
from ..core.secrets import HUB, credential, has_credential


class DatasetSource(ABC):
    """Fetches files of a dataset."""

    @abstractmethod
    def fetch(self, dataset: str, filename: str) -> Path:
        """The local path of `filename` of `dataset`. Raises when the file does not exist."""

    @abstractmethod
    def fetch_all(self, dataset: str, target: Path) -> Path:
        """Place every file of `dataset` under `target`, keeping its layout."""


DATASET_SOURCES: Registry[DatasetSource] = Registry("dataset source", DatasetSource)


def _safe_relative(filename: str) -> Path:
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise SettingsError(f"a dataset file name must stay inside the dataset: {filename!r}")
    return relative


@DATASET_SOURCES.register("hub")
class HubSource(DatasetSource):
    """Downloads from the dataset hub into its local cache."""

    def __init__(self, revision: str | None = None, cache_dir: str | None = None) -> None:
        self.revision = revision
        self.cache_dir = cache_dir

    def fetch(self, dataset: str, filename: str) -> Path:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

        relative = _safe_relative(filename)
        try:
            path = hf_hub_download(
                repo_id=dataset,
                filename=relative.as_posix(),
                repo_type="dataset",
                revision=self.revision,
                cache_dir=self.cache_dir,
                token=credential(HUB) if has_credential(HUB) else None,
            )
        except RepositoryNotFoundError as error:
            raise MissingSourceError(
                f"dataset {dataset!r} was not found, or the token has no access to it"
            ) from error
        except EntryNotFoundError as error:
            raise MissingSourceError(f"dataset {dataset!r} has no file {filename!r}") from error
        return Path(path)

    def fetch_all(self, dataset: str, target: Path) -> Path:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import RepositoryNotFoundError

        try:
            snapshot_download(
                repo_id=dataset,
                repo_type="dataset",
                revision=self.revision,
                local_dir=str(target),
                token=credential(HUB) if has_credential(HUB) else None,
            )
        except RepositoryNotFoundError as error:
            raise MissingSourceError(
                f"dataset {dataset!r} was not found, or the token has no access to it"
            ) from error
        return Path(target)


@DATASET_SOURCES.register("local_dir")
class LocalDirSource(DatasetSource):
    """Reads from a directory laid out like the dataset. The dataset name is the directory."""

    def fetch(self, dataset: str, filename: str) -> Path:
        root = Path(dataset)
        if not root.is_dir():
            raise MissingSourceError(f"dataset directory not found: {root}")
        path = root / _safe_relative(filename)
        if not path.is_file():
            raise MissingSourceError(f"{root} has no file {filename!r}")
        return path

    def fetch_all(self, dataset: str, target: Path) -> Path:
        """A local directory is used where it is; nothing is copied."""
        root = Path(dataset)
        if not root.is_dir():
            raise MissingSourceError(f"dataset directory not found: {root}")
        return root
