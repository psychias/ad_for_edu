"""Outputs are written once.

A later stage reads what an earlier stage wrote. If a re-run could replace that
file, every analysis built on it would change without notice, and the comparison
between the two runs would be lost. So a stage refuses to write over an existing
output; a re-run names a new file. Stages that resume work append to their own
file instead, through `completed_keys`.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from pathlib import Path
from typing import Any

from .errors import OutputExistsError
from .io import iter_jsonl


def guard_new(path: str | Path, *, overwrite: bool = False) -> Path:
    """Return `path` ready for writing, refusing when it already holds data."""
    target = Path(path)
    if target.exists() and not overwrite:
        if target.is_dir() and not any(target.iterdir()):
            return target
        if target.is_file() and target.stat().st_size == 0:
            return target
        raise OutputExistsError(
            f"{target} already exists. Choose another output name; "
            "an existing output is never replaced."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def is_error_row(row: Mapping[str, Any]) -> bool:
    """A row recording a failed call. It carries no result and must not be counted as one."""
    return bool(row.get("error"))


def completed_keys(
    path: str | Path,
    key: Callable[[Mapping[str, Any]], Hashable],
    *,
    is_complete: Callable[[Mapping[str, Any]], bool] | None = None,
) -> set[Hashable]:
    """Keys of the rows in `path` that need no further work.

    A row that records an error, or that `is_complete` rejects, is left out, so a
    resumed run asks again for exactly the units that have no usable result.
    """
    target = Path(path)
    if not target.is_file():
        return set()
    done: set[Hashable] = set()
    for row in iter_jsonl(target):
        if is_error_row(row):
            continue
        if is_complete is not None and not is_complete(row):
            continue
        done.add(key(row))
    return done
