"""Reading and writing the row files every stage exchanges."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .errors import ContractError, MissingSourceError


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Every non-blank line of a JSON-lines file, parsed."""
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise MissingSourceError(f"row file not found: {source}")
    with source.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"{source}:{number}: not valid JSON ({exc.msg})") from exc


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Write rows as JSON lines, creating parent directories. Returns the row count."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    """Add one row to the end of a JSON-lines file, creating it when absent."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> Any:
    source = Path(path)
    if not source.is_file():
        raise MissingSourceError(f"file not found: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=1, ensure_ascii=False), encoding="utf-8")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.is_file():
        raise MissingSourceError(f"file not found: {source}")
    with source.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def index_by(rows: Iterable[Mapping[str, Any]], key: str) -> dict[Any, Mapping[str, Any]]:
    """Rows keyed by one field. A repeated key is an error, never a silent overwrite."""
    indexed: dict[Any, Mapping[str, Any]] = {}
    for row in rows:
        if key not in row:
            raise ContractError(f"row lacks {key!r}: {sorted(row)[:8]}")
        if row[key] in indexed:
            raise ContractError(f"{key} {row[key]!r} appears more than once")
        indexed[row[key]] = row
    return indexed
