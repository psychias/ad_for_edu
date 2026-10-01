"""Where a training run lives, and the record it writes about itself.

A run is identified by four values: backbone, input arm, training method, seed.

    <work dir>/runs/<backbone>-<arm>/<method>/seed<n>/
        adapter files, checkpoint-*/, metrics.jsonl, predictions.jsonl, run_record.json

The seed is a directory level so that the seeds of one configuration never write
into one another. `run_record.json` holds the facts later stages read instead of
being told: which method trained the adapter, from which backbone revision, under
which settings.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ContractError, MissingSourceError

RECORD_NAME = "run_record.json"
STATUSES = ("running", "done", "failed")
_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"


@dataclass(frozen=True)
class RunKey:
    """The four values that identify a run."""

    backbone: str
    arm: str
    method: str
    seed: int

    def __post_init__(self) -> None:
        named = (
            ("backbone", self.backbone),
            ("arm", self.arm),
            ("method", self.method),
        )
        for label, value in named:
            if not value or "/" in value or chr(92) in value:
                raise ContractError(f"{label} must be a plain name, got {value!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ContractError(f"seed must be a non-negative integer, got {self.seed!r}")

    @property
    def row(self) -> str:
        """The name of the configuration, shared by its methods and seeds."""
        return f"{self.backbone}-{self.arm}"

    @property
    def name(self) -> str:
        return f"{self.row}-{self.method}-seed{self.seed}"

    def directory(self, work_dir: str | Path, *, smoke: bool = False) -> Path:
        row = self.row + ("-smoke" if smoke else "")
        return Path(work_dir) / "runs" / row / self.method / f"seed{self.seed}"


def settings_digest(settings: Mapping[str, Any]) -> str:
    """A stable hash of the resolved settings a run was started with."""
    canonical = json.dumps(settings, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def package_versions() -> dict[str, str | None]:
    """Versions of the libraries that decide how a run behaves; None when absent."""
    versions: dict[str, str | None] = {}
    for name in ("torch", "transformers", "peft", "trl", "accelerate", "datasets"):
        try:
            module = __import__(name)
        except ImportError:
            versions[name] = None
        else:
            versions[name] = getattr(module, "__version__", None)
    return versions


def write_record(
    directory: str | Path,
    key: RunKey,
    settings: Mapping[str, Any],
    *,
    status: str = "running",
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write the run record. Call once at the start and once at the end."""
    if status not in STATUSES:
        raise ContractError(f"status must be one of {STATUSES}, got {status!r}")
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / RECORD_NAME
    earlier: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    now = time.strftime(_TIME_FORMAT)
    started = earlier.get("started_at") or now
    record: dict[str, Any] = {
        "backbone": key.backbone,
        "arm": key.arm,
        "method": key.method,
        "seed": key.seed,
        "model_id": settings.get("model_id"),
        "revision": settings.get("revision"),
        "settings_sha256": settings_digest(settings),
        "status": status,
        "started_at": started,
        "ended_at": None if status == "running" else now,
        "wall_seconds": None,
        "python": sys.version.split()[0],
        "packages": package_versions(),
    }
    if status != "running":
        begun = time.mktime(time.strptime(started, _TIME_FORMAT))
        record["wall_seconds"] = round(time.time() - begun, 1)
    record.update(extra or {})
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def read_record(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / RECORD_NAME
    if not path.is_file():
        raise MissingSourceError(f"{path}: no run record; this directory holds no run")
    return json.loads(path.read_text(encoding="utf-8"))


def update_record(directory: str | Path, **fields: Any) -> Path:
    """Add fields to an existing record, for steps that follow training inside a run."""
    path = Path(directory) / RECORD_NAME
    record = read_record(directory)
    record.update(fields)
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def latest_checkpoint(directory: str | Path) -> Path | None:
    """The `checkpoint-N` directory with the highest N, or None."""
    numbered = [
        (int(entry.name.rsplit("-", 1)[-1]), entry)
        for entry in Path(directory).glob("checkpoint-*")
        if entry.is_dir() and entry.name.rsplit("-", 1)[-1].isdigit()
    ]
    return max(numbered)[1] if numbered else None
