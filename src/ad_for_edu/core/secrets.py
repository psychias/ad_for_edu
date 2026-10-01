"""Credentials come from the environment.

A `.env` file in the current directory is read once, for convenience: each
`NAME=value` line sets a variable that the environment does not already define.
A credential is never written to a settings file, a run record or a log line.
"""

from __future__ import annotations

import os
from pathlib import Path

from .errors import SettingsError

GEMINI = "GEMINI_API_KEY"
OPENROUTER = "OPENROUTER_API_KEY"
OPENAI = "OPENAI_API_KEY"
HUB = "HF_TOKEN"

_loaded: set[Path] = set()


def load_env_file(path: str | Path = ".env") -> int:
    """Set the variables listed in `path` that are not already set. Returns how many were set."""
    source = Path(path).resolve()
    if source in _loaded or not source.is_file():
        return 0
    _loaded.add(source)
    added = 0
    for line in source.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        name, value = name.strip(), value.strip().strip("\"'")
        if name and value and not os.environ.get(name):
            os.environ[name] = value
            added += 1
    return added


def credential(name: str) -> str:
    """The credential `name`; an error that names the variable when it is missing."""
    load_env_file()
    value = os.environ.get(name, "").strip()
    if not value:
        raise SettingsError(f"credential {name} is not set in the environment or in .env")
    return value


def has_credential(name: str) -> bool:
    load_env_file()
    return bool(os.environ.get(name, "").strip())
