"""Shared test configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

ENVIRONMENT = (
    "AD_FOR_EDU_DATA_DIR",
    "AD_FOR_EDU_WORK_DIR",
    "AD_FOR_EDU_CONFIG_DIR",
    "AD_FOR_EDU_REFERENCES_DATASET",
    "AD_FOR_EDU_PREFERENCES_DATASET",
    "AD_FOR_EDU_KEYFRAMES_DATASET",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "HF_TOKEN",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every test starts without the package's variables and in an empty directory.

    No test may depend on the machine it runs on, and none may reach a credential.
    """
    for name in ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def shipped_configs() -> Path:
    return ROOT / "configs"
