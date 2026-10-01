"""Wiring: from the settings file to the ways of asking and of reading a reply."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.errors import SettingsError
from ..core.settings import check_keys, config_dir, load_yaml
from ..core.strategy import StrategySpec
from .decoding import DECODING_ATTEMPTS, SALVAGE_STEPS, DecodingAttempt, SalvageStep

SETTINGS_FILE = "inference.yaml"
_KEYS = ("attempts", "salvage", "checkpoint")
#: Which adapter of a run is asked for.
CHECKPOINTS = ("final", "best")


@dataclass(frozen=True)
class InferenceSettings:
    """How a describer is asked, and how its reply is read."""

    attempts: tuple[StrategySpec, ...]
    salvage: tuple[StrategySpec, ...]
    checkpoint: str = "final"


def load_inference_settings(path: str | Path | None = None) -> InferenceSettings:
    source = Path(path) if path else config_dir() / SETTINGS_FILE
    raw = load_yaml(source)
    check_keys(raw, _KEYS, required=("attempts", "salvage"), where=source.name)
    attempts = tuple(StrategySpec.parse(entry) for entry in raw["attempts"] or ())
    salvage = tuple(StrategySpec.parse(entry) for entry in raw["salvage"] or ())
    if not attempts:
        raise SettingsError(f"{source.name}: no way of asking is listed")
    if not salvage:
        raise SettingsError(f"{source.name}: no way of reading a reply is listed")
    checkpoint = str(raw.get("checkpoint", "final"))
    if checkpoint not in CHECKPOINTS:
        raise SettingsError(
            f"{source.name}: checkpoint {checkpoint!r} is not one of {list(CHECKPOINTS)}"
        )
    return InferenceSettings(attempts, salvage, checkpoint)


def build_attempts(settings: InferenceSettings) -> list[DecodingAttempt]:
    return [DECODING_ATTEMPTS.create_from(spec) for spec in settings.attempts]


def build_salvage(settings: InferenceSettings) -> list[SalvageStep]:
    return [SALVAGE_STEPS.create_from(spec) for spec in settings.salvage]
