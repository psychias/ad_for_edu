"""Wiring: from the settings file to a reference generator."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.settings import check_keys, config_dir, load_yaml
from ..core.strategy import StrategySpec
from ..moments.build import build_context_builder, build_still_policy
from ..moments.context import ContextBuilder
from ..moments.stills import StillPolicy
from .generator import ReferenceGenerator, WriterAnswer
from .prompts import REFERENCE_PROMPTS, ReferencePrompt

SETTINGS_FILE = "references.yaml"
_KEYS = (
    "prompt",
    "writers",
    "temperature",
    "max_tokens",
    "reasoning_effort",
    "still_policy",
    "transcript_window",
    "rung_policy",
    "described_window",
)


@dataclass(frozen=True)
class ReferenceSettings:
    prompt: StrategySpec
    writers: tuple[str, ...]
    temperature: float
    max_tokens: int
    reasoning_effort: str | None
    still_policy: StrategySpec
    transcript_window: StrategySpec
    rung_policy: StrategySpec
    described_window: int


def load_reference_settings(path: str | Path | None = None) -> ReferenceSettings:
    source = Path(path) if path else config_dir() / SETTINGS_FILE
    raw = load_yaml(source)
    check_keys(raw, _KEYS, required=("prompt", "writers"), where=source.name)
    return ReferenceSettings(
        prompt=StrategySpec.parse(raw["prompt"]),
        writers=tuple(raw["writers"] or ()),
        temperature=float(raw.get("temperature", 0.2)),
        max_tokens=int(raw.get("max_tokens", 2000)),
        reasoning_effort=raw.get("reasoning_effort") or None,
        still_policy=StrategySpec.parse(raw.get("still_policy") or "keyframe_window"),
        transcript_window=StrategySpec.parse(raw.get("transcript_window") or "next_event"),
        rung_policy=StrategySpec.parse(raw.get("rung_policy") or "uniform_budget"),
        described_window=int(raw.get("described_window", 40)),
    )


def build_prompt(settings: ReferenceSettings) -> ReferencePrompt:
    return REFERENCE_PROMPTS.create_from(settings.prompt)


def build_context(settings: ReferenceSettings) -> ContextBuilder:
    return build_context_builder(settings.transcript_window, settings.rung_policy)


def build_stills(settings: ReferenceSettings) -> StillPolicy:
    return build_still_policy(settings.still_policy)


def build_generator(
    ask: Callable[[str, str, Sequence[Any]], WriterAnswer],
    settings: ReferenceSettings | None = None,
) -> ReferenceGenerator:
    settings = settings or load_reference_settings()
    return ReferenceGenerator(build_prompt(settings), settings.writers, ask)
