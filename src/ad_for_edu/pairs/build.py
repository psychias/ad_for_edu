"""Wiring: from the settings files to the pair stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.settings import check_keys, config_dir, load_yaml
from ..core.strategy import StrategySpec
from .axes import CONTROLLED_AXES, ControlledAxis
from .candidates import CANDIDATE_MODES, CandidateMode, check_directives
from .ordering import ORDER_COMBINATIONS, OrderCombination
from .pool import DrawPlan

CANDIDATES_FILE = "candidates.yaml"
CONTROLLED_FILE = "controlled_pairs.yaml"
PAIR_SETS_FILE = "pair_sets.yaml"
JUDGING_FILE = "pair_judging.yaml"


@dataclass(frozen=True)
class CandidateSettings:
    model: str
    mode: StrategySpec
    directives: tuple[str, ...]
    temperature: float
    max_tokens: int
    reasoning_effort: str | None


@dataclass(frozen=True)
class ControlledSettings:
    model: str
    axes: tuple[str, ...]
    temperature: float
    max_tokens: int
    per_axis: int
    compliant_side: str


@dataclass(frozen=True)
class PairSetSettings:
    maximum_per_moment: int
    minimum_distance: float
    maximum_per_description: int
    pairing_seed: int
    rated_pool: DrawPlan
    head_to_head: dict[str, Any] = field(default_factory=dict)
    development_share: float = 0.12
    development_seed: int = 17


@dataclass(frozen=True)
class JudgingSettings:
    model: str
    clip_seconds: float
    clip_width: int
    clip_frames_per_second: int
    temperature: float
    max_tokens: int
    combination: StrategySpec


def load_candidate_settings(path: str | Path | None = None) -> CandidateSettings:
    source = Path(path) if path else config_dir() / CANDIDATES_FILE
    raw = load_yaml(source)
    check_keys(
        raw,
        ("model", "mode", "directives", "temperature", "max_tokens", "reasoning_effort"),
        required=("model", "mode", "directives"),
        where=source.name,
    )
    directives = tuple(raw["directives"] or ())
    check_directives(directives)
    return CandidateSettings(
        model=str(raw["model"]),
        mode=StrategySpec.parse(raw["mode"]),
        directives=directives,
        temperature=float(raw.get("temperature", 0.7)),
        max_tokens=int(raw.get("max_tokens", 2000)),
        reasoning_effort=raw.get("reasoning_effort") or None,
    )


def load_controlled_settings(path: str | Path | None = None) -> ControlledSettings:
    source = Path(path) if path else config_dir() / CONTROLLED_FILE
    raw = load_yaml(source)
    check_keys(
        raw,
        ("model", "axes", "temperature", "max_tokens", "per_axis", "compliant_side"),
        required=("model", "axes"),
        where=source.name,
    )
    return ControlledSettings(
        model=str(raw["model"]),
        axes=tuple(raw["axes"] or ()),
        temperature=float(raw.get("temperature", 0.3)),
        max_tokens=int(raw.get("max_tokens", 1500)),
        per_axis=int(raw.get("per_axis", 40)),
        compliant_side=str(raw.get("compliant_side", "a")),
    )


def load_pair_set_settings(path: str | Path | None = None) -> PairSetSettings:
    source = Path(path) if path else config_dir() / PAIR_SETS_FILE
    raw = load_yaml(source)
    check_keys(
        raw,
        ("pairing", "rated_pool", "head_to_head", "development"),
        required=("pairing", "rated_pool"),
        where=source.name,
    )
    pairing = raw["pairing"]
    check_keys(
        pairing,
        ("maximum_per_moment", "minimum_distance", "maximum_per_description", "seed"),
        where=f"{source.name}:pairing",
    )
    pool = raw["rated_pool"]
    check_keys(
        pool,
        ("per_kind", "per_category", "category_default", "maximum_per_moment", "seed"),
        required=("per_kind",),
        where=f"{source.name}:rated_pool",
    )
    development = raw.get("development") or {}
    check_keys(development, ("share", "seed"), where=f"{source.name}:development")
    return PairSetSettings(
        maximum_per_moment=int(pairing.get("maximum_per_moment", 4)),
        minimum_distance=float(pairing.get("minimum_distance", 12.0)),
        maximum_per_description=int(pairing.get("maximum_per_description", 2)),
        pairing_seed=int(pairing.get("seed", 5)),
        rated_pool=DrawPlan(
            per_kind=dict(pool["per_kind"]),
            per_category=dict(pool.get("per_category") or {}),
            category_default=int(pool.get("category_default", 0)),
            maximum_per_moment=int(pool.get("maximum_per_moment", 2)),
            seed=int(pool.get("seed", 13)),
        ),
        head_to_head=dict(raw.get("head_to_head") or {}),
        development_share=float(development.get("share", 0.12)),
        development_seed=int(development.get("seed", 17)),
    )


def load_judging_settings(path: str | Path | None = None) -> JudgingSettings:
    source = Path(path) if path else config_dir() / JUDGING_FILE
    raw = load_yaml(source)
    check_keys(
        raw,
        (
            "model",
            "clip_seconds",
            "clip_width",
            "clip_frames_per_second",
            "temperature",
            "max_tokens",
            "combination",
        ),
        required=("model", "combination"),
        where=source.name,
    )
    return JudgingSettings(
        model=str(raw["model"]),
        clip_seconds=float(raw.get("clip_seconds", 16.0)),
        clip_width=int(raw.get("clip_width", 640)),
        clip_frames_per_second=int(raw.get("clip_frames_per_second", 2)),
        temperature=float(raw.get("temperature", 0.0)),
        max_tokens=int(raw.get("max_tokens", 1200)),
        combination=StrategySpec.parse(raw["combination"]),
    )


def build_candidate_mode(settings: CandidateSettings) -> CandidateMode:
    return CANDIDATE_MODES.create_from(settings.mode)


def build_axes(settings: ControlledSettings) -> list[ControlledAxis]:
    return [CONTROLLED_AXES.create(name) for name in settings.axes]


def build_combination(settings: JudgingSettings) -> OrderCombination:
    return ORDER_COMBINATIONS.create_from(settings.combination)
