"""Wiring: from the settings files to a training cell."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.errors import SettingsError
from ..core.runs import RunKey
from ..core.settings import check_keys, config_dir, load_yaml
from .backbones import Backbone, load_backbones, settings_for
from .examples import INPUT_ARMS, InputArm
from .methods import TRAINING_METHODS, LoraSettings, TrainingMethod, TrainingSettings

TRAINING_DIR = "training"
GRID_FILE = "grid.yaml"
DEFAULTS_FILE = "defaults.yaml"


@dataclass(frozen=True)
class Grid:
    """The cells of the experiment."""

    backbones: tuple[str, ...]
    arms: tuple[str, ...]
    methods: tuple[str, ...]
    seeds: tuple[int, ...]
    development: Mapping[str, Mapping[str, Any]]

    def cells(self) -> list[RunKey]:
        return [
            RunKey(backbone, arm, method, seed)
            for backbone in self.backbones
            for arm in self.arms
            for method in self.methods
            for seed in self.seeds
        ]

    def development_for(self, method: str) -> dict[str, Any]:
        if method not in self.development:
            raise SettingsError(f"the grid says nothing about what {method} holds out")
        return dict(self.development[method])


def training_dir(base: str | Path | None = None) -> Path:
    return (Path(base) if base else config_dir()) / TRAINING_DIR


def load_grid(path: str | Path | None = None) -> Grid:
    source = Path(path) if path else training_dir() / GRID_FILE
    raw = load_yaml(source)
    check_keys(
        raw,
        ("backbones", "arms", "methods", "seeds", "development"),
        required=("backbones", "arms", "methods", "seeds"),
        where=source.name,
    )
    development = raw.get("development") or {}
    for method, values in development.items():
        check_keys(
            values or {},
            ("share", "seed", "by"),
            where=f"{source.name}:development.{method}",
        )
    return Grid(
        backbones=tuple(raw["backbones"]),
        arms=tuple(raw["arms"]),
        methods=tuple(raw["methods"]),
        seeds=tuple(int(seed) for seed in raw["seeds"]),
        development={name: dict(values or {}) for name, values in development.items()},
    )


def load_layer(name: str, base: str | Path | None = None) -> dict[str, Any]:
    """One layer of settings: the defaults, or a method's, or an arm's."""
    path = training_dir(base) / name
    return dict(load_yaml(path))


def settings_of(
    key: RunKey,
    *,
    base: str | Path | None = None,
    backbones: Mapping[str, Backbone] | None = None,
) -> tuple[TrainingSettings, LoraSettings, Backbone]:
    """The settings of one cell, layered and checked."""
    available = backbones or load_backbones(training_dir(base) / "backbones")
    if key.backbone not in available:
        raise SettingsError(
            f"unknown backbone {key.backbone!r}; known: {sorted(available)}"
        )
    if key.arm not in INPUT_ARMS.names():
        raise SettingsError(f"unknown input arm {key.arm!r}; known: {list(INPUT_ARMS.names())}")
    if key.method not in TRAINING_METHODS.names():
        raise SettingsError(
            f"unknown method {key.method!r}; known: {list(TRAINING_METHODS.names())}"
        )
    backbone = available[key.backbone]
    layered = settings_for(
        key,
        backbone,
        defaults=load_layer(DEFAULTS_FILE, base),
        method=load_layer(f"methods/{key.method}.yaml", base),
    )
    lora = LoraSettings(**{**LoraSettings().__dict__, **(layered.pop("lora", None) or {})})
    lora = LoraSettings(
        rank=int(lora.rank),
        alpha=int(lora.alpha),
        dropout=float(lora.dropout),
        targets=tuple(lora.targets),
    )
    layered.pop("processor_image_kwargs", None)
    return TrainingSettings(**layered), lora, backbone


def build_arm(name: str) -> InputArm:
    return INPUT_ARMS.create(name)


def build_method(name: str) -> TrainingMethod:
    return TRAINING_METHODS.create(name)


def processor_kwargs(key: RunKey, backbone: Backbone) -> dict[str, Any]:
    """What the backbone needs told about how to prepare its pictures."""
    return dict(backbone.for_arm(key.arm).get("processor_image_kwargs") or {})
