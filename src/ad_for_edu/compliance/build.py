"""Wiring: from the settings file to a scorer.

This is the only module of the package that reads settings and looks strategies
up by name. Everything else receives what it needs through its constructor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import SettingsError
from ..core.settings import check_keys, config_dir, load_yaml
from ..core.strategy import StrategySpec
from . import components as _components  # noqa: F401  (registers the components)
from .components.base import COMPONENTS
from .models import ENCODER, INFERENCE, HuggingFaceModels, LocalModels
from .scorer import ComplianceScorer
from .sequence import SEQUENCE_FACTORS, SequenceFactor
from .tiers import Tier, tier_named

SETTINGS_FILE = "compliance.yaml"
_TOP_LEVEL = ("modes", "sequence_factor", "models", "kappa_sweep")


@dataclass(frozen=True)
class ComplianceSettings:
    """The content of the compliance settings file."""

    modes: Mapping[str, Tier]
    sequence_factor: StrategySpec | None
    models: Mapping[str, Any] = field(default_factory=dict)
    kappa_sweep: tuple[float, ...] = ()


def load_compliance_settings(path: str | Path | None = None) -> ComplianceSettings:
    source = Path(path) if path else config_dir() / SETTINGS_FILE
    raw = load_yaml(source)
    check_keys(raw, _TOP_LEVEL, required=("modes",), where=source.name)
    modes = {
        name: Tier.from_settings(name, entry or {}) for name, entry in (raw["modes"] or {}).items()
    }
    if not modes:
        raise SettingsError(f"{source.name}: no modes defined")
    models = raw.get("models") or {}
    check_keys(models, ("encoder", "inference", "offline"), where=f"{source.name}:models")
    factor = raw.get("sequence_factor")
    return ComplianceSettings(
        modes=modes,
        sequence_factor=StrategySpec.parse(factor) if factor else None,
        models=models,
        kappa_sweep=tuple(float(value) for value in raw.get("kappa_sweep") or ()),
    )


def build_models(settings: ComplianceSettings) -> LocalModels:
    return HuggingFaceModels(
        encoder=settings.models.get("encoder", ENCODER),
        inference=settings.models.get("inference", INFERENCE),
        offline=bool(settings.models.get("offline", True)),
    )


def build_scorer(
    mode: str,
    settings: ComplianceSettings | None = None,
    *,
    models: LocalModels | None = None,
) -> ComplianceScorer:
    """The scorer of one mode. `models` replaces the local models, for tests."""
    settings = settings or load_compliance_settings()
    tier = tier_named(mode, settings.modes)
    if tier.paid:
        raise SettingsError(
            f"mode {mode!r} is a hosted judge; use the rubric judging command for it"
        )
    built = []
    shared: LocalModels | None = models
    for entry in tier.components:
        spec = StrategySpec.parse(entry)
        params = dict(spec.params)
        if COMPONENTS.get(spec.name).requires_models:
            if shared is None:
                shared = build_models(settings)
            params["models"] = shared
        built.append(COMPONENTS.create(spec.name, **params))
    return ComplianceScorer(tier, built)


def build_sequence_factor(
    settings: ComplianceSettings | None = None,
    *,
    knee: float | None = None,
) -> SequenceFactor | None:
    """The sequence factor of the settings file; `knee` overrides its knee."""
    settings = settings or load_compliance_settings()
    if settings.sequence_factor is None:
        return None
    params = dict(settings.sequence_factor.params)
    if knee is not None:
        params["knee"] = knee
    return SEQUENCE_FACTORS.create(settings.sequence_factor.name, **params)
