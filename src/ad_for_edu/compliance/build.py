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
from .rules import RuleMap
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
    check_keys(
        raw, _TOP_LEVEL, required=("modes", "sequence_factor"), where=source.name
    )
    modes = {
        name: Tier.from_settings(name, entry or {}) for name, entry in (raw["modes"] or {}).items()
    }
    if not modes:
        raise SettingsError(f"{source.name}: no modes defined")
    models = raw.get("models") or {}
    check_keys(models, ("encoder", "inference", "offline"), where=f"{source.name}:models")
    factor = raw.get("sequence_factor")
    if not factor:
        # Required, not defaulted. A compliance score is defined to include the
        # sequence factor, so settings that omit it describe a different quantity and
        # are refused rather than quietly scoring without it.
        raise SettingsError(
            f"{source.name}: sequence_factor is required. A compliance score includes it "
            "by definition, so there is no setting of this file that leaves it out."
        )
    return ComplianceSettings(
        modes=modes,
        sequence_factor=StrategySpec.parse(factor),
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
    name_rules: bool = True,
) -> ComplianceScorer:
    """The scorer of one mode. `models` replaces the local models, for tests.

    `name_rules` reads the rule table so that a failing component can say which rule
    it broke. It is on by default, because a score that cannot be attributed is the
    thing this metric exists to avoid.
    """
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
    return ComplianceScorer(tier, built, RuleMap.load() if name_rules else None)


def build_sequence_factor(
    settings: ComplianceSettings | None = None,
    *,
    knee: float | None = None,
) -> SequenceFactor:
    """The sequence factor of the settings file; `knee` overrides its knee.

    Always returns one. The settings loader requires the factor, so there is no path
    through this function that yields a scorer without it.
    """
    settings = settings or load_compliance_settings()
    if settings.sequence_factor is None:
        raise SettingsError(
            "the compliance settings carry no sequence factor; a compliance score "
            "includes it by definition"
        )
    params = dict(settings.sequence_factor.params)
    if knee is not None:
        params["knee"] = knee
    return SEQUENCE_FACTORS.create(settings.sequence_factor.name, **params)
