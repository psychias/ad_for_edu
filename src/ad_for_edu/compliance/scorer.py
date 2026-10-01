"""The scorer: components bound to the mode they run as.

    score(text, moment) = mean of the components that apply to the moment

Binding the components to a mode is the point of this class. Every result is
checked against what the mode promises, so a scorer built with model-backed
components cannot report itself as mechanical, and a component that stops
scoring is noticed at the first description instead of in a table.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import ContractError
from ..data.schema import CATEGORIES, MomentContext, is_unfilled
from .components.base import Component
from .tiers import Tier

UNFILLED = "unfilled_text"
MISSING = "missing"


@dataclass
class ScoreRow:
    """One scored description. `components` has every category; None means not scored."""

    overall: float | None
    components: dict[str, float | None]
    abstained: tuple[str, ...]
    reason: str | None = None
    novelty: float | None = None
    novelty_factor: float | None = None
    fields: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def unscored(cls, reason: str) -> ScoreRow:
        return cls(None, {category: None for category in CATEGORIES}, tuple(CATEGORIES), reason)

    @property
    def scored(self) -> bool:
        return self.overall is not None

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "overall": self.overall,
            "components": dict(self.components),
            "abstained": list(self.abstained),
            "reason": self.reason,
        }
        if self.novelty is not None:
            row["novelty"] = self.novelty
            row["novelty_factor"] = self.novelty_factor
        row.update(self.fields)
        return row


class ComplianceScorer:
    """Scores one description against the context of its moment."""

    def __init__(self, tier: Tier, components: Sequence[Component]) -> None:
        if tier.paid:
            raise ContractError(
                f"mode {tier.name!r} is a hosted judge; it is run by the judging stage, "
                "not by this scorer"
            )
        categories = [component.category for component in components]
        repeated = sorted({c for c in categories if categories.count(c) > 1})
        if repeated:
            raise ContractError(f"mode {tier.name!r}: more than one component for {repeated}")
        expected = set(CATEGORIES) - set(tier.abstains)
        if set(categories) != expected:
            raise ContractError(
                f"mode {tier.name!r} promises {sorted(expected)} "
                f"but was given components for {sorted(categories)}"
            )
        self.tier = tier
        # Components are applied in category order so that the mean is summed in one
        # fixed order, whatever order the settings list them in.
        self.components = tuple(sorted(components, key=lambda c: CATEGORIES.index(c.category)))

    @property
    def name(self) -> str:
        return self.tier.name

    def score(self, text: Any, moment: MomentContext) -> ScoreRow:
        """Score `text`. An empty description keeps its place as an unscored row."""
        if is_unfilled(text):
            return ScoreRow.unscored(UNFILLED)
        description = str(text)
        scores: dict[str, float | None] = {category: None for category in CATEGORIES}
        for component in self.components:
            if component.applies(moment):
                value = float(component.score(description, moment))
                if not 0.0 <= value <= 1.0:
                    raise ContractError(
                        f"{type(component).__name__} returned {value} outside [0, 1]"
                    )
                scores[component.category] = value
        applied = [value for value in scores.values() if value is not None]
        if not applied:
            raise ContractError(f"no component applies to moment {moment.moment_id!r}")
        abstained = tuple(category for category in CATEGORIES if scores[category] is None)
        self._check_abstentions(abstained, moment)
        return ScoreRow(sum(applied) / len(applied), scores, abstained)

    def _check_abstentions(self, abstained: tuple[str, ...], moment: MomentContext) -> None:
        """The categories left out must be the ones the mode leaves out, plus deixis
        on a lecture stated to show no pointer."""
        unexpected = set(abstained) - set(self.tier.abstains) - {"deixis"}
        missing = set(self.tier.abstains) - set(abstained)
        if unexpected or missing:
            raise ContractError(
                f"mode {self.tier.name!r} left out {sorted(abstained)}; "
                f"it promises to leave out {sorted(self.tier.abstains)}"
            )
        if "deixis" in abstained and "deixis" not in self.tier.abstains:
            if moment.renders_cursor is not False:
                raise ContractError(
                    "deixis was left out although the lecture was not stated to show no pointer"
                )
