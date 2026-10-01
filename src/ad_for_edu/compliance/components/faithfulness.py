"""Faithfulness: the description does not contradict what is on the slide.

An inference model reads the slide premise and the description and returns the
probability that the second contradicts the first. The score is one minus that
probability. With no premise there is nothing to contradict, and the score is 1.
"""

from __future__ import annotations

from ...core.text import clean_description
from ...data.schema import MomentContext
from ..models import LocalModels
from .base import COMPONENTS, Component


@COMPONENTS.register("faithfulness_nli")
class FaithfulnessNLI(Component):
    """One minus the contradiction probability of (slide premise, description)."""

    category = "faithfulness"
    # One minus a contradiction probability. A small reading is not a
    # contradiction, so a shortfall alone does not name the rule.
    shortfall_is_a_breach = False
    requires_models = True

    def __init__(self, models: LocalModels) -> None:
        self.models = models

    def score(self, text: str, moment: MomentContext) -> float:
        description = clean_description(text)
        if not description:
            return 1.0
        premise = moment.premise
        if not premise:
            return 1.0
        return 1.0 - self.models.contradiction(premise, description)
