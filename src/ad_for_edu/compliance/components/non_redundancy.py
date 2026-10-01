"""Non-redundancy: the description does not repeat what the lecturer says.

A sentence encoder gives the cosine similarity between the description and the
transcript window of the moment. Similarity at or below the knee earns full
credit; above it the score falls linearly and reaches zero at similarity 1.

The knee sits at the lower decile of the similarities observed between
descriptions and their transcript windows, so most descriptions fall in the
graded region. It was set from that distribution alone and never fitted to
preference labels.

The transcript window is used as stored, including a marker for a silent window.
"""

from __future__ import annotations

from ...core.text import clean_description
from ...data.schema import MomentContext
from ..models import LocalModels
from .base import COMPONENTS, Component


def credit_for_similarity(similarity: float, knee: float) -> float:
    """Full credit at or below `knee`, falling linearly to 0 at similarity 1."""
    if similarity <= knee:
        return 1.0
    return max(0.0, (1.0 - similarity) / (1.0 - knee))


@COMPONENTS.register("non_redundancy_embedding")
class NonRedundancyEmbedding(Component):
    """Graded credit for low similarity between description and transcript window."""

    category = "non_redundancy"
    # Graded credit from similarity to what was just said. Some overlap is
    # unavoidable, so a shortfall alone does not name the rule.
    shortfall_is_a_breach = False
    requires_models = True

    def __init__(self, models: LocalModels, knee: float = 0.17) -> None:
        if not 0 <= knee < 1:
            raise ValueError(f"knee must be in [0, 1), got {knee}")
        self.models = models
        self.knee = knee

    def score(self, text: str, moment: MomentContext) -> float:
        description = clean_description(text)
        spoken = (moment.transcript_window or "").strip()
        if not description or not spoken:
            return 1.0
        return credit_for_similarity(self.models.similarity(description, spoken), self.knee)
