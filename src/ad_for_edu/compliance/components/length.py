"""Length: the description can be delivered in the time the moment offers.

The time on offer is the pause after the moment plus the longest delay the
delivery ladder allows. The score is the ratio of that budget to the spoken
duration of the description, capped at 1, so running over degrades the score
smoothly.
"""

from __future__ import annotations

from ...core.text import word_count
from ...data.schema import MomentContext
from .base import COMPONENTS, Component


@COMPONENTS.register("length_budget_ratio")
class LengthBudgetRatio(Component):
    """Budget over spoken duration, capped at 1."""

    category = "length"

    def __init__(self, words_per_minute: float = 140.0, lag_cap_seconds: float = 20.0) -> None:
        if words_per_minute <= 0:
            raise ValueError(f"words_per_minute must be positive, got {words_per_minute}")
        self.words_per_minute = words_per_minute
        self.lag_cap_seconds = lag_cap_seconds

    def score(self, text: str, moment: MomentContext) -> float:
        words = word_count(text)
        if not words:
            return 1.0
        duration = words / self.words_per_minute * 60.0
        budget = moment.pause + self.lag_cap_seconds
        return min(1.0, budget / duration) if duration > 0 else 1.0
