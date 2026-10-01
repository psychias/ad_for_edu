"""Deixis: a pointing word is followed by the name of what it points at.

"Look at this mitochondrion" resolves its pointing word; "look at this one here"
does not. The score is the share of pointing words in the description that are
followed, within a few words, by a term that is on the slide. A description that
narrates the gesture itself ("the cursor moves to ...") scores zero.

The component does not apply to a lecture whose recording shows no pointer,
because no description of such a moment could name a pointed-at element. That
fact has to be stated: when it is not supplied, the component scores.
"""

from __future__ import annotations

import re

from ...core.text import clean_description, content_terms, following_terms
from ...data.schema import MomentContext
from .base import COMPONENTS, Component

#: Wording that narrates the act of pointing instead of naming its target.
POINTING_ACT: tuple[str, ...] = (
    "points to",
    "pointing to",
    "pointing at",
    "points at",
    "is pointing",
    "the cursor",
    "the pointer",
    "the mouse",
    "gestures",
    "gesturing",
    "hovers over",
    "hovering over",
    "clicks on",
)

POINTING_WORD = re.compile(r"\b(?:this|these|those|that|here|there)\b", re.I)


@COMPONENTS.register("deixis_resolution")
class DeixisResolution(Component):
    """Share of pointing words that name their target within a window of words."""

    category = "deixis"

    def __init__(self, window: int = 6, premise_term_cap: int = 60) -> None:
        if window < 1:
            raise ValueError(f"window must be at least 1, got {window}")
        self.window = window
        self.premise_term_cap = premise_term_cap

    def applies(self, moment: MomentContext) -> bool:
        return moment.renders_cursor is not False

    def score(self, text: str, moment: MomentContext) -> float:
        description = clean_description(text)
        if not description:
            return 1.0
        lowered = description.lower()
        if any(phrase in lowered for phrase in POINTING_ACT):
            return 0.0
        pointing = list(POINTING_WORD.finditer(description))
        if not pointing:
            return 1.0
        premise = moment.premise.lower()
        if not content_terms(premise, cap=self.premise_term_cap):
            return 1.0
        resolved = 0
        for match in pointing:
            following = following_terms(description[match.end() :], self.window)
            if any(term.lower() in premise for term in content_terms(following, cap=self.window)):
                resolved += 1
        return resolved / len(pointing)
