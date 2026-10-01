"""Terminology: the description uses the terms that are on the slide.

The score is the share of the description's content terms that occur in the
slide premise. It is a proxy: a spatial word that is legitimately absent from
the slide counts against the description, so the component grades and never
gates.
"""

from __future__ import annotations

from ...core.text import clean_description, content_terms
from ...data.schema import MomentContext
from .base import COMPONENTS, Component


@COMPONENTS.register("terminology_grounding")
class TerminologyGrounding(Component):
    """Share of content terms found in the slide text or on-screen summary."""

    category = "terminology"

    def __init__(self, term_cap: int = 40) -> None:
        self.term_cap = term_cap

    def score(self, text: str, moment: MomentContext) -> float:
        description = clean_description(text)
        if not description:
            return 1.0
        terms = content_terms(description, cap=self.term_cap)
        if not terms:
            return 1.0
        premise = moment.premise.lower()
        grounded = sum(1 for term in terms if term.lower() in premise)
        return grounded / len(terms)
