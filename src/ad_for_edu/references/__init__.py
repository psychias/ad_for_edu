"""Reference descriptions: the ways of asking, the memory per lecture, the writers."""

from .described import DEFAULT_WINDOW, DescribedSoFar, by_lecture, in_order
from .generator import (
    ReferenceGenerator,
    ReferenceSet,
    WriterAnswer,
    complete_only,
    incomplete_moments,
    to_row,
    writers_per_moment,
)
from .prompts import COVERAGE, REFERENCE_PROMPTS, ReferenceDecision, ReferencePrompt

__all__ = [
    "COVERAGE",
    "DEFAULT_WINDOW",
    "REFERENCE_PROMPTS",
    "DescribedSoFar",
    "ReferenceDecision",
    "ReferenceGenerator",
    "ReferencePrompt",
    "ReferenceSet",
    "WriterAnswer",
    "by_lecture",
    "complete_only",
    "in_order",
    "incomplete_moments",
    "to_row",
    "writers_per_moment",
]
