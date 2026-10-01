"""The metric family: one score per described moment.

A metric is handed the items to score and returns one value per item, or None
where it cannot score that item. It never drops an item: the caller counts how
many were scored and reports that beside the mean.

Three properties of a metric decide how it can be used, and each is declared:
whether it needs the reference descriptions of a moment, whether it needs the
image, and whether it reads a store of answers a paid judge already gave.

A moment usually has several reference descriptions, written independently. A
reference-based metric scores a candidate against each of them and keeps the best
value: the candidate is not wrong for resembling one describer rather than another.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from ..core.registry import Registry


@dataclass(frozen=True)
class ScoringItem:
    """One description to score, with what is known about its moment."""

    moment_id: str
    text: str | None
    references: tuple[str, ...] = ()
    image: Path | None = None
    fields: dict[str, object] = field(default_factory=dict)

    @property
    def described(self) -> bool:
        return bool((self.text or "").strip())


class Metric(ABC):
    """Scores descriptions. One value per item, or None where it cannot score."""

    needs_references: ClassVar[bool] = False
    needs_image: ClassVar[bool] = False
    #: Whether the metric reads answers a paid judge already gave, rather than scoring itself.
    reads_answers: ClassVar[bool] = False

    @abstractmethod
    def score(self, items: Sequence[ScoringItem]) -> list[float | None]: ...

    def scorable(self, item: ScoringItem) -> bool:
        """Whether this metric can score the item at all."""
        if not item.described:
            return False
        if self.needs_references and not item.references:
            return False
        if self.needs_image and item.image is None:
            return False
        return True


METRICS: Registry[Metric] = Registry("metric", Metric)


def best_over_references(
    items: Sequence[ScoringItem],
    score_pairs: Callable[[list[str], list[str]], list[float]],
) -> list[float | None]:
    """The best score of each description over the references of its moment.

    Every candidate-reference pair of every item is scored in one call, so a metric
    that loads a model pays for loading it once.
    """
    pairs: list[tuple[str, str]] = []
    spans: list[tuple[int, int]] = []
    for item in items:
        start = len(pairs)
        if item.described:
            pairs.extend((item.text or "", reference) for reference in item.references)
        spans.append((start, len(pairs)))
    if not pairs:
        return [None] * len(items)
    candidates = [candidate for candidate, _reference in pairs]
    references = [reference for _candidate, reference in pairs]
    values = score_pairs(candidates, references)
    return [max(values[start:end]) if end > start else None for start, end in spans]


def mean_of(values: Sequence[float | None]) -> float | None:
    """The mean of the values that exist, or None when none do."""
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None
