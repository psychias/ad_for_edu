"""Agreement between raters.

A unit is one rated item: the list of the labels the raters gave it, with None
where a rater gave none. Units with fewer than two labels carry no information
about agreement and are skipped.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from ..core.registry import Registry


class AgreementCoefficient(ABC):
    """A chance-corrected measure of agreement; 1 is perfect agreement."""

    @abstractmethod
    def value(self, units: Sequence[Sequence[Any]]) -> float: ...


AGREEMENT: Registry[AgreementCoefficient] = Registry(
    "agreement coefficient", AgreementCoefficient
)


@AGREEMENT.register("krippendorff_nominal")
class KrippendorffNominal(AgreementCoefficient):
    """Krippendorff's alpha for nominal labels.

    alpha = 1 - observed disagreement / disagreement expected from the label totals.
    """

    def value(self, units: Sequence[Sequence[Any]]) -> float:
        coincidences: dict[tuple[Any, Any], float] = defaultdict(float)
        for unit in units:
            labels = [label for label in unit if label is not None]
            size = len(labels)
            if size < 2:
                continue
            weight = 1.0 / (size - 1)
            for i in range(size):
                for j in range(size):
                    if i != j:
                        coincidences[(labels[i], labels[j])] += weight
        total = sum(coincidences.values())
        if total <= 1:
            return 1.0
        totals: dict[Any, float] = defaultdict(float)
        for (label, _other), count in coincidences.items():
            totals[label] += count
        observed = sum(count for (a, b), count in coincidences.items() if a != b) / total
        expected = sum(
            totals[a] * totals[b] for a in totals for b in totals if a != b
        ) / (total * (total - 1))
        if expected == 0:
            return 1.0
        return 1.0 - observed / expected


def complete_units(
    labels_by_rater: dict[str, dict[str, Any]], items: Sequence[str]
) -> list[list[Any]]:
    """One unit per item that every rater labelled, raters in sorted order."""
    raters = sorted(labels_by_rater)
    units = []
    for item in items:
        labels = [labels_by_rater[rater].get(item) for rater in raters]
        if all(label is not None for label in labels):
            units.append(labels)
    return units


def unanimous(units: Sequence[Sequence[Any]], *, ignoring: Any = None) -> int:
    """Number of units on which every rater gave the same label, other than `ignoring`."""
    return sum(1 for unit in units if len(set(unit)) == 1 and unit[0] != ignoring)
