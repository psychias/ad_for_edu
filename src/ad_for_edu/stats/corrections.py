"""Correction for a family of tests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

from ..core.registry import Registry


@dataclass(frozen=True)
class Adjusted:
    raw: float
    adjusted: float
    reject: bool


class Correction(ABC):
    """Adjusts the p-values of a family of tests for their number."""

    @abstractmethod
    def adjust(self, p_values: Mapping[str, float], alpha: float = 0.05) -> dict[str, Adjusted]: ...


CORRECTIONS: Registry[Correction] = Registry("multiple-test correction", Correction)


@CORRECTIONS.register("holm")
class Holm(Correction):
    """Step-down correction: the smallest p-value is multiplied by the number of tests,
    the next by one fewer, and so on. Adjusted values never decrease along that order,
    and rejection stops at the first test that is not rejected.

    The adjusted value is not rounded: rounding belongs to a report, and a value
    rounded here would flatten a very small one to zero."""

    def adjust(self, p_values: Mapping[str, float], alpha: float = 0.05) -> dict[str, Adjusted]:
        count = len(p_values)
        ranked = sorted(p_values.items(), key=lambda item: item[1])
        adjusted: dict[str, Adjusted] = {}
        running = 0.0
        rejecting = True
        for position, (name, raw) in enumerate(ranked):
            running = max(running, min(1.0, (count - position) * raw))
            rejecting = rejecting and running <= alpha
            adjusted[name] = Adjusted(raw, running, rejecting)
        return adjusted
