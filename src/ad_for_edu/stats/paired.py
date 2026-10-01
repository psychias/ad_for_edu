"""Paired comparison of two scorers on the same pairs.

Each scorer gives a verdict per pair: True when it preferred the right side,
False when it preferred the wrong one, None when it did not decide. A paired
test needs a rule for the pairs on which a scorer did not decide, and the rule
changes the result, so it is a strategy of its own and is printed with every test.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

from ..core.registry import Registry

Verdicts = Mapping[str, "bool | None"]


@dataclass(frozen=True)
class TestResult:
    """The outcome of one paired test."""

    __test__ = False  # not a test class, whatever its name suggests to a test runner

    n: int
    first_only: int
    second_only: int
    p_value: float
    accuracy_first: float | None
    accuracy_second: float | None
    pairing: str
    test: str

    @property
    def discordant(self) -> tuple[int, int]:
        return (self.first_only, self.second_only)

    def as_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "discordant": list(self.discordant),
            "p": self.p_value,
            "accuracy_first": self.accuracy_first,
            "accuracy_second": self.accuracy_second,
            "pairing": self.pairing,
            "test": self.test,
        }


class PairingRule(ABC):
    """Which pairs enter a paired test."""

    @abstractmethod
    def select(self, first: Verdicts, second: Verdicts) -> list[str]: ...


PAIRINGS: Registry[PairingRule] = Registry("pairing rule", PairingRule)


@PAIRINGS.register("all_pairs")
class AllPairs(PairingRule):
    """Every pair both scorers were asked about. Not deciding counts as not right."""

    def select(self, first: Verdicts, second: Verdicts) -> list[str]:
        return sorted(set(first) & set(second))


@PAIRINGS.register("both_decide")
class BothDecide(PairingRule):
    """Only the pairs on which both scorers decided."""

    def select(self, first: Verdicts, second: Verdicts) -> list[str]:
        return sorted(
            pair
            for pair in set(first) & set(second)
            if first[pair] is not None and second[pair] is not None
        )


class PairedTest(ABC):
    """Tests whether one scorer is right more often than another on the same pairs."""

    @abstractmethod
    def test(self, first: Verdicts, second: Verdicts, pairing: PairingRule) -> TestResult: ...


PAIRED_TESTS: Registry[PairedTest] = Registry("paired test", PairedTest)


def exact_two_sided(first_only: int, second_only: int) -> float:
    """Exact two-sided binomial p-value on the discordant pairs; 1 when there are none."""
    from scipy.stats import binomtest

    total = first_only + second_only
    if not total:
        return 1.0
    return float(binomtest(first_only, total, 0.5).pvalue)


@PAIRED_TESTS.register("exact_mcnemar")
class ExactMcNemar(PairedTest):
    """Exact binomial test on the pairs where exactly one scorer is right."""

    def test(self, first: Verdicts, second: Verdicts, pairing: PairingRule) -> TestResult:
        selected = pairing.select(first, second)
        first_right = {pair: first[pair] is True for pair in selected}
        second_right = {pair: second[pair] is True for pair in selected}
        first_only = sum(1 for pair in selected if first_right[pair] and not second_right[pair])
        second_only = sum(1 for pair in selected if second_right[pair] and not first_right[pair])
        count = len(selected)
        return TestResult(
            n=count,
            first_only=first_only,
            second_only=second_only,
            p_value=exact_two_sided(first_only, second_only),
            accuracy_first=round(sum(first_right.values()) / count, 4) if count else None,
            accuracy_second=round(sum(second_right.values()) / count, 4) if count else None,
            pairing=getattr(pairing, "strategy_name", type(pairing).__name__),
            test="exact_mcnemar",
        )


def coverage_and_accuracy(verdicts: Verdicts) -> dict[str, float | int | None]:
    """How many pairs a scorer decides, and how often it is right on those."""
    total = len(verdicts)
    decided = [verdict for verdict in verdicts.values() if verdict is not None]
    right = sum(1 for verdict in decided if verdict)
    return {
        "n_pairs": total,
        "n_decided": len(decided),
        "n_right": right,
        "coverage": len(decided) / total if total else None,
        "accuracy": right / len(decided) if decided else None,
    }
