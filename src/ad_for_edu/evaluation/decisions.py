"""Deciding which side of a pair a scorer prefers.

A scorer decides a pair when it gives the two sides different values. Equal values
are a tie: the scorer did not decide, and counting a tie as a failure or dropping
it are different measurements, so both are reported.

Coverage and accuracy are reported together and never separately. A scorer that
decides few pairs and is right on them looks better than one that decides all of
them, purely by choosing an easier denominator; the pair of numbers is the
measurement, either number alone is not.

A cascade puts scorers in order: the first that decides a pair decides it. It
raises coverage at whatever the later scorers are worth, so the stage that decided
each pair is recorded.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..core.registry import Registry
from ..data.schema import SIDES


@dataclass(frozen=True)
class PairScores:
    """What one scorer gave the two sides of a pair."""

    a: float | None
    b: float | None

    @property
    def decided(self) -> bool:
        return self.a is not None and self.b is not None and self.a != self.b

    @property
    def preferred(self) -> str | None:
        if not self.decided:
            return None
        return "a" if self.a > self.b else "b"

    @property
    def tied(self) -> bool:
        return self.a is not None and self.b is not None and self.a == self.b


@dataclass(frozen=True)
class PairDecision:
    """Which side a scorer preferred, and which scorer that was."""

    pair_id: str
    preferred: str | None
    by: str = ""

    @property
    def decided(self) -> bool:
        return self.preferred in SIDES


class PairDecisionRule(ABC):
    """Decides pairs from the scores of one or more scorers."""

    @abstractmethod
    def decide(self, pair_id: str, scores: Mapping[str, PairScores]) -> PairDecision: ...


PAIR_DECISION_RULES: Registry[PairDecisionRule] = Registry("pair decision rule", PairDecisionRule)


@PAIR_DECISION_RULES.register("single_scorer")
class SingleScorer(PairDecisionRule):
    """One scorer decides, or the pair is undecided."""

    def __init__(self, scorer: str) -> None:
        self.scorer = scorer

    def decide(self, pair_id: str, scores: Mapping[str, PairScores]) -> PairDecision:
        found = scores.get(self.scorer)
        if found is None or not found.decided:
            return PairDecision(pair_id, None, self.scorer)
        return PairDecision(pair_id, found.preferred, self.scorer)


@PAIR_DECISION_RULES.register("cascade")
class Cascade(PairDecisionRule):
    """Scorers in order: the first that decides the pair decides it."""

    def __init__(self, scorers: Sequence[str]) -> None:
        if not scorers:
            raise ValueError("a cascade needs at least one scorer")
        self.scorers = tuple(scorers)

    def decide(self, pair_id: str, scores: Mapping[str, PairScores]) -> PairDecision:
        for scorer in self.scorers:
            found = scores.get(scorer)
            if found is not None and found.decided:
                return PairDecision(pair_id, found.preferred, scorer)
        return PairDecision(pair_id, None, "")


def verdicts_against(
    decisions: Sequence[PairDecision], correct: Mapping[str, str]
) -> dict[str, bool | None]:
    """For each pair: whether the scorer preferred the right side, or None if undecided.

    `correct` names the side that keeps the rule. A pair whose right side is unknown
    is left out: it cannot say whether a scorer was right.
    """
    return {
        decision.pair_id: (decision.preferred == correct[decision.pair_id])
        if decision.decided
        else None
        for decision in decisions
        if decision.pair_id in correct
    }


@dataclass
class CoverageAndAccuracy:
    """How many pairs a scorer decided, and how often it was right on those."""

    pairs: int
    decided: int
    right: int
    ties: int = 0
    by_stage: Mapping[str, int] = field(default_factory=dict)

    @property
    def coverage(self) -> float | None:
        return self.decided / self.pairs if self.pairs else None

    @property
    def accuracy(self) -> float | None:
        return self.right / self.decided if self.decided else None

    @property
    def accuracy_with_ties_as_failures(self) -> float | None:
        """Accuracy over every pair, counting a tie against the scorer.

        Reported beside the other, because a scorer that decides half the pairs and
        is right on all of them reads very differently under the two.
        """
        return self.right / self.pairs if self.pairs else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "pairs": self.pairs,
            "decided": self.decided,
            "right": self.right,
            "ties": self.ties,
            "coverage": round(self.coverage, 4) if self.coverage is not None else None,
            "accuracy": round(self.accuracy, 4) if self.accuracy is not None else None,
            "accuracy_ties_as_failures": (
                round(self.accuracy_with_ties_as_failures, 4)
                if self.accuracy_with_ties_as_failures is not None
                else None
            ),
            "decided_by": dict(self.by_stage),
        }


def measure(
    decisions: Sequence[PairDecision], correct: Mapping[str, str]
) -> CoverageAndAccuracy:
    """Coverage and accuracy of a rule against the sides that keep the rule."""
    relevant = [decision for decision in decisions if decision.pair_id in correct]
    decided = [decision for decision in relevant if decision.decided]
    right = sum(1 for decision in decided if decision.preferred == correct[decision.pair_id])
    by_stage: dict[str, int] = {}
    for decision in decided:
        by_stage[decision.by] = by_stage.get(decision.by, 0) + 1
    return CoverageAndAccuracy(
        pairs=len(relevant),
        decided=len(decided),
        right=right,
        ties=len(relevant) - len(decided),
        by_stage=by_stage,
    )
