"""Which side of a pair is preferred, and where that preference comes from.

A pair is ordered in one of two ways.

**By construction.** A controlled pair knows its own direction: one side was built
to break a rule. Nothing has to judge it.

**By a judge, in both presentation orders.** A judge shown the same pair twice,
once each way round, may answer differently. The two answers are combined into one
outcome, and which rule combines them decides how many pairs end up ordered:

    agree_or_undecided   both orders must name the same side. A judge that answers
                         differently each way has not told you which side is better;
                         it has told you it prefers a position.
    agree_or_tie         as above, but a disagreement is recorded as a tie rather
                         than as no answer.

A pair the judge could not order is kept and marked, not dropped: it is evidence
about the judge, and a set that quietly drops them describes a population that
excludes the hard cases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..core.registry import Registry
from ..data.schema import SIDES, TIE
from ..llm.replies import parse_choice

#: The two orders a pair is shown in. In the first, the side recorded as `a` is
#: shown first; in the second, the side recorded as `b` is.
ORDERS: tuple[int, int] = (0, 1)

#: Where the direction of a pair comes from.
BY_CONSTRUCTION = "construction"
BY_JUDGE = "judge"


class Outcome(str, Enum):
    """What the two orders together said."""

    FIRST = "a"
    SECOND = "b"
    TIE = "tie"
    UNDECIDED = "undecided"

    @property
    def decided(self) -> bool:
        return self in (Outcome.FIRST, Outcome.SECOND)


def side_shown_first(order: int) -> str:
    """Which recorded side is presented first in this order."""
    if order not in ORDERS:
        raise ValueError(f"an order is one of {ORDERS}, got {order}")
    return SIDES[order]


def winner_of(answer: str | None, order: int) -> str | None:
    """Which recorded side an answer names, whatever order it was shown in.

    The judge answers "the first" or "the second"; which recorded side that is
    depends on the order. Resolving it here is what makes the two orders comparable.
    """
    if answer is None:
        return None
    if answer == TIE:
        return TIE
    if answer not in ("1", "2"):
        return None
    first = answer == "1"
    if order == 0:
        return SIDES[0] if first else SIDES[1]
    return SIDES[1] if first else SIDES[0]


def read_answer(reply: str | None, order: int) -> str | None:
    """The recorded side a reply names, or None when the reply named none."""
    return winner_of(parse_choice(reply), order)


class OrderCombination(ABC):
    """Combines what the two presentation orders said."""

    @abstractmethod
    def combine(self, first: str | None, second: str | None) -> Outcome: ...


ORDER_COMBINATIONS: Registry[OrderCombination] = Registry("order combination", OrderCombination)


@ORDER_COMBINATIONS.register("agree_or_undecided")
class AgreeOrUndecided(OrderCombination):
    """Both orders must name the same side; anything else leaves the pair unordered."""

    def combine(self, first: str | None, second: str | None) -> Outcome:
        if first in SIDES and first == second:
            return Outcome(first)
        if first == TIE and second == TIE:
            return Outcome.TIE
        return Outcome.UNDECIDED


@ORDER_COMBINATIONS.register("agree_or_tie")
class AgreeOrTie(OrderCombination):
    """Both orders must name the same side; a disagreement is a tie."""

    def combine(self, first: str | None, second: str | None) -> Outcome:
        if first in SIDES and first == second:
            return Outcome(first)
        if first is None or second is None:
            return Outcome.UNDECIDED
        return Outcome.TIE


@dataclass(frozen=True)
class Ordered:
    """A pair's direction, and where it came from."""

    pair_id: str
    outcome: Outcome
    source: str
    answers: tuple[str | None, str | None] = (None, None)

    @property
    def chosen(self) -> str | None:
        return self.outcome.value if self.outcome.decided else None

    @property
    def flipped(self) -> bool:
        """Whether the judge named a different side in each order."""
        first, second = self.answers
        return (
            self.source == BY_JUDGE
            and first in SIDES
            and second in SIDES
            and first != second
        )


def order_by_construction(pair: Mapping[str, Any], compliant_side: str = "a") -> Ordered:
    """The direction of a pair that was built with one."""
    if compliant_side not in SIDES:
        raise ValueError(f"the compliant side is one of {SIDES}, got {compliant_side!r}")
    return Ordered(str(pair["pair_id"]), Outcome(compliant_side), BY_CONSTRUCTION)


def order_by_judge(
    pair_id: str,
    answers: Mapping[int, str | None],
    combination: OrderCombination,
) -> Ordered:
    """The direction of a pair from what the judge said in each order."""
    first, second = answers.get(0), answers.get(1)
    return Ordered(pair_id, combination.combine(first, second), BY_JUDGE, (first, second))


def summarise(ordered: Sequence[Ordered]) -> dict[str, Any]:
    """How the pairs came out: ordered, tied, unordered, and how many flipped."""
    decided = [entry for entry in ordered if entry.outcome.decided]
    total = len(ordered) or 1
    return {
        "pairs": len(ordered),
        "ordered": len(decided),
        "tied": sum(1 for entry in ordered if entry.outcome is Outcome.TIE),
        "unordered": sum(1 for entry in ordered if entry.outcome is Outcome.UNDECIDED),
        "flipped": sum(1 for entry in ordered if entry.flipped),
        "share_ordered": round(len(decided) / total, 4),
    }
