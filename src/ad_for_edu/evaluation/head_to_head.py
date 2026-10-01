"""Comparing two systems directly on the same moments.

A judge is shown the two descriptions of one moment, in both presentation orders,
and the win rate is the share of the decided moments that one system won. A moment
counts as decided when a system won it in both orders; a moment on which the judge
named a different system each way round is not decided, and is reported rather
than dropped.

The interval is over lectures, not over moments. The moments of one lecture share
slides, a lecturer and a subject, so an interval that treats them as independent
observations comes out too narrow.

The direction is stated by the name: a win rate of one system over another is the
share that system won, and swapping the names swaps the number.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..core.errors import ContractError
from ..core.ids import lecture_of
from ..pairs.ordering import ORDERS, OrderCombination, Outcome
from ..stats.intervals import ClusterBootstrap, Interval, Observation


@dataclass(frozen=True)
class HeadToHead:
    """The result of comparing two systems."""

    first: str
    second: str
    judge: str
    won: int
    lost: int
    undecided: int
    interval: Interval

    @property
    def decided(self) -> int:
        return self.won + self.lost

    @property
    def win_rate(self) -> float | None:
        return self.won / self.decided if self.decided else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "first": self.first,
            "second": self.second,
            "judge": self.judge,
            "won": self.won,
            "lost": self.lost,
            "undecided": self.undecided,
            "decided": self.decided,
            "win_rate": round(self.win_rate, 4) if self.win_rate is not None else None,
            "interval": self.interval.as_dict(),
        }


def draw_moments(
    moments: Sequence[str], count: int, seed: int
) -> list[str]:
    """A reproducible draw of moments to compare on."""
    if count > len(moments):
        raise ContractError(
            f"{count} moments were asked for and only {len(moments)} are available"
        )
    generator = random.Random(seed)
    drawn = list(moments)
    generator.shuffle(drawn)
    return sorted(drawn[:count])


def outcomes(
    answers: Mapping[str, Mapping[int, str | None]],
    combination: OrderCombination,
) -> dict[str, Outcome]:
    """What the two orders said about each moment.

    A moment with only one order answered is undecided: the second order is what
    tells a preference for a description from a preference for a position.
    """
    settled = {}
    for moment, per_order in answers.items():
        missing = [order for order in ORDERS if per_order.get(order) is None]
        if len(missing) == len(ORDERS):
            settled[moment] = Outcome.UNDECIDED
            continue
        settled[moment] = combination.combine(per_order.get(0), per_order.get(1))
    return settled


def win_rate(
    settled: Mapping[str, Outcome],
    *,
    first: str,
    second: str,
    judge: str,
    first_is_side: str = "a",
    resamples: int = 5000,
    seed: int = 0,
) -> HeadToHead:
    """The share of decided moments the first system won, with a lecture-clustered interval."""
    if first_is_side not in ("a", "b"):
        raise ContractError(f"the first system is side a or b, got {first_is_side!r}")
    wins = Outcome(first_is_side)
    observations = []
    won = lost = undecided = 0
    for moment, outcome in settled.items():
        if not outcome.decided:
            undecided += 1
            continue
        if outcome is wins:
            won += 1
        else:
            lost += 1
        observations.append(Observation(lecture_of(moment), 1.0 if outcome is wins else 0.0))
    interval = ClusterBootstrap(resamples=resamples, seed=seed).interval(observations)
    return HeadToHead(first, second, judge, won, lost, undecided, interval)


def agreement_between(
    first: Mapping[str, Outcome], second: Mapping[str, Outcome]
) -> dict[str, Any]:
    """How far two judges agree on the moments both decided."""
    shared = [
        moment
        for moment in set(first) & set(second)
        if first[moment].decided and second[moment].decided
    ]
    same = sum(1 for moment in shared if first[moment] is second[moment])
    return {
        "both_decided": len(shared),
        "agreed": same,
        "share_agreed": round(same / len(shared), 4) if shared else None,
    }
