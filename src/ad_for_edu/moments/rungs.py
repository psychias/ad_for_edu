"""How a description reaches the listener, and how many words it may have.

A description is delivered on a ladder. The first rung speaks it inside the gap
after the moment. The second compresses it to fit that gap. The third delivers it
late, on a track of the listener's own, so long as it is not later than a cap. The
fourth is a placeholder of a few words. The fifth is not to deliver it at all,
which is never an answer a writer gives: staying silent is the answer instead.

The word budget is what a rung allows, not what the gap allows. Most moments offer
no usable gap at all, so budgeting to the gap would produce nothing but
placeholders.

Each rung is tested against its own budget. A rung that compresses to fifteen
words must be tested at fifteen: testing every rung at the same number makes the
compressing rung impossible to satisfy, whatever the gap, and every moment then
falls through to the rung that delivers late.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from ..core.registry import Registry

#: Words a rung allows.
RUNG_WORDS: Mapping[int, int] = {1: 25, 2: 15, 3: 25, 4: 5}
#: Words a minute, for turning a number of words into a length of time.
WORDS_PER_MINUTE = 140.0
#: How late the third rung may deliver.
LAG_CAP_SECONDS = 20.0
#: The rungs a writer may answer with.
REACHABLE: tuple[int, ...] = (1, 2, 3, 4)


def spoken_seconds(words: int, words_per_minute: float = WORDS_PER_MINUTE) -> float:
    return words / words_per_minute * 60.0


def fits(
    rung: int,
    words: int,
    pause: float,
    *,
    words_per_minute: float = WORDS_PER_MINUTE,
    lag_cap: float = LAG_CAP_SECONDS,
) -> bool:
    """Whether a description of `words` words can be delivered at `rung`."""
    length = spoken_seconds(words, words_per_minute)
    if rung == 1:
        return length <= pause
    if rung == 2:
        return length <= pause and words <= RUNG_WORDS[2]
    if rung == 3:
        return length <= pause + lag_cap
    if rung == 4:
        return words <= RUNG_WORDS[4]
    if rung == 5:
        return True
    return False


def deliverable_words(
    rung: int,
    pause: float,
    *,
    words_per_minute: float = WORDS_PER_MINUTE,
    lag_cap: float = LAG_CAP_SECONDS,
) -> float:
    """The most words `rung` can deliver given the gap after the moment."""
    if rung == 1:
        return pause / 60.0 * words_per_minute
    if rung == 3:
        return (pause + lag_cap) / 60.0 * words_per_minute
    return float(RUNG_WORDS.get(rung, RUNG_WORDS[4]))


class RungPolicy(ABC):
    """Says which rung a moment can reach."""

    @abstractmethod
    def reachable(self, pause: float) -> int: ...


RUNG_POLICIES: Registry[RungPolicy] = Registry("rung policy", RungPolicy)


@RUNG_POLICIES.register("rung_budgets")
class RungBudgets(RungPolicy):
    """The lowest rung whose own word budget fits the gap.

    Lower is better: the first rung speaks inside the gap, the fourth is a
    placeholder.
    """

    def __init__(
        self,
        budgets: Mapping[int, int] | None = None,
        words_per_minute: float = WORDS_PER_MINUTE,
        lag_cap: float = LAG_CAP_SECONDS,
    ) -> None:
        self.budgets = dict(budgets or RUNG_WORDS)
        self.words_per_minute = words_per_minute
        self.lag_cap = lag_cap

    def reachable(self, pause: float) -> int:
        for rung in REACHABLE:
            words = self.budgets[rung]
            if fits(
                rung,
                words,
                pause,
                words_per_minute=self.words_per_minute,
                lag_cap=self.lag_cap,
            ):
                return rung
        return REACHABLE[-1]


@RUNG_POLICIES.register("uniform_budget")
class UniformBudget(RungPolicy):
    """The lowest rung that fits one budget, the same for every rung.

    The compressing rung cannot be reached under this policy when the budget is
    above its own, since it is tested at a number of words it does not allow.
    """

    def __init__(
        self,
        words: int = 25,
        words_per_minute: float = WORDS_PER_MINUTE,
        lag_cap: float = LAG_CAP_SECONDS,
    ) -> None:
        self.words = words
        self.words_per_minute = words_per_minute
        self.lag_cap = lag_cap

    def reachable(self, pause: float) -> int:
        for rung in REACHABLE:
            if fits(
                rung,
                self.words,
                pause,
                words_per_minute=self.words_per_minute,
                lag_cap=self.lag_cap,
            ):
                return rung
        return REACHABLE[-1]
