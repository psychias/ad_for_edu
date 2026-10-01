"""Turning candidate descriptions into pairs, and drawing sets of pairs.

**Pairing.** Of all the ways to pair the candidates of one moment, the ones kept
are those between different writers first, then the ones that differ most, and no
more than a stated number per moment. Two descriptions too alike to tell apart are
not a pair: a rater asked to choose between them is being asked nothing. No single
description is paired against everything, so the pairs of a moment spread over the
descriptions rather than radiating from one.

The candidates are shuffled before they are ranked, so that two pairings equally
far apart are not decided by which writer happens to sort first.

**Drawing a set.** A draw fills a stated number of pairs from each kind, and within
a kind spreads over the categories and the lectures, taking the moments used least
so far. No moment supplies more than a stated number of pairs, because pairs on one
moment are not independent of one another.

Every draw takes a seed and is reproducible from it.
"""

from __future__ import annotations

import itertools
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import ContractError

#: How far apart two descriptions must be to be worth pairing, as a distance in [0, 100].
MINIMUM_DISTANCE = 12.0
#: How many pairs one moment may supply.
MAXIMUM_PER_MOMENT = 4
#: How often one description may appear across the pairs of its moment.
MAXIMUM_PER_DESCRIPTION = 2


@dataclass(frozen=True)
class Offered:
    """One description offered for pairing, and who wrote it."""

    text: str
    writer: str
    focus: str | None = None
    rung: int | None = None


@dataclass(frozen=True)
class Pairing:
    """Two descriptions of one moment, and how far apart they are."""

    first: Offered
    second: Offered
    distance: float
    across_writers: bool


def distance_between(first: str, second: str, overlap: Callable[[str, str], float]) -> float:
    """How far apart two descriptions are, on a scale of 0 to 100.

    Symmetrised, so the distance does not depend on which side is given first.
    """
    return 100.0 - 50.0 * (overlap(first, second) + overlap(second, first))


def pairs_of_moment(
    candidates: Sequence[Offered],
    overlap: Callable[[str, str], float],
    generator: random.Random,
    *,
    cap: int = MAXIMUM_PER_MOMENT,
    minimum_distance: float = MINIMUM_DISTANCE,
    per_description: int = MAXIMUM_PER_DESCRIPTION,
) -> list[Pairing]:
    """The pairs to keep from the candidates of one moment."""
    shuffled = list(candidates)
    generator.shuffle(shuffled)
    ranked: list[Pairing] = []
    for first, second in itertools.combinations(shuffled, 2):
        apart = distance_between(first.text, second.text, overlap)
        if apart < minimum_distance:
            continue
        ranked.append(Pairing(first, second, apart, first.writer != second.writer))
    # Across writers first, then furthest apart.
    ranked.sort(key=lambda pairing: (not pairing.across_writers, -pairing.distance))
    kept: list[Pairing] = []
    used: Counter[str] = Counter()
    for pairing in ranked:
        if len(kept) >= cap:
            break
        if used[pairing.first.text] >= per_description:
            continue
        if used[pairing.second.text] >= per_description:
            continue
        used[pairing.first.text] += 1
        used[pairing.second.text] += 1
        kept.append(pairing)
    return kept


@dataclass(frozen=True)
class DrawPlan:
    """How many pairs of each kind a draw takes, and how it spreads them."""

    #: Pairs per kind, such as how many controlled and how many natural.
    per_kind: Mapping[str, int]
    #: Pairs per category, where a category has its own number.
    per_category: Mapping[str, int] = field(default_factory=dict)
    #: The number for a category not named above.
    category_default: int = 0
    maximum_per_moment: int = 2
    seed: int = 0

    @property
    def total(self) -> int:
        return sum(self.per_kind.values())


def take(
    candidates: Sequence[Mapping[str, Any]],
    wanted: int,
    generator: random.Random,
    used: Counter,
    *,
    group_of: Callable[[Mapping[str, Any]], Any] = lambda row: None,
    maximum_per_moment: int = 2,
) -> list[Mapping[str, Any]]:
    """`wanted` pairs, spread evenly over the groups, taking the least-used moments.

    Within a group the order is shuffled and then sorted by how often that moment has
    been taken, so the moments carrying fewest pairs are taken first.
    """
    grouped: dict[Any, list[Mapping[str, Any]]] = {}
    for row in candidates:
        grouped.setdefault(group_of(row), []).append(row)
    for rows in grouped.values():
        generator.shuffle(rows)
        rows.sort(key=lambda row: used[row["moment_id"]])
    order = sorted(grouped, key=lambda name: (name is None, name))
    taken: list[Mapping[str, Any]] = []
    while len(taken) < wanted and any(grouped[name] for name in order):
        for name in order:
            while grouped[name] and len(taken) < wanted:
                row = grouped[name].pop(0)
                if used[row["moment_id"]] >= maximum_per_moment:
                    continue
                used[row["moment_id"]] += 1
                taken.append(row)
                break
    return taken


def draw(
    by_kind: Mapping[str, Sequence[Mapping[str, Any]]],
    plan: DrawPlan,
    *,
    category_of: Callable[[Mapping[str, Any]], Any] = lambda row: row.get("axis"),
    lecture_of: Callable[[Mapping[str, Any]], Any] = lambda row: row.get("lecture"),
) -> dict[str, list[Mapping[str, Any]]]:
    """A set of pairs, by kind, reproducible from the seed of the plan.

    A kind with its own numbers per category is filled category by category; the
    rest is spread over the lectures.
    """
    generator = random.Random(plan.seed)
    used: Counter = Counter()
    drawn: dict[str, list[Mapping[str, Any]]] = {}
    for kind, wanted in plan.per_kind.items():
        available = list(by_kind.get(kind, ()))
        if plan.per_category and kind == "controlled":
            taken: list[Mapping[str, Any]] = []
            categories = sorted({category_of(row) for row in available if category_of(row)})
            for category in categories:
                number = plan.per_category.get(category, plan.category_default)
                of_category = [row for row in available if category_of(row) == category]
                taken += take(
                    of_category,
                    number,
                    generator,
                    used,
                    group_of=lecture_of,
                    maximum_per_moment=plan.maximum_per_moment,
                )
            drawn[kind] = taken
        else:
            drawn[kind] = take(
                available,
                wanted,
                generator,
                used,
                group_of=lecture_of,
                maximum_per_moment=plan.maximum_per_moment,
            )
    return drawn


def hold_out_by_lecture(
    pairs: Sequence[Mapping[str, Any]],
    share: float,
    seed: int,
    *,
    lecture_of: Callable[[Mapping[str, Any]], str] = lambda row: str(row.get("lecture") or ""),
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split the pairs in two, by lecture, leaving no lecture on both sides.

    By lecture rather than by pair: two pairs of one moment, and two moments of one
    lecture, are not independent, so a split by pair would leave the held-out side
    describing material the other side also holds.
    """
    if not 0 < share < 1:
        raise ContractError(f"the share held out must be between 0 and 1, got {share}")
    lectures = sorted({lecture_of(row) for row in pairs})
    if len(lectures) < 2:
        raise ContractError("a split by lecture needs at least two lectures")
    generator = random.Random(seed)
    generator.shuffle(lectures)
    wanted = max(1, round(share * len(pairs)))
    held: set[str] = set()
    counted = 0
    per_lecture = Counter(lecture_of(row) for row in pairs)
    for lecture in lectures:
        if counted >= wanted:
            break
        held.add(lecture)
        counted += per_lecture[lecture]
    kept = [row for row in pairs if lecture_of(row) not in held]
    out = [row for row in pairs if lecture_of(row) in held]
    if not kept or not out:
        raise ContractError("the split left one side empty; adjust the share")
    return kept, out
