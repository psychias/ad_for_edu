"""Factors that depend on a system's whole output for a lecture.

A description that repeats what the same system already said earlier in the
lecture adds nothing for the listener, however well it follows the rules. That
is a property of the sequence of descriptions, not of one description, so it is
computed here, outside the components, and multiplied into the score.

    novelty(a_i) = 1 - max over earlier a_j of the lecture of Jaccard(tokens(a_i), tokens(a_j))
    factor(a_i)  = min(1, novelty(a_i) / knee)

The first description of a lecture has novelty 1. A single description, or a pair
whose sides come from different systems, has no sequence; the factor is then not
available and must not be assumed to be 1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..core.ids import lecture_of
from ..core.registry import Registry
from ..core.text import jaccard, token_set
from .scorer import ScoreRow


@dataclass(frozen=True)
class SequenceItem:
    """One description of a system, placed in its lecture.

    `in_lecture` is the lecture, given rather than derived where a dataset knows it. It
    falls back to the part of the moment id before the separator, which is this package's
    own convention. An identifier from elsewhere has no separator and falls back to
    itself, which would make every moment its own lecture and leave nothing able to
    repeat, so a reader of such a dataset must pass the lecture; `data.fields` is how.
    """

    moment_id: str
    time: float
    text: str | None
    in_lecture: str = ""

    @property
    def lecture(self) -> str:
        return self.in_lecture or lecture_of(self.moment_id)


@dataclass(frozen=True)
class SequenceEffect:
    novelty: float
    factor: float


class SequenceFactor(ABC):
    """A multiplier per description, computed from the system's output sequence."""

    @abstractmethod
    def effects(self, sequence: Sequence[SequenceItem]) -> dict[str, SequenceEffect]:
        """The effect per moment id, for the descriptions of one system."""

    def apply(self, rows: dict[str, ScoreRow], sequence: Sequence[SequenceItem]) -> None:
        """Multiply the factor into every scored row, in place.

        Only scored descriptions take part in the ordering: an empty description
        says nothing that a later one could repeat.
        """
        scored = [item for item in sequence if rows[item.moment_id].scored]
        effects = self.effects(scored)
        for item in scored:
            row, effect = rows[item.moment_id], effects[item.moment_id]
            row.novelty = effect.novelty
            row.novelty_factor = effect.factor
            row.overall = float(row.overall) * effect.factor


SEQUENCE_FACTORS: Registry[SequenceFactor] = Registry("sequence factor", SequenceFactor)


def within_lecture_novelty(sequence: Iterable[SequenceItem]) -> dict[str, float]:
    """Novelty per moment id.

    Inside a lecture the descriptions are ordered by time, then by moment id, so
    the order is total. Each description is compared with every earlier one.
    """
    by_lecture: dict[str, list[SequenceItem]] = {}
    for item in sequence:
        by_lecture.setdefault(item.lecture, []).append(item)
    novelty: dict[str, float] = {}
    for items in by_lecture.values():
        items.sort(key=lambda item: (float(item.time), item.moment_id))
        earlier: list[frozenset[str]] = []
        for item in items:
            tokens = token_set(item.text)
            if earlier:
                novelty[item.moment_id] = 1.0 - max(jaccard(tokens, seen) for seen in earlier)
            else:
                novelty[item.moment_id] = 1.0
            earlier.append(tokens)
    return novelty


@SEQUENCE_FACTORS.register("novelty_knee")
class NoveltyKnee(SequenceFactor):
    """Full credit at or above the knee, a proportional cut below it."""

    def __init__(self, knee: float = 0.4) -> None:
        if not 0 < knee <= 1:
            raise ValueError(f"knee must be in (0, 1], got {knee}")
        self.knee = knee

    def factor(self, novelty: float) -> float:
        return min(1.0, novelty / self.knee)

    def effects(self, sequence: Sequence[SequenceItem]) -> dict[str, SequenceEffect]:
        return {
            moment: SequenceEffect(value, self.factor(value))
            for moment, value in within_lecture_novelty(sequence).items()
        }
