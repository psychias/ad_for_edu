"""Metrics whose values a judge already gave.

A judge is asked once and its answers are stored. A metric of this kind reads
that store and never calls anything: a column of a table can be rebuilt without
spending, and the value of a described moment cannot change between two readings
of the same table.

An item the store does not cover scores None. Filling it in would mean either
inventing a value or spending without being asked.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ..core.errors import ContractError, MissingSourceError
from ..core.io import iter_jsonl
from .base import METRICS, Metric, ScoringItem


def read_answers(
    path: str | Path,
    *,
    key: str = "moment_id",
    value: str = "score",
    system: str | None = None,
) -> dict[str, float]:
    """Stored answers as `{moment: value}`, from a file of rows.

    A row without a value is a call that failed and is left out. A moment answered
    twice is an error: which of the two a table used would otherwise depend on the
    order of the rows.
    """
    source = Path(path)
    if not source.is_file():
        raise MissingSourceError(f"no stored answers at {source}")
    answers: dict[str, float] = {}
    for row in iter_jsonl(source):
        if row.get("error") or row.get(value) is None:
            continue
        if system is not None and row.get("system") != system:
            continue
        moment = str(row.get(key) or "")
        if not moment:
            continue
        if moment in answers:
            raise ContractError(f"{source.name}: moment {moment!r} is answered more than once")
        answers[moment] = float(row[value])
    return answers


@METRICS.register("reference_rating")
class StoredRating(Metric):
    """A judge's rating of each description against the references of its moment."""

    needs_references = True
    reads_answers = True

    def __init__(self, answers: Mapping[str, float], scale: tuple[float, float] = (1.0, 5.0)):
        self.answers = dict(answers)
        self.low, self.high = scale
        if self.high <= self.low:
            raise ValueError(f"the scale must rise, got {scale}")

    def normalised(self, value: float) -> float:
        return (value - self.low) / (self.high - self.low)

    def score(self, items: Sequence[ScoringItem]) -> list[float | None]:
        found = []
        for item in items:
            value = self.answers.get(item.moment_id)
            found.append(self.normalised(value) if value is not None else None)
        return found


@METRICS.register("rubric")
class StoredRubric(Metric):
    """A judge's overall grade of each description against the rule book."""

    reads_answers = True

    def __init__(self, answers: Mapping[str, float], scale: tuple[float, float] = (1.0, 10.0)):
        self.answers = dict(answers)
        self.low, self.high = scale
        if self.high <= self.low:
            raise ValueError(f"the scale must rise, got {scale}")

    def score(self, items: Sequence[ScoringItem]) -> list[float | None]:
        return [self.answers.get(item.moment_id) for item in items]
