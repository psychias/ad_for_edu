"""The rule-compliance metric, as one of the metrics of a table.

The compliance score of a system is a property of its whole output for a lecture,
because a description that repeats an earlier one of the same system adds nothing.
So this metric is handed the scorer and, where the settings ask for it, the factor
that reads the sequence; it scores every item of one system together.

Items are expected in the order the system produced them, each carrying the time
of its moment, so that "earlier in this lecture" is defined.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..compliance.scorer import ComplianceScorer
from ..compliance.sequence import SequenceFactor, SequenceItem
from ..core.errors import ContractError
from ..data.schema import MomentContext
from .base import METRICS, Metric, ScoringItem


@METRICS.register("compliance")
class ComplianceMetric(Metric):
    """Compliance with the rule book, in one of its offline modes."""

    def __init__(
        self,
        scorer: ComplianceScorer,
        contexts: dict[str, MomentContext],
        sequence_factor: SequenceFactor | None = None,
    ) -> None:
        self.scorer = scorer
        self.contexts = dict(contexts)
        self.sequence_factor = sequence_factor

    @property
    def mode(self) -> str:
        return self.scorer.name

    def score(self, items: Sequence[ScoringItem]) -> list[float | None]:
        rows = {}
        for item in items:
            context = self.contexts.get(item.moment_id)
            if context is None:
                raise ContractError(
                    f"moment {item.moment_id!r} has no context, so its compliance "
                    "cannot be scored"
                )
            rows[item.moment_id] = self.scorer.score(item.text, context)
        if self.sequence_factor is not None:
            sequence = [
                SequenceItem(item.moment_id, float(item.fields.get("time", 0.0)), item.text)
                for item in items
            ]
            self.sequence_factor.apply(rows, sequence)
        return [rows[item.moment_id].overall for item in items]
