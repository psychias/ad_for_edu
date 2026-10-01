"""Metrics that compare a description with the reference descriptions of its moment.

The character-level score counts how much of the wording two texts share. The
meaning-level score embeds both texts and compares them.

The meaning-level score has a narrow useful range: two unrelated sentences
already score about 0.76, because any two English sentences share structure. So
the value is also offered on a rescaled axis, where that floor is zero:

    rescaled = (raw - floor) / (1 - floor)

The rescaling cannot change an order, since it is the same increasing map for
every value. It changes every absolute number, which is why both are reported:
a threshold or a materiality band set on one axis means nothing on the other. A
value below the floor stays negative rather than being clipped, because a
description that resembles its references less than an unrelated sentence would
is genuinely worse than that.
"""

from __future__ import annotations

from collections.abc import Sequence

from .base import METRICS, Metric, ScoringItem, best_over_references

#: What two unrelated English sentences score on the meaning-level metric with the
#: model below. Measured by its authors, not by this project.
SIMILARITY_FLOOR = 0.7568846


def rescale(value: float, floor: float = SIMILARITY_FLOOR) -> float:
    """Put `value` on the axis whose zero is the score of two unrelated sentences."""
    return (value - floor) / (1.0 - floor)


@METRICS.register("chrf")
class CharacterOverlap(Metric):
    """Character n-gram overlap with the references, in [0, 1]."""

    needs_references = True

    def __init__(self) -> None:
        self._scorer = None

    def _load(self):
        if self._scorer is None:
            from sacrebleu.metrics import CHRF

            self._scorer = CHRF()
        return self._scorer

    def one(self, candidate: str, reference: str) -> float:
        return self._load().sentence_score(candidate, [reference]).score / 100.0

    def score(self, items: Sequence[ScoringItem]) -> list[float | None]:
        return best_over_references(
            items,
            lambda candidates, references: [
                self.one(candidate, reference)
                for candidate, reference in zip(candidates, references, strict=True)
            ],
        )


@METRICS.register("bertscore")
class EmbeddedSimilarity(Metric):
    """Similarity of contextual embeddings with the references.

    `rescaled` reports the value on the axis whose zero is the score of two
    unrelated sentences. Both axes order systems identically.
    """

    needs_references = True

    def __init__(
        self,
        model: str = "roberta-large",
        language: str = "en",
        batch_size: int = 32,
        rescaled: bool = False,
    ) -> None:
        self.model = model
        self.language = language
        self.batch_size = batch_size
        self.rescaled = rescaled
        self._scorer = None

    def _load(self):
        if self._scorer is None:
            from bert_score import BERTScorer

            # The model is asked for raw values; the rescaling is applied here, so
            # that both axes come from one call and cannot disagree.
            self._scorer = BERTScorer(
                lang=self.language, model_type=self.model, rescale_with_baseline=False
            )
        return self._scorer

    def many(self, candidates: list[str], references: list[str]) -> list[float]:
        if not candidates:
            return []
        _precision, _recall, f1 = self._load().score(
            candidates, references, batch_size=self.batch_size
        )
        values = [float(value) for value in f1]
        return [rescale(value) for value in values] if self.rescaled else values

    def score(self, items: Sequence[ScoringItem]) -> list[float | None]:
        return best_over_references(items, self.many)
