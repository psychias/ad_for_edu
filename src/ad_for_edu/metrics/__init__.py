"""The metrics of the evaluation: overlap with references, alignment with the
picture, a judge's stored answers, and compliance with the rule book."""

from . import alignment, answered, compliance, overlap  # noqa: F401  (registers the metrics)
from .alignment import TOKEN_LIMIT, ImageTextAlignment
from .answered import StoredRating, StoredRubric, read_answers
from .base import METRICS, Metric, ScoringItem, best_over_references, mean_of
from .compliance import ComplianceMetric
from .overlap import SIMILARITY_FLOOR, CharacterOverlap, EmbeddedSimilarity, rescale

__all__ = [
    "METRICS",
    "SIMILARITY_FLOOR",
    "TOKEN_LIMIT",
    "CharacterOverlap",
    "ComplianceMetric",
    "EmbeddedSimilarity",
    "ImageTextAlignment",
    "Metric",
    "ScoringItem",
    "StoredRating",
    "StoredRubric",
    "best_over_references",
    "mean_of",
    "read_answers",
    "rescale",
]
