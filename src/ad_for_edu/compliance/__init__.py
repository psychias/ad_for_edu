"""The rule-compliance metric.

Six components, one per rule category, each in [0, 1]. The score of a description
is the unweighted mean of the components that apply to its moment. For a system,
that mean is multiplied by a factor that falls when the system repeats itself
across the moments of a lecture.
"""

from .build import (
    ComplianceSettings,
    build_scorer,
    build_sequence_factor,
    load_compliance_settings,
)
from .components.base import COMPONENTS, Component
from .contexts import CONTEXT_SOURCES, ContextSource
from .scorer import ComplianceScorer, ScoreRow
from .sequence import SEQUENCE_FACTORS, NoveltyKnee, SequenceFactor, SequenceItem
from .tiers import Tier

__all__ = [
    "COMPONENTS",
    "CONTEXT_SOURCES",
    "SEQUENCE_FACTORS",
    "ComplianceScorer",
    "ComplianceSettings",
    "Component",
    "ContextSource",
    "NoveltyKnee",
    "ScoreRow",
    "SequenceFactor",
    "SequenceItem",
    "Tier",
    "build_scorer",
    "build_sequence_factor",
    "load_compliance_settings",
]
