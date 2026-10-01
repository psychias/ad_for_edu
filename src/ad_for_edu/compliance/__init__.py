"""The rule-compliance metric.

Six components, one per rule category, each in [0, 1]. The score of a description
is the unweighted mean of the components that apply to its moment. For a system,
that mean is multiplied by a factor that falls when the system repeats itself
across the moments of a lecture.

The metric has two uses, and they are not the same thing. Scored over a system's
whole output, with the novelty factor applied, it **ranks** systems. Scored on one
description it says **which rule that description broke** — the diagnostic use, which
is what `rules.py` and `report.py` are for. Neither reading is a claim that a
compliant description is a good one.
"""

from .build import (
    ComplianceSettings,
    build_scorer,
    build_sequence_factor,
    load_compliance_settings,
)
from .components.base import COMPONENTS, Component
from .contexts import CONTEXT_SOURCES, ContextSource
from .report import (
    mean_and_median,
    render,
    render_rule_diagnostic,
    rule_diagnostic,
    score_predictions,
    summarise,
    system_score,
)
from .rules import (
    CHECKS_BY_COMPONENT,
    SEQUENCE_RULES,
    WITHOUT_A_COMPONENT,
    RuleMap,
)
from .scorer import ComplianceScorer, ScoreRow
from .sequence import SEQUENCE_FACTORS, NoveltyKnee, SequenceFactor, SequenceItem
from .tiers import Tier

__all__ = [
    "CHECKS_BY_COMPONENT",
    "COMPONENTS",
    "CONTEXT_SOURCES",
    "SEQUENCE_FACTORS",
    "SEQUENCE_RULES",
    "WITHOUT_A_COMPONENT",
    "ComplianceScorer",
    "ComplianceSettings",
    "Component",
    "ContextSource",
    "NoveltyKnee",
    "RuleMap",
    "ScoreRow",
    "SequenceFactor",
    "SequenceItem",
    "Tier",
    "build_scorer",
    "build_sequence_factor",
    "load_compliance_settings",
    "mean_and_median",
    "render",
    "render_rule_diagnostic",
    "rule_diagnostic",
    "score_predictions",
    "summarise",
    "system_score",
]
