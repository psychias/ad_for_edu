"""The tables and comparisons of the evaluation."""

from .comparisons import (
    Comparison,
    compare,
    compare_all,
    correlations,
    rank_stability,
    ranks_of,
    render_correlations,
)
from .decisions import (
    PAIR_DECISION_RULES,
    Cascade,
    CoverageAndAccuracy,
    PairDecision,
    PairDecisionRule,
    PairScores,
    SingleScorer,
    measure,
    verdicts_against,
)
from .head_to_head import HeadToHead, agreement_between, draw_moments, outcomes, win_rate
from .systems_table import SystemRow, items_for, over_seeds, render, score_system

__all__ = [
    "PAIR_DECISION_RULES",
    "Cascade",
    "Comparison",
    "CoverageAndAccuracy",
    "HeadToHead",
    "PairDecision",
    "PairDecisionRule",
    "PairScores",
    "SingleScorer",
    "SystemRow",
    "agreement_between",
    "compare",
    "compare_all",
    "correlations",
    "draw_moments",
    "items_for",
    "measure",
    "outcomes",
    "over_seeds",
    "rank_stability",
    "ranks_of",
    "render",
    "render_correlations",
    "score_system",
    "verdicts_against",
    "win_rate",
]
