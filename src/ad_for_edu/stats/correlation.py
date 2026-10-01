"""Rank correlation between two metrics, with a bootstrap interval."""

from __future__ import annotations

import random
from collections.abc import Sequence

from .intervals import Interval, percentile_bounds


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).statistic)


def spearman_with_interval(
    x: Sequence[float],
    y: Sequence[float],
    generator: random.Random,
    *,
    resamples: int = 2000,
    level: float = 0.95,
    quantile_rule: str = "symmetric",
) -> Interval:
    """Spearman's rho over paired units, with a percentile interval from resampling the units.

    The generator is passed in so that a matrix of correlations draws all its
    resamples from one stream and stays reproducible as a whole. A resample in
    which either variable is constant has no rank correlation and is skipped.
    """
    if len(x) != len(y):
        raise ValueError(f"paired values differ in length: {len(x)} and {len(y)}")
    size = len(x)
    method = (
        f"bootstrap over units, {resamples} resamples, "
        f"{level:.0%} percentile ({quantile_rule})"
    )
    if size < 3:
        return Interval(None, None, None, size, size, method)
    estimate = spearman(x, y)
    drawn: list[float] = []
    for _ in range(resamples):
        picks = [generator.randrange(size) for _ in range(size)]
        resampled_x = [x[i] for i in picks]
        resampled_y = [y[i] for i in picks]
        if len(set(resampled_x)) < 2 or len(set(resampled_y)) < 2:
            continue
        drawn.append(spearman(resampled_x, resampled_y))
    if not drawn:
        return Interval(estimate, None, None, size, size, method)
    drawn.sort()
    low, high = percentile_bounds(drawn, level, quantile_rule)
    return Interval(estimate, low, high, size, size, method)


def rank_of(scores: dict[str, float], name: str, *, highest_first: bool = True) -> int:
    """Position of `name` among `scores`, 1 being the best."""
    ordered = sorted(scores, key=lambda key: scores[key], reverse=highest_first)
    return ordered.index(name) + 1
