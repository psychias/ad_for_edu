"""Confidence intervals by resampling.

The moments of a lecture share a slide deck and a lecturer, so they are not
independent observations. The cluster bootstrap therefore resamples whole
lectures with replacement and carries each lecture's moments along; an interval
computed over moments as if they were independent comes out too narrow.

A percentile interval needs a rule for turning a quantile into a position in the
sorted resamples. Three rules are in use, and an interval states which one made it:

    inclusive   low = x[floor(q (B-1))]   high = x[floor((1-q) (B-1))]
    upper       low = x[floor(q B)]       high = x[floor((1-q) B)]
    symmetric   low = x[floor(q B)]       high = x[floor((1-q) B) - 1]
"""

from __future__ import annotations

import math
import random
import statistics
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

from ..core.registry import Registry

Statistic = Callable[[Sequence[float]], float]

QUANTILE_RULES = ("inclusive", "upper", "symmetric")


@dataclass(frozen=True)
class Observation:
    """One value and the cluster it belongs to."""

    cluster: str
    value: float


@dataclass(frozen=True)
class Interval:
    """A point estimate with its interval and what it rests on."""

    estimate: float | None
    low: float | None
    high: float | None
    n: int
    n_clusters: int
    method: str

    @property
    def defined(self) -> bool:
        return self.low is not None and self.high is not None

    def covers(self, value: float) -> bool:
        return self.defined and self.low <= value <= self.high

    def as_dict(self) -> dict[str, object]:
        return {
            "estimate": self.estimate,
            "low": self.low,
            "high": self.high,
            "n": self.n,
            "n_clusters": self.n_clusters,
            "method": self.method,
        }


def percentile_bounds(
    sorted_values: Sequence[float], level: float, rule: str
) -> tuple[float, float]:
    """The two ends of a percentile interval under one of the three position rules."""
    if rule not in QUANTILE_RULES:
        raise ValueError(f"unknown quantile rule {rule!r}; known: {QUANTILE_RULES}")
    if not sorted_values:
        raise ValueError("no resamples to take a percentile of")
    size = len(sorted_values)
    lower, upper = (1 - level) / 2, 1 - (1 - level) / 2
    if rule == "inclusive":
        low_at, high_at = int(lower * (size - 1)), int(upper * (size - 1))
    elif rule == "upper":
        low_at, high_at = int(lower * size), int(upper * size)
    else:
        low_at, high_at = int(lower * size), int(upper * size) - 1
    last = size - 1
    return sorted_values[min(max(low_at, 0), last)], sorted_values[min(max(high_at, 0), last)]


class IntervalEstimator(ABC):
    """Computes an interval for a statistic of some observations."""

    @abstractmethod
    def interval(
        self, observations: Sequence[Observation], statistic: Statistic = statistics.fmean
    ) -> Interval: ...


INTERVALS: Registry[IntervalEstimator] = Registry("interval estimator", IntervalEstimator)


class _Resampling(IntervalEstimator):
    def __init__(
        self,
        resamples: int = 2000,
        seed: int = 0,
        level: float = 0.95,
        quantile_rule: str = "inclusive",
    ) -> None:
        if resamples < 1:
            raise ValueError(f"resamples must be positive, got {resamples}")
        if not 0 < level < 1:
            raise ValueError(f"level must be in (0, 1), got {level}")
        if quantile_rule not in QUANTILE_RULES:
            raise ValueError(f"unknown quantile rule {quantile_rule!r}; known: {QUANTILE_RULES}")
        self.resamples = resamples
        self.seed = seed
        self.level = level
        self.quantile_rule = quantile_rule

    def _describe(self, kind: str) -> str:
        return (
            f"{kind}, {self.resamples} resamples, seed {self.seed}, "
            f"{self.level:.0%} percentile ({self.quantile_rule})"
        )


@INTERVALS.register("cluster_bootstrap")
class ClusterBootstrap(_Resampling):
    """Resamples clusters with replacement; a cluster brings all its observations."""

    def interval(
        self, observations: Sequence[Observation], statistic: Statistic = statistics.fmean
    ) -> Interval:
        by_cluster: dict[str, list[float]] = {}
        for observation in observations:
            by_cluster.setdefault(observation.cluster, []).append(float(observation.value))
        values = [value for members in by_cluster.values() for value in members]
        clusters = list(by_cluster)
        method = self._describe("cluster bootstrap")
        if not values:
            return Interval(None, None, None, 0, 0, method)
        estimate = statistic(values)
        if len(clusters) < 2:
            return Interval(estimate, None, None, len(values), len(clusters), method)
        generator = random.Random(self.seed)
        drawn: list[float] = []
        for _ in range(self.resamples):
            pool: list[float] = []
            for _ in range(len(clusters)):
                pool.extend(by_cluster[clusters[generator.randrange(len(clusters))]])
            if pool:
                drawn.append(statistic(pool))
        drawn.sort()
        low, high = percentile_bounds(drawn, self.level, self.quantile_rule)
        return Interval(estimate, low, high, len(values), len(clusters), method)


@INTERVALS.register("bootstrap")
class UnitBootstrap(_Resampling):
    """Resamples the observations themselves; for units that are independent."""

    def interval(
        self, observations: Sequence[Observation], statistic: Statistic = statistics.fmean
    ) -> Interval:
        values = [float(observation.value) for observation in observations]
        method = self._describe("bootstrap")
        if not values:
            return Interval(None, None, None, 0, 0, method)
        estimate = statistic(values)
        if len(values) < 2:
            return Interval(estimate, None, None, len(values), len(values), method)
        generator = random.Random(self.seed)
        size = len(values)
        drawn = sorted(
            statistic([values[generator.randrange(size)] for _ in range(size)])
            for _ in range(self.resamples)
        )
        low, high = percentile_bounds(drawn, self.level, self.quantile_rule)
        return Interval(estimate, low, high, size, size, method)


def wilson(successes: int, trials: int, *, level: float = 0.05) -> Interval:
    """A score interval for a share, which resampling cannot give at small n.

    A per-category cell often rests on a few dozen observations, where a bootstrap
    of the cell alone is unstable and a normal approximation runs past 0 and 1. The
    score interval stays inside the bounds and is defined at zero successes.
    """
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError(f"{successes} successes out of {trials} is not a share")
    if trials == 0:
        return Interval(None, None, None, 0, 0, "wilson")
    z = NormalDist().inv_cdf(1 - level / 2)
    share = successes / trials
    denominator = 1 + z * z / trials
    centre = (share + z * z / (2 * trials)) / denominator
    spread = (
        z * math.sqrt(share * (1 - share) / trials + z * z / (4 * trials * trials)) / denominator
    )
    return Interval(
        share, max(0.0, centre - spread), min(1.0, centre + spread), trials, trials, "wilson"
    )


def paired_difference(
    left: dict[str, float],
    right: dict[str, float],
    cluster_of: Callable[[str], str],
) -> list[Observation]:
    """Per-unit differences `left - right` on the units both sides have.

    Differencing per unit removes the difficulty of the unit from the comparison,
    which comparing two marginal intervals does not.
    """
    shared = sorted(set(left) & set(right))
    return [Observation(cluster_of(unit), left[unit] - right[unit]) for unit in shared]
