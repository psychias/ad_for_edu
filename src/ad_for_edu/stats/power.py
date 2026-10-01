"""What a test of a given size can detect.

A result that fails to reach significance says little unless it states the
smallest effect the test could have found. Two such effects are computed here,
both exactly, from the binomial distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

ALPHA = 0.05
POWER = 0.80


def detectable_agreement(n: int, alpha: float = ALPHA, power: float = POWER) -> float | None:
    """Smallest true agreement above one half that a one-sided exact test on `n` labels
    detects with probability `power`. None when no outcome of `n` labels rejects."""
    from scipy.stats import binom

    if n <= 0:
        return None
    critical = next(
        (k for k in range(n + 1) if binom.sf(k - 1, n, 0.5) <= alpha),
        None,
    )
    if critical is None:
        return None
    low, high = 0.5, 1.0
    for _ in range(60):
        middle = (low + high) / 2
        if binom.sf(critical - 1, n, middle) >= power:
            high = middle
        else:
            low = middle
    return high


@lru_cache(maxsize=256)
def rejection_region(discordant: int, alpha: float = ALPHA) -> tuple[int, ...]:
    """The win counts at which the exact two-sided test on `discordant` pairs rejects.

    The region depends on the number of pairs and the level only, so it is computed
    once and reused for every win share the power is evaluated at.
    """
    from scipy.stats import binomtest

    return tuple(
        wins
        for wins in range(discordant + 1)
        if binomtest(wins, discordant, 0.5).pvalue <= alpha
    )


def two_sided_power(discordant: int, win_share: float, alpha: float = ALPHA) -> float:
    """Power of the exact two-sided test on `discordant` pairs when one scorer wins each
    discordant pair with probability `win_share`."""
    from scipy.stats import binom

    region = rejection_region(discordant, alpha)
    if not region:
        return 0.0
    return math.fsum(binom.pmf(list(region), discordant, win_share))


@dataclass(frozen=True)
class DetectableDifference:
    """The smallest difference in agreement a paired test could have detected."""

    win_share: float | None
    percentage_points: float | None


def detectable_difference(
    discordant: int,
    n_shared: int,
    alpha: float = ALPHA,
    power: float = POWER,
) -> DetectableDifference:
    """Smallest share of discordant pairs one scorer must win for the test to reach
    `power`, and the difference in agreement that share implies.

    With d discordant pairs among n shared pairs, the difference in agreement is
    (2 p - 1) d / n. Power is not monotone in p when d is small, so the search takes
    the first share from which power stays at or above the target, and then refines
    inside that step.
    """
    if discordant <= 0 or n_shared <= 0:
        return DetectableDifference(None, None)
    grid = [0.5 + step / 2000 for step in range(1, 1001)]
    powers = [two_sided_power(discordant, share, alpha) for share in grid]
    # Walk from the top: the answer is the start of the last run that stays at or above
    # the target all the way to the end of the grid.
    start = len(grid)
    while start > 0 and powers[start - 1] >= power:
        start -= 1
    if start == len(grid):
        return DetectableDifference(None, None)
    low = grid[start - 1] if start > 0 else 0.5
    high = grid[start]
    for _ in range(60):
        middle = (low + high) / 2
        if two_sided_power(discordant, middle, alpha) >= power:
            high = middle
        else:
            low = middle
    return DetectableDifference(high, 100 * (2 * high - 1) * discordant / n_shared)
