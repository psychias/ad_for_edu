"""Comparing scorers, and comparing metrics.

**Scorers against one another.** Two scorers are compared on the pairs both were
asked about, by counting the pairs exactly one of them got right. The rule for a
pair a scorer did not decide changes the answer, so it is part of the result and
is printed with it; a family of such comparisons is corrected for its size.

A comparison that finds nothing says little on its own. Every one therefore states
the smallest difference it could have found, so that "no difference" and "too few
pairs to tell" are distinguishable.

**Metrics against one another.** Two metrics are compared by how they order the
systems. A correlation near zero means they order along different axes; it does
not mean either is wrong.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..stats.corrections import Correction
from ..stats.correlation import spearman_with_interval
from ..stats.intervals import Interval
from ..stats.paired import PairedTest, PairingRule, TestResult, Verdicts
from ..stats.power import detectable_difference


@dataclass(frozen=True)
class Comparison:
    """One scorer against another, with what it could have found."""

    first: str
    second: str
    result: TestResult
    smallest_detectable: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "first": self.first,
            "second": self.second,
            **self.result.as_dict(),
            "smallest_detectable_difference_pp": (
                round(self.smallest_detectable, 2)
                if self.smallest_detectable is not None
                else None
            ),
        }


def compare(
    first: str,
    second: str,
    verdicts: Mapping[str, Verdicts],
    test: PairedTest,
    pairing: PairingRule,
) -> Comparison:
    """One scorer against another, on the pairs the pairing rule keeps."""
    result = test.test(verdicts[first], verdicts[second], pairing)
    detectable = detectable_difference(
        result.first_only + result.second_only, result.n
    ).percentage_points
    return Comparison(first, second, result, detectable)


def compare_all(
    against: str,
    verdicts: Mapping[str, Verdicts],
    test: PairedTest,
    pairing: PairingRule,
    correction: Correction,
    *,
    alpha: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Every scorer against one, corrected for the size of the family."""
    comparisons = {
        name: compare(name, against, verdicts, test, pairing)
        for name in verdicts
        if name != against
    }
    adjusted = correction.adjust(
        {name: comparison.result.p_value for name, comparison in comparisons.items()}, alpha
    )
    return {
        name: {
            **comparison.as_dict(),
            "p_corrected": adjusted[name].adjusted,
            "distinct": adjusted[name].reject,
        }
        for name, comparison in comparisons.items()
    }


def correlations(
    values: Mapping[str, Mapping[str, float]],
    *,
    resamples: int = 2000,
    seed: int = 0,
    quantile_rule: str = "symmetric",
) -> dict[tuple[str, str], Interval]:
    """How each pair of metrics orders the same systems.

    Every correlation of the matrix draws from one stream, so the matrix as a whole
    is reproducible from the seed.
    """
    names = sorted(values)
    systems = sorted(set.intersection(*(set(values[name]) for name in names))) if names else []
    generator = random.Random(seed)
    found: dict[tuple[str, str], Interval] = {}
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            found[(first, second)] = spearman_with_interval(
                [values[first][system] for system in systems],
                [values[second][system] for system in systems],
                generator,
                resamples=resamples,
                quantile_rule=quantile_rule,
            )
    return found


def render_correlations(found: Mapping[tuple[str, str], Interval]) -> str:
    """The correlations as Markdown, each with its interval and the systems behind it."""
    lines = ["| metrics | rho | 95% interval | systems |", "|---|---|---|---|"]
    for (first, second), interval in sorted(found.items()):
        if interval.estimate is None:
            lines.append(f"| {first} and {second} | - | - | {interval.n} |")
            continue
        bounds = (
            f"[{interval.low:.2f}, {interval.high:.2f}]" if interval.defined else "-"
        )
        lines.append(
            f"| {first} and {second} | {interval.estimate:.2f} | {bounds} | {interval.n} |"
        )
    return "\n".join(lines)


def ranks_of(values: Mapping[str, float], *, highest_first: bool = True) -> dict[str, int]:
    """Each system's position under one metric, 1 being the best."""
    ordered = sorted(values, key=lambda name: values[name], reverse=highest_first)
    return {name: position for position, name in enumerate(ordered, start=1)}


def rank_stability(
    by_setting: Mapping[Any, Mapping[str, float]],
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """How much the order of the systems changes as a setting of the metric changes.

    Reported as the correlation of each setting's order with the first, and as
    where one named system lands under each.
    """
    settings = list(by_setting)
    if len(settings) < 2:
        raise ValueError("stability needs at least two settings to compare")
    generator = random.Random(seed)
    first = by_setting[settings[0]]
    systems = sorted(set.intersection(*(set(values) for values in by_setting.values())))
    out: dict[str, Any] = {"settings": settings, "systems": len(systems), "against_first": {}}
    for setting in settings[1:]:
        out["against_first"][str(setting)] = spearman_with_interval(
            [first[system] for system in systems],
            [by_setting[setting][system] for system in systems],
            generator,
            resamples=resamples,
        ).estimate
    out["ranks"] = {
        str(setting): ranks_of({system: by_setting[setting][system] for system in systems})
        for setting in settings
    }
    return out
