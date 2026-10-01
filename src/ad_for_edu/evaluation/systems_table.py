"""The table of systems: every metric, for every system, on the same moments.

Every system is scored on the same set of moments, which is the set the reference
writers described. A system scored on the moments it chose is scored on an easier
set, and the rows would not be comparable.

A row says how many of its moments each metric could score. A metric that needs
the references cannot score a moment with none; a metric that needs the picture
cannot score a moment whose keyframe is absent. Reporting the mean without the
number behind it hides which moments a row rests on.

A row averages over the seeds it has, and says how many that is. A row at one seed
and a row at three are not the same measurement.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..core.errors import ContractError
from ..inference.predictions import Prediction
from ..metrics.base import Metric, ScoringItem, mean_of
from ..systems.base import EvalMoment


@dataclass(frozen=True)
class SystemRow:
    """One system's values, one per metric, with how many moments each rests on."""

    system: str
    values: Mapping[str, float | None]
    counts: Mapping[str, int]
    moments: int
    described: int
    seeds: tuple[int, ...] = ()

    @property
    def seed_count(self) -> int:
        return len(self.seeds)

    def as_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "moments": self.moments,
            "described": self.described,
            "seeds": list(self.seeds),
            **{
                name: {"value": value, "n": self.counts.get(name, 0)}
                for name, value in self.values.items()
            },
        }


def items_for(
    predictions: Sequence[Prediction],
    moments: Mapping[str, EvalMoment],
    references: Mapping[str, Sequence[str]],
) -> list[ScoringItem]:
    """The predictions as things to score, in the order of the predictions."""
    items = []
    for prediction in predictions:
        moment = moments.get(prediction.moment_id)
        if moment is None:
            raise ContractError(
                f"there is nothing known about moment {prediction.moment_id!r}"
            )
        items.append(
            ScoringItem(
                moment_id=prediction.moment_id,
                text=prediction.ad_text,
                references=tuple(references.get(prediction.moment_id, ())),
                image=moment.keyframes[0] if moment.keyframes else None,
                fields={"time": moment.time, "type": moment.type},
            )
        )
    return items


def score_system(
    system: str,
    predictions: Sequence[Prediction],
    metrics: Mapping[str, Metric],
    moments: Mapping[str, EvalMoment],
    references: Mapping[str, Sequence[str]],
    *,
    seeds: Sequence[int] = (),
) -> SystemRow:
    """One row of the table."""
    items = items_for(predictions, moments, references)
    values: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for name, metric in metrics.items():
        scored = metric.score(items)
        if len(scored) != len(items):
            raise ContractError(
                f"metric {name!r} returned {len(scored)} values for {len(items)} moments"
            )
        values[name] = mean_of(scored)
        counts[name] = sum(1 for value in scored if value is not None)
    return SystemRow(
        system=system,
        values=values,
        counts=counts,
        moments=len(items),
        described=sum(1 for prediction in predictions if prediction.filled),
        seeds=tuple(seeds),
    )


def over_seeds(rows: Sequence[SystemRow]) -> SystemRow:
    """One row from the rows of one system at several seeds.

    The mean is over the seeds that exist, and the row says how many that is.
    """
    if not rows:
        raise ContractError("there are no rows to average")
    names = {name for row in rows for name in row.values}
    if any(set(row.values) != names for row in rows):
        raise ContractError("the rows do not carry the same metrics")
    values = {
        name: mean_of([row.values[name] for row in rows]) for name in sorted(names)
    }
    counts = {
        name: round(statistics.fmean([row.counts.get(name, 0) for row in rows]))
        for name in sorted(names)
    }
    return SystemRow(
        system=rows[0].system,
        values=values,
        counts=counts,
        moments=rows[0].moments,
        described=round(statistics.fmean([row.described for row in rows])),
        seeds=tuple(sorted({seed for row in rows for seed in row.seeds})),
    )


def render(rows: Sequence[SystemRow], metrics: Sequence[str], *, title: str) -> str:
    """The table as Markdown, with the number behind every value."""
    lines = [f"# {title}", ""]
    lines.append("| system | moments | described | seeds | " + " | ".join(metrics) + " |")
    lines.append("|---|---|---|---|" + "---|" * len(metrics))
    for row in rows:
        cells = []
        for name in metrics:
            value = row.values.get(name)
            if value is None:
                cells.append("-")
            else:
                cells.append(f"{value:.3f} ({row.counts.get(name, 0)})")
        lines.append(
            f"| {row.system} | {row.moments} | {row.described} | {row.seed_count} | "
            + " | ".join(cells)
            + " |"
        )
    lines.append("")
    lines.append(
        "Each cell is the mean and, in brackets, the number of moments it rests on. "
        "A row is the mean over the seeds it has, and the seed column says how many."
    )
    return "\n".join(lines)
