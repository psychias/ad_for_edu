"""Scoring a system's predictions and summarising the result.

`score_predictions` returns one row per prediction, empty ones included, plus one
row for every moment the system never reached, so the number of rows always
equals the number of moments asked for. The summary lists each group before the
aggregate and prints the coverage beside every mean.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..core.errors import ContractError
from ..core.ids import moment_of
from ..data.schema import CATEGORIES, MomentContext
from .scorer import MISSING, UNFILLED, ComplianceScorer, ScoreRow
from .sequence import SequenceFactor, SequenceItem


def score_predictions(
    predictions: Iterable[Mapping[str, Any]],
    contexts: Mapping[str, MomentContext],
    scorer: ComplianceScorer,
    *,
    sequence_factor: SequenceFactor | None = None,
    times: Mapping[str, float] | None = None,
    system: str = "",
) -> list[dict[str, Any]]:
    """Score every prediction of one system.

    `sequence_factor` needs `times`, the time of each moment, to order the
    descriptions inside a lecture.
    """
    if sequence_factor is not None and times is None:
        raise ContractError("a sequence factor needs the time of each moment")
    base = {"unit": "prediction", "system": system, "mode": scorer.name}
    scored: dict[str, ScoreRow] = {}
    texts: dict[str, Any] = {}
    output_ids: dict[str, Any] = {}
    for prediction in predictions:
        moment = moment_of(prediction)
        context = contexts.get(moment)
        if context is None:
            raise ContractError(
                f"prediction for {moment!r} has no context. Check that predictions and "
                "context rows come from the same set of moments."
            )
        if moment in scored:
            raise ContractError(f"moment {moment!r} is predicted twice; one row per moment")
        scored[moment] = scorer.score(prediction.get("ad_text"), context)
        texts[moment] = prediction.get("ad_text")
        output_ids[moment] = prediction.get("output_id")
    if sequence_factor is not None:
        sequence = [
            SequenceItem(moment, float(times.get(moment, 0.0)), texts[moment]) for moment in scored
        ]
        sequence_factor.apply(scored, sequence)
    rows = [
        {
            **base,
            "output_id": output_ids[moment],
            "moment_id": moment,
            "type": contexts[moment].type,
            **row.as_dict(),
        }
        for moment, row in scored.items()
    ]
    for moment, context in contexts.items():
        if moment not in scored:
            rows.append(
                {
                    **base,
                    "output_id": None,
                    "moment_id": moment,
                    "type": context.type,
                    **ScoreRow.unscored(MISSING).as_dict(),
                }
            )
    return rows


def mean_and_median(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None}
    return {
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
    }


def summarise(rows: Sequence[Mapping[str, Any]], *, group_key: str = "type") -> dict[str, Any]:
    """Per group and in aggregate: counts, mean and median score, and each component."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key) or "unknown")].append(row)

    def block(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        scored = [row for row in members if row.get("overall") is not None]
        summary: dict[str, Any] = {
            "n_total": len(members),
            "n_scored": len(scored),
            "n_unfilled": sum(1 for row in members if row.get("reason") == UNFILLED),
            "n_missing": sum(1 for row in members if row.get("reason") == MISSING),
            "n_moments": len({row.get("moment_id") for row in members}),
            "overall": mean_and_median([float(row["overall"]) for row in scored]),
            "components": {},
        }
        for category in CATEGORIES:
            values = [
                float(row["components"][category])
                for row in scored
                if row["components"].get(category) is not None
            ]
            compliant = sum(1 for value in values if value >= 1.0)
            summary["components"][category] = {
                "n_applicable": len(values),
                "n_abstained": len(scored) - len(values),
                **mean_and_median(values),
                "n_compliant": compliant,
                "share_compliant": round(compliant / len(values), 4) if values else None,
            }
        return summary

    strata = {name: block(members) for name, members in sorted(groups.items())}
    return {
        "unit": "prediction",
        "group_key": group_key,
        "strata": strata,
        "aggregate": {"pools": sorted(strata), **block(rows)},
    }


def system_score(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """The score of a system: the mean over the descriptions that were scored."""
    values = [float(row["overall"]) for row in rows if row.get("overall") is not None]
    return statistics.fmean(values) if values else None


def render(summary: Mapping[str, Any], *, title: str) -> str:
    """The summary as a Markdown report."""
    lines = [f"# {title}", ""]

    def table(name: str, block: Mapping[str, Any]) -> None:
        overall = block["overall"]
        lines.append(f"## {name}")
        lines.append("")
        lines.append(
            f"{block['n_scored']} of {block['n_total']} descriptions scored "
            f"({block['n_unfilled']} empty, {block['n_missing']} not produced) "
            f"on {block['n_moments']} moments."
        )
        lines.append(f"Score: mean {overall['mean']}, median {overall['median']}.")
        lines.append("")
        lines.append("| category | scored | left out | mean | median | fully compliant |")
        lines.append("|---|---|---|---|---|---|")
        for category, cell in block["components"].items():
            share = cell["share_compliant"]
            shown = "-" if share is None else f"{100 * share:.1f}% ({cell['n_compliant']})"
            lines.append(
                f"| {category} | {cell['n_applicable']} | {cell['n_abstained']} | "
                f"{cell['mean']} | {cell['median']} | {shown} |"
            )
        lines.append("")

    for name, block in summary["strata"].items():
        table(f"{summary['group_key']}: {name}", block)
    table("all groups together: " + ", ".join(summary["aggregate"]["pools"]), summary["aggregate"])
    return "\n".join(lines)
