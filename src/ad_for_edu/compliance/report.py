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
from .rules import RuleMap
from .scorer import MISSING, UNFILLED, ComplianceScorer, ScoreRow, registered_name
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
        # The factor is the only check on the rules against re-describing what an
        # earlier description already covered, so a description it cuts has to name
        # them. Without this the diagnostic reports those rules as unbroken on a
        # system that repeated itself on every moment.
        if scorer.rules is not None:
            for row in scored.values():
                if row.novelty_factor is not None and row.novelty_factor < 1.0:
                    row.broken = tuple(
                        dict.fromkeys([*row.broken, *scorer.rules.sequence])
                    )
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


# ------------------------------------------------------- which rule was broken
def rule_diagnostic(
    rows: Sequence[Mapping[str, Any]],
    scorer: ComplianceScorer,
    rules: RuleMap,
    *,
    examples: int = 3,
) -> dict[str, Any]:
    """Per rule: how many descriptions could break it, and how many did.

    Two denominators matter and both are easy to get wrong.

    The rules reported are only those whose component **ran in this mode**. A mode that
    abstains on faithfulness never tested the faithfulness rules, and listing them at
    nought per cent broken would read as checked and passed when the check was absent.

    Within a reported rule, the denominator is the scored descriptions the rule
    **applies** to by its own type condition, not every description. A rule for figures
    broken on 3 of 4 figures is a different finding from 3 of 400 descriptions, and only
    the first is about the rule.
    """
    if not scorer.names_rules:
        raise ContractError(
            f"the {scorer.name!r} scorer was built without a rule map, so no row carries "
            "a rule id and a diagnostic would report every rule as unbroken"
        )
    scored = [row for row in rows if row.get("overall") is not None]
    if not scored:
        raise ContractError("there is nothing to diagnose: no description was scored")
    if not any(row.get("novelty_factor") is not None for row in scored):
        raise ContractError(
            "these rows were not scored with the sequence factor, so the two rules it "
            "answers for carry no verdict. Score the predictions as a system, which "
            "applies it, before diagnosing them."
        )
    ran = [registered_name(component) for component in scorer.components]
    reachable = rules.rules_of_mode(ran) + rules.sequence
    per_rule: dict[str, dict[str, Any]] = {}
    for rule_id in reachable:
        entry = rules.table[rule_id]
        applicable = [row for row in scored if entry.applies(row.get("type"), True)]
        offenders = [row for row in applicable if rule_id in (row.get("broken") or ())]
        below = [row for row in applicable if rule_id in (row.get("short") or ())]
        per_rule[rule_id] = {
            "number": entry.number,
            "route": entry.route,
            "summary": entry.summary,
            "applies_to": len(applicable),
            "broken_by": len(offenders),
            "share_broken": (
                round(len(offenders) / len(applicable), 4) if applicable else None
            ),
            # A graded component's shortfall, which is not a violation. Reported beside
            # the breaches rather than added to them.
            "scored_below_full_credit": len(below),
            "share_below": round(len(below) / len(applicable), 4) if applicable else None,
            "graded": bool(below) and not offenders,
            "examples": [
                {
                    "moment_id": row.get("moment_id"),
                    "type": row.get("type"),
                    "overall": row.get("overall"),
                }
                for row in sorted(
                    offenders, key=lambda row: float(row.get("overall") or 0.0)
                )[:examples]
            ],
        }
    named = {
        rule_id
        for row in scored
        for rule_id in (*(row.get("broken") or ()), *(row.get("short") or ()))
    }
    unknown = sorted(named - set(reachable))
    if unknown:
        raise ContractError(
            f"rows name rules the map cannot reach: {unknown}. A diagnostic must not "
            "report a rule whose denominator it cannot state."
        )
    return {
        "mode": scorer.name,
        "components_that_ran": ran,
        "descriptions_scored": len(scored),
        "rules_reachable": list(reachable),
        "per_rule": per_rule,
        "rules_no_description_broke": sorted(set(reachable) - named),
        "rules_this_mode_cannot_reach": [
            rule_id
            for rule_id in rules.rules_of_mode(sorted(rules.by_component)) + rules.sequence
            if rule_id not in reachable
        ],
        "coverage": rules.summary(),
    }


def render_rule_diagnostic(found: Mapping[str, Any], *, title: str) -> str:
    """The diagnostic as Markdown, most-broken rule first."""
    lines = [f"# {title}", ""]
    lines.append(
        f"Mode {found['mode']}, {found['descriptions_scored']} descriptions, against the "
        f"{len(found['rules_reachable'])} rules its components can attribute a failure to."
    )
    lines.append("")
    lines.append("| rule | # | route | applies to | broken | share | below full credit |")
    lines.append("|---|---|---|---|---|---|---|")
    ordered = sorted(
        found["per_rule"].items(),
        key=lambda item: (
            -(item[1]["share_broken"] or 0.0),
            -(item[1]["share_below"] or 0.0),
            item[1]["number"],
        ),
    )
    for rule_id, cell in ordered:
        share = cell["share_broken"]
        shown = "-" if share is None else f"{100 * share:.1f}%"
        below = cell["share_below"]
        graded = (
            "-"
            if not cell["scored_below_full_credit"]
            else f"{cell['scored_below_full_credit']} ({100 * below:.1f}%)"
        )
        lines.append(
            f"| {rule_id} | {cell['number']} | {cell['route']} | "
            f"{cell['applies_to']} | {cell['broken_by']} | {shown} | {graded} |"
        )
    lines.append("")
    if any(cell["scored_below_full_credit"] for cell in found["per_rule"].values()):
        lines.append(
            "The last column is a graded component scoring below full credit, which is "
            "not a breach: a description that names something the slide does not spell "
            "out scores below one on terminology and has broken no rule. Those counts "
            "are not added to the broken column."
        )
        lines.append("")
    unreached = found["rules_this_mode_cannot_reach"]
    if unreached:
        lines.append(
            f"Not tested here, because this mode runs no component for them: "
            f"{', '.join(unreached)}. They are absent from the table above rather than "
            "listed as unbroken."
        )
        lines.append("")
    gaps = found["coverage"]["routed_mechanical_without_a_component"]
    if gaps:
        lines.append(
            "The standard routes these as mechanically checkable and this repository "
            f"implements no check for them: {', '.join(gaps)}."
        )
    never = found["rules_no_description_broke"]
    if never:
        lines.append("")
        lines.append(f"Tested and broken by no description: {', '.join(never)}.")
    lines.append("")
    lines.append(
        "A share is over the descriptions the rule applies to, by its own type "
        "condition, not over every description scored."
    )
    return "\n".join(lines)
