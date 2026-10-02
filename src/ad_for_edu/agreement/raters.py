"""How far a metric agrees with the people who rated the pairs.

The measurement is per rater, over that rater's own choices. The raters differ in
how often they call a pair equal, so a consensus label would answer a different
question: it would keep the pairs they found easy and drop the rest.

Three rules hold.

**A rater's ties are not choices.** A pair a rater called equal says nothing about
which side they preferred, and is left out of that rater's cells.

**A metric's ties are counted, not dropped silently.** A metric that gives both
sides the same value did not agree or disagree, so the number of such pairs is
reported beside the agreement.

**The pairs a judge could not order are their own row.** Those are the pairs a
judge answered differently each way round, and averaging them into the rest would
describe a different population. They are reported, never mixed in.

**Raters are named by what a report calls them, not by who they are.** That is
enforced rather than asked for: `checked_rater_names` refuses anything but a
pseudonym, so a label file named after the person who produced it cannot put their
name into a report, a table or a saved result. People who rate study material are
study participants, and a pseudonym is the only identity this package will carry.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import ContractError
from ..data.schema import SIDES, TIE
from ..evaluation.decisions import PairScores
from ..stats.agreement import AgreementCoefficient, complete_units, unanimous

#: Pairs a judge answered differently in the two orders. Reported on their own.
ORDER_SENSITIVE = "natural_flip"

#: The only shape a rater's name may take: R1, or rater1, or rater_1.
PSEUDONYM = re.compile(r"^(?:R|rater[_-]?)(\d{1,3})$", re.IGNORECASE)


def is_pseudonym(name: str) -> bool:
    """Whether `name` says what a report calls a rater rather than who they are."""
    return bool(PSEUDONYM.match(str(name).strip()))


def checked_rater_names(names: Iterable[str]) -> tuple[str, ...]:
    """The names, if every one is a pseudonym; otherwise an error quoting none of them.

    A name that is not a pseudonym may be a participant's, and this package cannot
    tell which it is. So it refuses the whole set rather than guess, and the message
    counts the offenders instead of repeating them: an error that prints the name it
    is protecting has written it into every log that captured the error.
    """
    given = [str(name).strip() for name in names]
    if not given:
        raise ContractError("no rater was named")
    wrong = sum(1 for name in given if not is_pseudonym(name))
    if wrong:
        raise ContractError(
            f"{wrong} of {len(given)} rater names are not pseudonyms. A rater is named "
            "R1, rater1 or rater_1, never after the person who rated. Rename the label "
            "files, or name the raters with --raters in the order the files are given."
        )
    repeated = sorted({name for name in given if given.count(name) > 1})
    if repeated:
        raise ContractError(f"a rater is named more than once: {repeated}")
    return tuple(given)


@dataclass(frozen=True)
class Cell:
    """One metric against one rater, on one group of pairs."""

    agreed: int
    disagreed: int
    metric_ties: int

    @property
    def n(self) -> int:
        return self.agreed + self.disagreed

    @property
    def agreement(self) -> float | None:
        return self.agreed / self.n if self.n else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "agreed": self.agreed,
            "metric_ties": self.metric_ties,
            "agreement": round(self.agreement, 4) if self.agreement is not None else None,
        }


def cell_for(
    labels: Mapping[str, str],
    scores: Mapping[str, PairScores],
    pairs: Sequence[str] | None = None,
) -> Cell:
    """How often the metric preferred the side the rater chose."""
    considered = pairs if pairs is not None else sorted(set(labels) & set(scores))
    agreed = disagreed = ties = 0
    for pair in considered:
        chosen = labels.get(pair)
        found = scores.get(pair)
        if chosen not in SIDES or found is None:
            continue
        if found.tied:
            ties += 1
        elif found.decided:
            if found.preferred == chosen:
                agreed += 1
            else:
                disagreed += 1
    return Cell(agreed, disagreed, ties)


def verdicts_for(
    labels: Mapping[str, str], scores: Mapping[str, PairScores]
) -> dict[str, bool | None]:
    """Per pair: whether the metric agreed with the rater, or None when either did not choose."""
    verdicts: dict[str, bool | None] = {}
    for pair, chosen in labels.items():
        found = scores.get(pair)
        if chosen not in SIDES or found is None:
            continue
        verdicts[pair] = found.preferred == chosen if found.decided else None
    return verdicts


@dataclass
class RaterReport:
    """Every metric against every rater, per group and in total."""

    raters: tuple[str, ...]
    metrics: tuple[str, ...]
    overall: Mapping[str, Mapping[str, Cell]] = field(default_factory=dict)
    by_group: Mapping[str, Mapping[str, Mapping[str, Cell]]] = field(default_factory=dict)
    order_sensitive: Mapping[str, Mapping[str, Cell]] = field(default_factory=dict)
    agreement: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "raters": list(self.raters),
            "metrics": list(self.metrics),
            "overall": {
                rater: {metric: cell.as_dict() for metric, cell in cells.items()}
                for rater, cells in self.overall.items()
            },
            "by_group": {
                group: {
                    rater: {metric: cell.as_dict() for metric, cell in cells.items()}
                    for rater, cells in per_rater.items()
                }
                for group, per_rater in self.by_group.items()
            },
            "order_sensitive": {
                rater: {metric: cell.as_dict() for metric, cell in cells.items()}
                for rater, cells in self.order_sensitive.items()
            },
            "agreement": dict(self.agreement),
        }


def build_report(
    pairs: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, str]],
    scores: Mapping[str, Mapping[str, PairScores]],
    *,
    coefficient: AgreementCoefficient,
    group_by: str = "axis",
) -> RaterReport:
    """The report: every metric against every rater, grouped, with the pairs a judge
    could not order kept out of every other cell."""
    raters = tuple(sorted(labels))
    metrics = tuple(sorted(scores))
    by_id = {str(pair["pair_id"]): pair for pair in pairs}
    order_sensitive_pairs = [
        pair_id
        for pair_id, pair in by_id.items()
        if pair.get("stratum") == ORDER_SENSITIVE
    ]
    rest = [pair_id for pair_id in by_id if pair_id not in set(order_sensitive_pairs)]

    groups: dict[str, list[str]] = {}
    for pair_id in rest:
        name = str(by_id[pair_id].get(group_by) or "natural")
        groups.setdefault(name, []).append(pair_id)

    report = RaterReport(raters=raters, metrics=metrics)
    report.overall = {
        rater: {metric: cell_for(labels[rater], scores[metric], rest) for metric in metrics}
        for rater in raters
    }
    report.by_group = {
        group: {
            rater: {
                metric: cell_for(labels[rater], scores[metric], members) for metric in metrics
            }
            for rater in raters
        }
        for group, members in sorted(groups.items())
    }
    report.order_sensitive = {
        rater: {
            metric: cell_for(labels[rater], scores[metric], order_sensitive_pairs)
            for metric in metrics
        }
        for rater in raters
    }
    units = complete_units(labels, sorted(by_id))
    report.agreement = {
        "units": len(units),
        "coefficient": round(coefficient.value(units), 4) if units else None,
        "unanimous_on_a_side": unanimous(units, ignoring=TIE),
        "ties_per_rater": {
            rater: sum(1 for side in labels[rater].values() if side == TIE) for rater in raters
        },
    }
    return report


def render(report: RaterReport, *, title: str) -> str:
    """The report as Markdown, with the number behind every cell."""
    lines = [f"# {title}", ""]
    lines.append(
        "Per rater, over that rater's own choices. A cell gives the agreement and, in "
        "brackets, how many of that rater's choices it rests on."
    )
    lines.append("")
    lines.append("| metric | " + " | ".join(report.raters) + " |")
    lines.append("|---|" + "---|" * len(report.raters))
    for metric in report.metrics:
        cells = []
        for rater in report.raters:
            cell = report.overall[rater][metric]
            if cell.agreement is None:
                cells.append("-")
            else:
                cells.append(f"{100 * cell.agreement:.1f} ({cell.n})")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines.append("")
    agreement = report.agreement
    lines.append(
        f"Agreement between the raters over {agreement['units']} pairs every rater judged: "
        f"{agreement['coefficient']}. They agreed on a side on "
        f"{agreement['unanimous_on_a_side']} of them."
    )
    lines.append("")
    lines.append(
        "The pairs a judge answered differently in the two presentation orders are reported "
        "separately and are in no cell above."
    )
    return "\n".join(lines)
