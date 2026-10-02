"""The commands that produce the tables.

Two habits run through all of them. A figure is reported with the number of units
behind it, because a share over eleven pairs and a share over four hundred read the
same otherwise. And a figure over mixed strata is reported per stratum first: a mean
over categories that pull in opposite directions lands at chance and says nothing.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...core.ids import lecture_of
from ...core.io import read_json, read_jsonl, write_json
from ...core.outputs import guard_new
from ...data.catalog import DataCatalog
from ...data.schema import MomentContext
from ..common import COMMANDS, Command, output_path, report

#: The six rule categories, in the order the tables print them.
CATEGORIES = ("style", "terminology", "length", "deixis", "faithfulness", "non_redundancy")


def contexts_of(rows: Sequence[Mapping[str, Any]]) -> dict[str, MomentContext]:
    """What is known about each moment when a description of it is scored.

    Takes rows already put through a field map, so the names here are this package's own
    whatever the dataset called them.
    """
    out = {}
    for row in rows:
        moment_id = str(row.get("moment_id") or row.get("id"))
        out[moment_id] = MomentContext(
            moment_id=moment_id,
            type=row.get("type"),
            pause_after=row.get("pause_after"),
            slide_ocr=row.get("slide_ocr"),
            what_on_screen=row.get("what_on_screen"),
            transcript_window=row.get("transcript_window"),
            renders_cursor=row.get("renders_cursor"),
        )
    return out


def add_field_options(parser: argparse.ArgumentParser) -> None:
    """How to read a dataset that calls its columns something else."""
    parser.add_argument(
        "--fields",
        nargs="*",
        default=None,
        metavar="NAME=COLUMN",
        help=(
            "map this package's field names onto a dataset's own columns, for example "
            "moment_id=id lecture=course time=start. Run with none to see the names."
        ),
    )
    parser.add_argument(
        "--field-map",
        type=Path,
        default=None,
        help="a YAML file of the same mapping, instead of listing it",
    )


def read_moments(args: argparse.Namespace, path: Path) -> tuple[list[dict], Any]:
    """The moments of any dataset, under this package's names, grouped by lecture.

    The grouping is checked rather than assumed: a dataset whose lecture cannot be found
    would give one lecture per moment, which silently stops the novelty term from ever
    firing.
    """
    from ...data.fields import FieldMap, assert_lectures_group

    given = getattr(args, "field_map", None)
    fields = FieldMap.load(given) if given else FieldMap.parse(getattr(args, "fields", None))
    rows = fields.rows(read_jsonl(path))
    assert_lectures_group(rows, fields=fields, what=str(path.name))
    return rows, fields


def read_described(args: argparse.Namespace, path: Path):
    """One system's descriptions, under this package's names whatever the dataset called them."""
    from ...data.fields import FieldMap
    from ...inference import predictions_from

    given = getattr(args, "field_map", None)
    fields = FieldMap.load(given) if given else FieldMap.parse(getattr(args, "fields", None))
    if not fields.renamed:
        from ...inference import read_predictions

        return read_predictions(path)
    return predictions_from(fields.rows(read_jsonl(path)), where=str(path))


def references_by_moment(path: Path | None, catalog: DataCatalog) -> dict[str, list[str]]:
    """Every description written for each moment, which the overlap metrics score against."""
    source = path or catalog.dataset_file("reference_rows")
    out: dict[str, list[str]] = {}
    for row in read_jsonl(source):
        if not row.get("emit") or not row.get("ad_text"):
            continue
        moment = str(row.get("moment_id") or str(row.get("output_id", "")).split("::")[0])
        out.setdefault(moment, []).append(str(row["ad_text"]))
    return out


def build_metrics(
    names: Sequence[str],
    contexts: Mapping[str, MomentContext],
    knee: float | None = None,
) -> dict[str, Any]:
    """The metrics asked for, by their registered names.

    A compliance mode is named `compliance:<mode>`, because the mode is what the
    figure is about and a table column called `compliance` would not say which.
    """
    from ...compliance.build import build_scorer, build_sequence_factor, load_compliance_settings
    from ...metrics import METRICS, ComplianceMetric

    settings = None
    built: dict[str, Any] = {}
    for name in names:
        if name.startswith("compliance:"):
            settings = settings or load_compliance_settings()
            mode = name.split(":", 1)[1]
            built[name] = ComplianceMetric(
                build_scorer(mode, settings),
                dict(contexts),
                build_sequence_factor(settings, knee=knee),
            )
            continue
        known = list(METRICS.names())
        if name not in known:
            raise SystemExit(f"unknown metric {name!r}; known: {known} and compliance:<mode>")
        built[name] = METRICS.create(name)
    return built


def side_scores(path: Path) -> dict[str, dict[str, Any]]:
    """Per-pair scores as `{pair_id: {scorer: PairScores}}`, read from a scored file.

    Reading them, rather than recomputing, is what lets a paid scorer's answers serve
    more than one table.
    """
    from ...evaluation import PairScores

    per_pair: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        pair_id = str(row["pair_id"])
        scorer = str(row.get("scorer") or row.get("metric") or "")
        first, second = row.get("a"), row.get("b")
        per_pair.setdefault(pair_id, {})[scorer] = PairScores(
            None if first is None else float(first),
            None if second is None else float(second),
        )
    return per_pair


def compliant_sides(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """The side that keeps the rule, for the pairs whose direction is known.

    Only the controlled pairs have one. A natural pair has no side that is right by
    construction, and giving it one would invent the answer being measured.
    """
    out = {}
    for row in rows:
        side = row.get("compliant_side")
        if side in ("a", "b"):
            out[str(row["pair_id"])] = side
    return out


@COMMANDS.register("score-systems")
class ScoreSystems(Command):
    name = "score-systems"
    help = "the table of systems: every metric on every system, with the moments behind it"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--predictions",
            type=Path,
            nargs="+",
            required=True,
            help="one predictions file per system; the file name names the system",
        )
        parser.add_argument("--moments", type=Path, required=True)
        parser.add_argument("--references", type=Path, default=None)
        parser.add_argument(
            "--metrics",
            nargs="+",
            default=["chrf", "compliance:mechanical", "compliance:local"],
        )
        parser.add_argument(
            "--knee",
            type=float,
            default=None,
            help="the novelty knee, when it is to differ from the settings file",
        )
        add_field_options(parser)

    def run(self, args: argparse.Namespace) -> int:
        from ...evaluation import over_seeds, render, score_system
        from .describe import eval_moments

        catalog = DataCatalog.load()
        rows, _fields = read_moments(args, args.moments)
        moments = {
            moment.moment_id: moment
            for moment in eval_moments(rows, catalog, with_pictures=False)
        }
        references = references_by_moment(args.references, catalog)
        metrics = build_metrics(args.metrics, contexts_of(rows), args.knee)

        by_system: dict[str, list[Any]] = {}
        for path in args.predictions:
            system = path.stem.replace("predictions_", "")
            seed = 0
            if "-seed" in system:
                system, _, tail = system.partition("-seed")
                seed = int(tail) if tail.isdigit() else 0
            by_system.setdefault(system, []).append(
                score_system(
                    system,
                    read_described(args, path),
                    metrics,
                    moments,
                    references,
                    seeds=(seed,),
                )
            )
        table_rows = [over_seeds(found) for _, found in sorted(by_system.items())]
        table = render(table_rows, list(metrics), title="Systems")
        target = guard_new(
            output_path(args, "evaluation", "systems.md"), overwrite=args.overwrite
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(table + "\n", encoding="utf-8")
        write_json(target.with_suffix(".json"), [row.as_dict() for row in table_rows])
        print(table)
        report(
            {"written": str(target), "systems": len(table_rows), "metrics": list(metrics)}
        )
        return 0


@COMMANDS.register("head-to-head-win-rate")
class WinRate(Command):
    name = "head-to-head-win-rate"
    help = "the share of decided moments one system wins, over both presentation orders"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--answers",
            type=Path,
            required=True,
            help="one row per moment and order, with the side the judge chose",
        )
        parser.add_argument("--first", required=True, help="the system on side a")
        parser.add_argument("--second", required=True, help="the system on side b")
        parser.add_argument("--judge", required=True)
        parser.add_argument("--resamples", type=int, default=5000)
        parser.add_argument("--seed", type=int, default=0)

    def run(self, args: argparse.Namespace) -> int:
        from ...evaluation import outcomes, win_rate
        from ...pairs import build_combination, load_judging_settings

        combination = build_combination(load_judging_settings())
        answers: dict[str, dict[int, str | None]] = {}
        for row in read_jsonl(args.answers):
            moment = str(row["moment_id"])
            answers.setdefault(moment, {})[int(row["order"])] = row.get("choice")
        settled = outcomes(answers, combination)
        found = win_rate(
            settled,
            first=args.first,
            second=args.second,
            judge=args.judge,
            resamples=args.resamples,
            seed=args.seed,
        )
        report(
            {
                **found.as_dict(),
                "combination": type(combination).__name__,
                "lectures": len({lecture_of(moment) for moment in settled}),
                "note": "the interval resamples lectures, not moments",
            },
            path=args.out,
        )
        return 0


@COMMANDS.register("pair-validity")
class PairValidity(Command):
    name = "pair-validity"
    help = "how many controlled pairs each scorer decides, and how often it is right"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--pairs", type=Path, required=True)
        parser.add_argument("--scores", type=Path, required=True)
        parser.add_argument(
            "--against",
            default="chrf",
            help="the scorer the others are tested against",
        )
        parser.add_argument(
            "--cascade",
            nargs="*",
            default=None,
            help="scorers tried in order, each deciding what the one before left tied",
        )

    def run(self, args: argparse.Namespace) -> int:
        from ...evaluation import Cascade, SingleScorer, measure, verdicts_against
        from ...evaluation.comparisons import compare_all
        from ...stats import ExactMcNemar, Holm
        from ...stats.paired import BothDecide

        rows = read_jsonl(args.pairs)
        correct = compliant_sides(rows)
        if not correct:
            raise SystemExit(
                "no pair in this file states the side that keeps the rule; coverage and "
                "accuracy are defined on the controlled pairs only"
            )
        scores = side_scores(args.scores)
        scorers = sorted({name for per_pair in scores.values() for name in per_pair})
        rules: dict[str, Any] = {name: SingleScorer(name) for name in scorers}
        if args.cascade:
            unknown = [name for name in args.cascade if name not in scorers]
            if unknown:
                raise SystemExit(f"the cascade names scorers that are not scored: {unknown}")
            rules["cascade"] = Cascade(args.cascade)

        measured, verdicts = {}, {}
        for name, rule in rules.items():
            decisions = [rule.decide(pair_id, scores[pair_id]) for pair_id in sorted(scores)]
            measured[name] = measure(decisions, correct).as_dict()
            verdicts[name] = verdicts_against(decisions, correct)
        if args.against not in verdicts:
            raise SystemExit(f"{args.against!r} is not among the scorers: {scorers}")
        tested = compare_all(args.against, verdicts, ExactMcNemar(), BothDecide(), Holm())
        report(
            {
                "pairs_with_a_known_side": len(correct),
                "per_scorer": measured,
                "against": args.against,
                "tests": tested,
                "note": (
                    "each test runs on the pairs both scorers decide, and its n says how "
                    "many that is; the p values are corrected over the family"
                ),
            },
            path=args.out,
        )
        return 0


@COMMANDS.register("localisation")
class Localisation(Command):
    name = "localisation"
    help = "whether the subscore for a category prefers the side that keeps that category"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--pairs", type=Path, required=True)
        parser.add_argument(
            "--subscores",
            type=Path,
            required=True,
            help="per pair and category, the grade given to each side",
        )

    def run(self, args: argparse.Namespace) -> int:
        from ...stats import wilson

        rows = {str(row["pair_id"]): row for row in read_jsonl(args.pairs)}
        correct = compliant_sides(list(rows.values()))
        graded = {str(row["pair_id"]): row for row in read_jsonl(args.subscores)}
        per_category = {}
        for category in CATEGORIES:
            members = [
                pair_id
                for pair_id, row in rows.items()
                if str(row.get("axis") or row.get("dimension") or "") == category
                and pair_id in correct
            ]
            right = decided = tied = 0
            for pair_id in members:
                grades = graded.get(pair_id) or {}
                first = (grades.get("a") or {}).get(category)
                second = (grades.get("b") or {}).get(category)
                if first is None or second is None:
                    continue
                if first == second:
                    tied += 1
                    continue
                decided += 1
                right += ("a" if first > second else "b") == correct[pair_id]
            interval = wilson(right, decided)
            per_category[category] = {
                "pairs_on_this_category": len(members),
                "decided": decided,
                "ties": tied,
                "right": right,
                "accuracy": interval.estimate,
                "interval": interval.as_dict(),
            }
        report(
            {
                "per_category": per_category,
                "note": (
                    "each category is measured on the pairs built to differ on it, by the "
                    "subscore for that category alone; the categories are not pooled, "
                    "because they pull in opposite directions"
                ),
            },
            path=args.out,
        )
        return 0


@COMMANDS.register("rank-stability")
class RankStability(Command):
    name = "rank-stability"
    help = "how far the order of the systems moves as the novelty knee changes"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--per-moment",
            type=Path,
            required=True,
            help="per system and moment: the score before the factor, and the description",
        )
        parser.add_argument("--resamples", type=int, default=2000)
        parser.add_argument("--seed", type=int, default=0)

    def run(self, args: argparse.Namespace) -> int:
        from ...compliance.build import load_compliance_settings
        from ...compliance.sequence import SEQUENCE_FACTORS, SequenceItem
        from ...evaluation import rank_stability

        settings = load_compliance_settings()
        if not settings.kappa_sweep:
            raise SystemExit("the settings file lists no knees to sweep")
        if settings.sequence_factor is None:
            raise SystemExit("the settings file declares no sequence factor to sweep")

        by_system: dict[str, list[Mapping[str, Any]]] = {}
        for row in read_jsonl(args.per_moment):
            by_system.setdefault(str(row["system"]), []).append(row)
        by_knee: dict[Any, dict[str, float]] = {}
        for knee in settings.kappa_sweep:
            factor = SEQUENCE_FACTORS.create(settings.sequence_factor.name, knee=knee)
            per_system = {}
            for system, moments in by_system.items():
                scored = [row for row in moments if row.get("score") is not None]
                sequence = [
                    SequenceItem(
                        str(row["moment_id"]),
                        float(row.get("time") or 0.0),
                        row.get("ad_text"),
                    )
                    for row in scored
                ]
                effects = factor.effects(sequence)
                values = [
                    float(row["score"]) * effects[str(row["moment_id"])].factor for row in scored
                ]
                per_system[system] = sum(values) / len(values) if values else 0.0
            by_knee[knee] = per_system
        found = rank_stability(by_knee, resamples=args.resamples, seed=args.seed)
        report(
            {
                "knees": list(settings.kappa_sweep),
                "systems": found["systems"],
                "spearman_against_the_first_knee": found["against_first"],
                "ranks": found["ranks"],
            },
            path=args.out,
        )
        return 0


@COMMANDS.register("metric-correlations")
class MetricCorrelations(Command):
    name = "metric-correlations"
    help = "how far each pair of metrics puts the systems in the same order"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--table",
            type=Path,
            required=True,
            help="the systems table written by score-systems, as JSON",
        )
        parser.add_argument("--resamples", type=int, default=2000)
        parser.add_argument("--seed", type=int, default=0)

    def run(self, args: argparse.Namespace) -> int:
        from ...evaluation import correlations, render_correlations

        table = read_json(args.table)
        values: dict[str, dict[str, float]] = {}
        for row in table:
            for name, cell in row.items():
                if not isinstance(cell, Mapping) or cell.get("value") is None:
                    continue
                values.setdefault(name, {})[str(row["system"])] = float(cell["value"])
        if len(values) < 2:
            raise SystemExit("a correlation needs at least two metrics with values")
        found = correlations(values, resamples=args.resamples, seed=args.seed)
        rendered = render_correlations(found)
        print(rendered)
        report(
            {
                "metrics": sorted(values),
                "systems": len(next(iter(values.values()))),
                "correlations": {
                    f"{first} and {second}": interval.as_dict()
                    for (first, second), interval in sorted(found.items())
                },
            },
            path=args.out,
        )
        return 0


@COMMANDS.register("reference-writer-compliance")
class ReferenceWriterCompliance(Command):
    name = "reference-writer-compliance"
    help = "the same metric on the writers of the references, scored like any system"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--references", type=Path, default=None)
        parser.add_argument("--moments", type=Path, required=True)
        parser.add_argument(
            "--metrics",
            nargs="+",
            default=["compliance:mechanical", "compliance:local"],
        )
        add_field_options(parser)

    def run(self, args: argparse.Namespace) -> int:
        from ...evaluation import render, score_system
        from ...inference import Prediction
        from .describe import eval_moments

        catalog = DataCatalog.load()
        source = args.references or catalog.dataset_file("reference_rows")
        rows, _fields = read_moments(args, args.moments)
        moments = {
            moment.moment_id: moment
            for moment in eval_moments(rows, catalog, with_pictures=False)
        }
        metrics = build_metrics(args.metrics, contexts_of(rows))

        by_writer: dict[str, list[Any]] = {}
        for row in read_jsonl(source):
            moment = str(row.get("moment_id") or str(row.get("output_id", "")).split("::")[0])
            if moment not in moments:
                continue
            writer = str(row.get("family") or row.get("writer") or "")
            by_writer.setdefault(writer, []).append(
                Prediction(
                    output_id=str(row.get("output_id") or f"{moment}::{writer}"),
                    moment_id=moment,
                    emit=bool(row.get("emit")),
                    ad_text=row.get("ad_text"),
                    rung=row.get("rung"),
                    forced=False,
                    how="reference",
                )
            )
        table_rows = []
        for writer, predictions in sorted(by_writer.items()):
            # A writer chose which moments to describe, so its set is not the shared
            # one. The row carries the count beside every value, and the note says so.
            described = [prediction for prediction in predictions if prediction.filled]
            table_rows.append(
                score_system(
                    writer, described, metrics, moments, {moment: [] for moment in moments}
                )
            )
        table = render(table_rows, list(metrics), title="Reference writers")
        print(table)
        report(
            {
                "writers": {row.system: row.described for row in table_rows},
                "note": (
                    "a writer is scored on the moments it chose to describe, which differ "
                    "between writers; the count is beside every value"
                ),
            },
            path=args.out,
        )
        return 0
