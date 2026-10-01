"""The whole evaluation in one invocation, and the per-rule diagnostic.

`evaluate` runs the stages of the evaluation that the inputs it is given allow, and
writes one directory. It calls the individual commands rather than reimplementing
them, so a stage cannot behave one way on its own and another way in the suite.

Two properties are worth stating because they are what make the suite honest.

*One estimate, one decision.* Each paid stage prices its own work. Run together they
would ask several times over, so the suite totals them first, prints one estimate,
and stops unless it is told to spend. Nothing that can call a model is built before
that point.

*A stage that cannot run is named, not skipped quietly.* The suite reports, per stage,
whether it ran, what it wrote, and why it did not — so a half-filled directory cannot
be read as a complete evaluation.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ...core.errors import AdForEduError
from ...core.io import read_jsonl, write_json
from ...core.settings import work_dir
from ..common import COMMANDS, Command, approved, report

#: The stages the suite runs, in order, with the inputs each one needs.
#: A stage whose inputs are absent is reported as not run, with the input named.
FREE_STAGES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("score-systems", ("predictions", "moments")),
    ("diagnose", ("predictions", "moments")),
    ("reference-writer-compliance", ("moments",)),
    ("rank-stability", ("per_moment",)),
    ("metric-correlations", ("systems_table",)),
    ("pair-validity", ("pairs", "scores")),
    ("localisation", ("pairs", "subscores")),
    ("rater-agreement", ("pairs", "labels", "scores")),
    ("head-to-head-win-rate", ("answers",)),
)

#: The paid stages, with the input each one reads.
PAID_STAGES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("score-rubric", ("predictions",)),
    ("rate-against-references", ("predictions",)),
    ("judge-head-to-head", ("head_to_head",)),
)


def _call(name: str, arguments: Sequence[str], *, approve: bool) -> dict[str, Any]:
    """Run one command as the program would, and report what happened.

    The command is invoked through the same entry point a person would use, so the
    suite cannot diverge from the single-command behaviour it is standing in for.
    """
    from ..main import main

    argv = [name, *arguments]
    if approve:
        argv.append("--approve-spend")
    try:
        status = main(argv)
    except SystemExit as stop:  # a priced stage that stopped, or a refusal
        code = stop.code if isinstance(stop.code, int) else 1
        return {"ran": code == 0, "status": code, "argv": argv}
    except AdForEduError as error:
        return {"ran": False, "status": 1, "argv": argv, "why": str(error)}
    return {"ran": status == 0, "status": status, "argv": argv}


@COMMANDS.register("diagnose")
class Diagnose(Command):
    name = "diagnose"
    help = "which rule each description breaks, per rule, with the moments behind it"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--predictions", type=Path, required=True)
        parser.add_argument("--moments", type=Path, required=True)
        parser.add_argument(
            "--mode",
            default="mechanical",
            help="which offline mode to diagnose; the mode decides which rules are tested",
        )

    def run(self, args: argparse.Namespace) -> int:
        from ...compliance import (
            RuleMap,
            build_scorer,
            build_sequence_factor,
            render_rule_diagnostic,
            rule_diagnostic,
            score_predictions,
        )
        from ...core.outputs import guard_new
        from ...inference import read_predictions
        from ..common import output_path
        from .evaluate import contexts_of

        moment_rows = read_jsonl(args.moments)
        contexts = contexts_of(moment_rows)
        times = {
            str(row.get("moment_id") or row.get("id")): float(
                row.get("t") or row.get("time") or 0.0
            )
            for row in moment_rows
        }
        predictions = [
            prediction.as_dict() for prediction in read_predictions(args.predictions)
        ]
        scorer = build_scorer(args.mode)
        rows = score_predictions(
            predictions,
            contexts,
            scorer,
            sequence_factor=build_sequence_factor(),
            times=times,
            system=args.predictions.stem.replace("predictions_", ""),
        )
        found = rule_diagnostic(rows, scorer, RuleMap.load())
        rendered = render_rule_diagnostic(
            found, title=f"Rules broken, {args.mode} mode"
        )
        print(rendered)
        target = guard_new(
            output_path(args, "evaluation", f"rules_broken_{args.mode}.md"),
            overwrite=args.overwrite,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
        write_json(target.with_suffix(".json"), found)
        report(
            {
                "written": str(target),
                "mode": args.mode,
                "descriptions_scored": found["descriptions_scored"],
                "rules_tested": len(found["rules_reachable"]),
                "rules_this_mode_cannot_reach": found["rules_this_mode_cannot_reach"],
            }
        )
        return 0


@COMMANDS.register("evaluate")
class Evaluate(Command):
    name = "evaluate"
    help = "run the evaluation: every table the given inputs allow, into one directory"
    paid = True

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--predictions", type=Path, nargs="*", default=None)
        parser.add_argument("--moments", type=Path, default=None)
        parser.add_argument("--references", type=Path, default=None)
        parser.add_argument("--pairs", type=Path, default=None)
        parser.add_argument("--scores", type=Path, default=None)
        parser.add_argument("--subscores", type=Path, default=None)
        parser.add_argument("--labels", type=Path, nargs="*", default=None)
        parser.add_argument("--answers", type=Path, default=None)
        parser.add_argument("--per-moment", type=Path, default=None)
        parser.add_argument("--head-to-head", type=Path, default=None)
        parser.add_argument("--judge", default=None, help="the judge of the paid stages")
        parser.add_argument(
            "--metrics",
            nargs="+",
            default=["chrf", "compliance:mechanical", "compliance:local"],
        )
        parser.add_argument(
            "--offline",
            action="store_true",
            help="run only the stages that cost nothing, and do not price the others",
        )
        parser.add_argument(
            "--mode",
            default="mechanical",
            help="which mode the per-rule diagnostic reports",
        )

    def run(self, args: argparse.Namespace) -> int:
        base = args.out or (work_dir() / "evaluation")
        base.mkdir(parents=True, exist_ok=True)
        present = self._inputs(args)

        plan: list[dict[str, Any]] = []
        for name, needs in FREE_STAGES:
            missing = [need for need in needs if not present.get(need)]
            plan.append({"stage": name, "paid": False, "missing": missing})
        if not args.offline:
            for name, needs in PAID_STAGES:
                missing = [need for need in needs if not present.get(need)]
                if name == "judge-head-to-head" and not args.judge:
                    missing = [*missing, "judge"]
                plan.append({"stage": name, "paid": True, "missing": missing})

        runnable = [entry for entry in plan if not entry["missing"]]
        paid = [entry for entry in runnable if entry["paid"]]
        if paid and not approved(args):
            self._price(args, paid, base)
            return 0

        done = []
        for entry in plan:
            if entry["missing"]:
                done.append(
                    {
                        "stage": entry["stage"],
                        "ran": False,
                        "why": "nothing was given for " + ", ".join(entry["missing"]),
                    }
                )
                continue
            arguments = self._arguments(entry["stage"], args, base)
            outcome = _call(entry["stage"], arguments, approve=approved(args))
            done.append({"stage": entry["stage"], **outcome})

        summary = {
            "wrote_into": str(base),
            "ran": [entry["stage"] for entry in done if entry.get("ran")],
            "did_not_run": {
                entry["stage"]: entry.get("why") or f"status {entry.get('status')}"
                for entry in done
                if not entry.get("ran")
            },
        }
        write_json(base / "evaluation_run.json", {"stages": done, **summary})
        report(summary)
        return 0

    # ----------------------------------------------------------------- helpers
    def _inputs(self, args: argparse.Namespace) -> dict[str, Any]:
        """Which inputs were given. The systems table is written by an earlier stage."""
        return {
            "predictions": bool(args.predictions),
            "moments": bool(args.moments),
            "pairs": bool(args.pairs),
            "scores": bool(args.scores),
            "subscores": bool(args.subscores),
            "labels": bool(args.labels),
            "answers": bool(args.answers),
            "per_moment": bool(args.per_moment),
            "head_to_head": bool(args.head_to_head),
            # The correlations read what score-systems writes, so they can run
            # whenever that stage can.
            "systems_table": bool(args.predictions and args.moments),
        }

    def _arguments(
        self, stage: str, args: argparse.Namespace, base: Path
    ) -> list[str]:
        """The arguments one stage is called with, and where it writes."""
        shared = ["--overwrite"] if args.overwrite else []

        def paths(given: Sequence[Path]) -> list[str]:
            return [str(path) for path in given]

        if stage == "score-systems":
            return [
                "--predictions", *paths(args.predictions),
                "--moments", str(args.moments),
                *(["--references", str(args.references)] if args.references else []),
                "--metrics", *args.metrics,
                "--out", str(base / "systems.md"),
                *shared,
            ]
        if stage == "diagnose":
            return [
                "--predictions", str(args.predictions[0]),
                "--moments", str(args.moments),
                "--mode", args.mode,
                "--out", str(base / f"rules_broken_{args.mode}.md"),
                *shared,
            ]
        if stage == "reference-writer-compliance":
            # Only the compliance modes: a writer is scored against the references, and
            # it wrote one of them, so an overlap metric would be scoring it on itself.
            modes = [name for name in args.metrics if name.startswith("compliance:")]
            return [
                "--moments", str(args.moments),
                *(["--references", str(args.references)] if args.references else []),
                *(["--metrics", *modes] if modes else []),
                "--out", str(base / "reference_writers.json"),
            ]
        if stage == "rank-stability":
            return [
                "--per-moment", str(args.per_moment),
                "--out", str(base / "rank_stability.json"),
            ]
        if stage == "metric-correlations":
            return [
                "--table", str(base / "systems.json"),
                "--out", str(base / "correlations.json"),
            ]
        if stage == "pair-validity":
            return [
                "--pairs", str(args.pairs), "--scores", str(args.scores),
                "--out", str(base / "pair_validity.json"),
            ]
        if stage == "localisation":
            return [
                "--pairs", str(args.pairs), "--subscores", str(args.subscores),
                "--out", str(base / "localisation.json"),
            ]
        if stage == "rater-agreement":
            return [
                "--pairs", str(args.pairs),
                "--labels", *paths(args.labels),
                "--scores", str(args.scores),
                "--out", str(base / "rater_agreement.json"),
            ]
        if stage == "head-to-head-win-rate":
            return [
                "--answers", str(args.answers),
                "--first", "first", "--second", "second",
                "--judge", args.judge or "unnamed",
                "--out", str(base / "win_rate.json"),
            ]
        if stage == "score-rubric":
            return ["--predictions", str(args.predictions[0]), "--out", str(base / "rubric.json")]
        if stage == "rate-against-references":
            return [
                "--predictions", str(args.predictions[0]),
                "--out", str(base / "reference_rating.json"),
            ]
        if stage == "judge-head-to-head":
            return [
                "--moments", str(args.head_to_head), "--judge", args.judge,
                "--out", str(base / "head_to_head_answers.json"),
            ]
        raise AdForEduError(f"the suite has no arguments for stage {stage!r}")

    def _price(
        self, args: argparse.Namespace, paid: Sequence[dict[str, Any]], base: Path
    ) -> None:
        """Print one estimate over every paid stage, and write it beside the outputs.

        Each stage is asked for its own estimate without approval, which makes it
        print and stop. Their totals are summed here so that the decision is taken
        once rather than per stage.
        """
        from ...llm import ModelCatalog, PriceTable

        models = ModelCatalog.load()
        prices = PriceTable.load(models)
        judge = args.judge or "gpt-5.5"
        described = len(read_jsonl(args.predictions[0])) if args.predictions else 0
        drawn = len(read_jsonl(args.head_to_head)) if args.head_to_head else 0
        wanted = {
            "score-rubric": ("score_rubric", {"described": {judge: described}}),
            "rate-against-references": (
                "rate_against_references",
                {"described": {judge: described}},
            ),
            "judge-head-to-head": (
                "judge_head_to_head",
                {"both orders": {judge: drawn * 2}},
            ),
        }
        total, lines = 0.0, []
        for entry in paid:
            stage, calls = wanted[entry["stage"]]
            estimate = prices.estimate(stage, calls)
            total += estimate.total
            lines.append(
                {"stage": entry["stage"], "calls": estimate.calls, "dollars": estimate.total}
            )
        report(
            {
                "would_run": [entry["stage"] for entry in paid],
                "per_stage": lines,
                "total_dollars": round(total, 2),
                "note": (
                    "no --approve-spend: nothing was called, and no client was built. "
                    "Re-run with the flag to spend, or with --offline to run only the "
                    "stages that cost nothing."
                ),
            },
            path=base / "spend_estimate.json",
        )
