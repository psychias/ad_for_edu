"""Commands that ask models to write, and commands that ask judges to decide.

Every one of them prices the work per row and per model and stops, unless it is
told to spend. The call loop itself runs where the recordings are: the material a
writer or a judge is shown cannot be sent from a machine that does not hold it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ...core.io import read_jsonl
from ...core.spend import require_approval
from ...llm import LLMClient, ModelCatalog, PriceTable
from ..common import COMMANDS, Command, approved, report


def priced(stage: str, calls: dict[str, dict[str, int]], free: tuple[str, ...] = ()):
    models = ModelCatalog.load()
    prices = PriceTable.load(models)
    return models, prices.estimate(stage, calls, free_columns=free)


@COMMANDS.register("generate-references")
class GenerateReferences(Command):
    name = "generate-references"
    help = "ask every writer for a description of every moment, or for silence"
    paid = True

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--moments", type=Path, required=True)

    def run(self, args: argparse.Namespace) -> int:
        from ...references.build import load_reference_settings

        settings = load_reference_settings()
        moments = read_jsonl(args.moments)
        by_lecture: dict[str, int] = {}
        for row in moments:
            lecture = str(row.get("lecture") or str(row.get("id", "")).split("#")[0])
            by_lecture[lecture] = by_lecture.get(lecture, 0) + 1
        calls = {
            lecture: {writer: count for writer in settings.writers}
            for lecture, count in by_lecture.items()
        }
        models, estimate = priced("generate_references", calls)
        approval = require_approval(estimate, approved(args))
        client = LLMClient(approval, models)
        report(
            {
                "moments": len(moments),
                "writers": list(settings.writers),
                "calls": estimate.calls,
                "note": (
                    "the moments of a lecture are written in order, because the prompt for "
                    "one names what the earlier ones described; run this where the keyframes are"
                ),
                "client": type(client).__name__,
            },
            path=args.out,
        )
        return 0


@COMMANDS.register("generate-candidates")
class GenerateCandidates(Command):
    name = "generate-candidates"
    help = "ask for several descriptions of one moment, to pair against each other"
    paid = True

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--moments", type=Path, required=True)

    def run(self, args: argparse.Namespace) -> int:
        from ...pairs import build_candidate_mode, load_candidate_settings

        settings = load_candidate_settings()
        mode = build_candidate_mode(settings)
        moments = read_jsonl(args.moments)
        models, estimate = priced(
            "generate_candidates",
            {"all moments": {settings.model: len(moments) * len(settings.directives)}},
        )
        approval = require_approval(estimate, approved(args))
        client = LLMClient(approval, models)
        report(
            {
                "moments": len(moments),
                "directives": list(settings.directives),
                "descriptions_per_call": mode.per_call,
                "calls": estimate.calls,
                "client": type(client).__name__,
            },
            path=args.out,
        )
        return 0


@COMMANDS.register("generate-controlled-pairs")
class GenerateControlledPairs(Command):
    name = "generate-controlled-pairs"
    help = "ask for pairs that differ on exactly one rule category"
    paid = True

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--moments", type=Path, required=True)

    def run(self, args: argparse.Namespace) -> int:
        from ...pairs import build_axes, load_controlled_settings

        settings = load_controlled_settings()
        axes = build_axes(settings)
        moments = read_jsonl(args.moments)
        calls = 0
        per_axis = {}
        for axis in axes:
            eligible = [row for row in moments if axis.applies_to(row.get("type"))]
            wanted = min(settings.per_axis, len(eligible)) * axis.pairs_per_moment
            per_axis[axis.category] = {"eligible_moments": len(eligible), "pairs": wanted}
            calls += wanted
        models, estimate = priced(
            "generate_controlled_pairs", {"all categories": {settings.model: calls}}
        )
        approval = require_approval(estimate, approved(args))
        client = LLMClient(approval, models)
        report(
            {"per_category": per_axis, "calls": calls, "client": type(client).__name__},
            path=args.out,
        )
        return 0


@COMMANDS.register("judge-pairs")
class JudgePairs(Command):
    name = "judge-pairs"
    help = "order the pairs, judged in both presentation orders by a judge that watches"
    paid = True

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--pairs", type=Path, required=True)

    def run(self, args: argparse.Namespace) -> int:
        from ...pairs import load_judging_settings
        from ...pairs.ordering import ORDERS

        settings = load_judging_settings()
        pairs = read_jsonl(args.pairs)
        models, estimate = priced(
            "judge_pairs", {"both orders": {settings.model: len(pairs) * len(ORDERS)}}
        )
        approval = require_approval(estimate, approved(args))
        client = LLMClient(approval, models)
        report(
            {
                "pairs": len(pairs),
                "calls": estimate.calls,
                "judge": settings.model,
                "clip_seconds": settings.clip_seconds,
                "combination": settings.combination.name,
                "client": type(client).__name__,
            },
            path=args.out,
        )
        return 0


@COMMANDS.register("judge-head-to-head")
class JudgeHeadToHead(Command):
    name = "judge-head-to-head"
    help = "ask a judge to choose between two systems on the drawn moments, both orders"
    paid = True

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--moments", type=Path, required=True)
        parser.add_argument("--judge", required=True, help="a model that may annotate")

    def run(self, args: argparse.Namespace) -> int:
        from ...pairs.ordering import ORDERS

        moments = read_jsonl(args.moments)
        models, estimate = priced(
            "judge_head_to_head", {"both orders": {args.judge: len(moments) * len(ORDERS)}}
        )
        models.get(args.judge).require_role("annotator")
        approval = require_approval(estimate, approved(args))
        client = LLMClient(approval, models)
        report(
            {
                "moments": len(moments),
                "calls": estimate.calls,
                "judge": args.judge,
                "client": type(client).__name__,
            },
            path=args.out,
        )
        return 0


@COMMANDS.register("score-rubric")
class ScoreRubric(Command):
    name = "score-rubric"
    help = "grade each description in the six rule categories and overall"
    paid = True

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--predictions", type=Path, required=True)
        parser.add_argument("--model", default="gpt-5.5")

    def run(self, args: argparse.Namespace) -> int:
        from ...inference import read_predictions

        predictions = read_predictions(args.predictions)
        described = [prediction for prediction in predictions if prediction.filled]
        models, estimate = priced("score_rubric", {"described": {args.model: len(described)}})
        models.get(args.model).require_role("metric_judge")
        approval = require_approval(estimate, approved(args))
        client = LLMClient(approval, models)
        report(
            {
                "moments": len(predictions),
                "described": len(described),
                "calls": estimate.calls,
                "client": type(client).__name__,
            },
            path=args.out,
        )
        return 0


@COMMANDS.register("rate-against-references")
class RateAgainstReferences(Command):
    name = "rate-against-references"
    help = "rate each description against the references of its moment"
    paid = True

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--predictions", type=Path, required=True)
        parser.add_argument("--model", default="gpt-5.5")

    def run(self, args: argparse.Namespace) -> int:
        from ...inference import read_predictions

        predictions = read_predictions(args.predictions)
        described = [prediction for prediction in predictions if prediction.filled]
        models = ModelCatalog.load()
        entry = models.get(args.model)
        entry.require_role("metric_judge")
        if entry.free:
            report(
                {
                    "moments": len(described),
                    "model": args.model,
                    "note": "this model runs on this machine and costs nothing",
                },
                path=args.out,
            )
            return 0
        prices = PriceTable.load(models)
        estimate = prices.estimate(
            "rate_against_references", {"described": {args.model: len(described)}}
        )
        approval = require_approval(estimate, approved(args))
        client = LLMClient(approval, models)
        report(
            {
                "moments": len(described),
                "calls": estimate.calls,
                "client": type(client).__name__,
            },
            path=args.out,
        )
        return 0


@COMMANDS.register("annotate-rated-pairs")
class AnnotateRatedPairs(Command):
    name = "annotate-rated-pairs"
    help = "let a model stand beside the raters on the same pairs and the same evidence"
    paid = True

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--pairs", type=Path, required=True)
        parser.add_argument("--model", required=True, help="a model that may annotate")

    def run(self, args: argparse.Namespace) -> int:
        from ...pairs.ordering import ORDERS

        pairs = read_jsonl(args.pairs)
        models, estimate = priced(
            "annotate_rated_pairs", {"both orders": {args.model: len(pairs) * len(ORDERS)}}
        )
        models.get(args.model).require_role("annotator")
        approval = require_approval(estimate, approved(args))
        client = LLMClient(approval, models)
        report(
            {
                "pairs": len(pairs),
                "calls": estimate.calls,
                "annotator": args.model,
                "client": type(client).__name__,
            },
            path=args.out,
        )
        return 0
