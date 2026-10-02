"""The commands that compare the metrics with the people who rated the pairs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ...core.io import read_jsonl, write_json, write_jsonl
from ...core.outputs import guard_new
from ..common import COMMANDS, Command, output_path, report


def rater_labels(
    paths: list[Path], names: list[str] | None = None
) -> dict[str, dict[str, str]]:
    """One rater per file. A label that is not a side stays as it is.

    A tie is kept rather than dropped, because how often a rater declined to choose is
    part of what the comparison is about.

    The rater's name is what a report will print, so it has to be a pseudonym. Given
    explicitly it is used in the order the files were given; otherwise it is taken from
    the file name, which is then checked. A file named after the person who rated is
    refused rather than quietly turned into a column heading.
    """
    from ...agreement.raters import checked_rater_names

    if names is not None and len(names) != len(paths):
        raise SystemExit(
            f"{len(names)} rater names for {len(paths)} label files; give one name per file"
        )
    chosen = checked_rater_names(names if names is not None else [p.stem for p in paths])
    per_rater: dict[str, dict[str, str]] = {}
    for name, path in zip(chosen, paths, strict=True):
        per_rater[name] = {
            str(row["pair_id"]): str(row.get("side") or row.get("choice") or "")
            for row in read_jsonl(path)
        }
    return per_rater


@COMMANDS.register("rater-agreement")
class RaterAgreement(Command):
    name = "rater-agreement"
    help = "how often each metric prefers the side each rater chose, per rater and category"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--pairs", type=Path, required=True)
        parser.add_argument(
            "--labels",
            type=Path,
            nargs="+",
            required=True,
            help="one file per rater; its name is the rater unless --raters says",
        )
        parser.add_argument(
            "--raters",
            nargs="*",
            default=None,
            help="the pseudonym for each label file, in the order the files are given",
        )
        parser.add_argument(
            "--scores",
            type=Path,
            required=True,
            help="per pair and scorer, the number given to each side",
        )
        parser.add_argument(
            "--group-by",
            default="axis",
            help="the field that groups the pairs in the per-group table",
        )

    def run(self, args: argparse.Namespace) -> int:
        from ...agreement import build_report, render
        from ...stats import KrippendorffNominal
        from .evaluate import side_scores

        pairs = read_jsonl(args.pairs)
        labels = rater_labels(args.labels, args.raters)
        per_pair = side_scores(args.scores)
        scorers = sorted({name for scores in per_pair.values() for name in scores})
        by_scorer = {
            scorer: {
                pair_id: scores[scorer] for pair_id, scores in per_pair.items() if scorer in scores
            }
            for scorer in scorers
        }
        found = build_report(
            pairs,
            labels,
            by_scorer,
            coefficient=KrippendorffNominal(),
            group_by=args.group_by,
        )
        table = render(found, title="Metric agreement with the raters")
        print(table)
        target = output_path(args, "evaluation", "rater_agreement.json")
        write_json(target, found.as_dict())
        report(
            {
                "written": str(target),
                "raters": list(found.raters),
                "metrics": list(found.metrics),
                "agreement_between_raters": found.agreement,
                "note": (
                    "every cell is over one rater's own choices, so the cells of a column "
                    "rest on different numbers of pairs; the number is printed with each"
                ),
            }
        )
        return 0


@COMMANDS.register("fit-preference-head")
class FitPreferenceHead(Command):
    name = "fit-preference-head"
    help = "fit a small model to the raters' choices, predicted out of fold by lecture"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--pairs", type=Path, required=True)
        parser.add_argument(
            "--labels",
            type=Path,
            nargs="+",
            required=True,
            help="one file per rater; its name is the rater unless --raters says",
        )
        parser.add_argument(
            "--raters",
            nargs="*",
            default=None,
            help="the pseudonym for each label file, in the order the files are given",
        )
        parser.add_argument(
            "--features",
            type=Path,
            required=True,
            help="per pair and feature, the number on each side",
        )
        parser.add_argument(
            "--use",
            nargs="+",
            default=None,
            help="which features to fit on; every feature in the file by default",
        )
        parser.add_argument(
            "--keep",
            nargs="*",
            default=(),
            help="features kept even where the raters weight them in opposite directions",
        )
        parser.add_argument(
            "--hold-out-rater",
            default=None,
            help="fit without this rater and score it once, with the weights frozen",
        )
        parser.add_argument(
            "--held-at-equal",
            default=None,
            help="a feature to hold equal when asking what the head prefers",
        )

    def run(self, args: argparse.Namespace) -> int:
        from ...agreement.head import (
            FeaturedPair,
            freeze,
            out_of_fold,
            score_frozen,
            sign_stable,
        )

        pairs_rows = {str(row["pair_id"]): row for row in read_jsonl(args.pairs)}
        labels = rater_labels(args.labels, args.raters)
        featured: dict[str, FeaturedPair] = {}
        available: set[str] = set()
        for row in read_jsonl(args.features):
            pair_id = str(row["pair_id"])
            pair = pairs_rows.get(pair_id)
            if pair is None:
                continue
            first = {str(k): float(v) for k, v in (row.get("a") or {}).items() if v is not None}
            second = {str(k): float(v) for k, v in (row.get("b") or {}).items() if v is not None}
            available |= set(first) & set(second)
            featured[pair_id] = FeaturedPair(
                pair_id,
                str(pair.get("lecture") or pair_id.split("#")[0]),
                first,
                second,
            )
        names = list(args.use) if args.use else sorted(available)
        missing = [name for name in names if name not in available]
        if missing:
            raise SystemExit(f"no feature file carries {missing}; it has {sorted(available)}")

        stable, dropped = sign_stable(featured, labels, names, keep=tuple(args.keep))
        if args.held_at_equal and args.held_at_equal not in stable:
            # Holding a feature equal is a question about the fitted model, so the
            # feature has to be in it. The screen dropped it, and --keep is how a
            # feature is kept in spite of the screen.
            raise SystemExit(
                f"{args.held_at_equal!r} cannot be held equal: the raters weight it in "
                f"opposite directions, so the screen dropped it. Pass it to --keep to "
                f"fit on it anyway. Kept: {stable}"
            )
        fitted_on = {
            rater: sides
            for rater, sides in labels.items()
            if rater != args.hold_out_rater
        }
        found = out_of_fold(featured, fitted_on, stable)
        summary: dict[str, Any] = {
            "features_offered": names,
            "features_kept": stable,
            "features_dropped_as_idiosyncratic": dropped,
            "fitted_on_raters": sorted(fitted_on),
            **found.as_dict(),
        }

        per_rater = {}
        for rater, sides in sorted(fitted_on.items()):
            agreed = decided = 0
            for pair_id, side in sides.items():
                predicted = found.preferred.get(pair_id)
                if side not in ("a", "b") or predicted is None:
                    continue
                decided += 1
                agreed += predicted == side
            per_rater[rater] = {
                "n": decided,
                "agreed": agreed,
                "agreement": round(agreed / decided, 4) if decided else None,
            }
        summary["out_of_fold_agreement"] = per_rater

        if args.hold_out_rater:
            frozen = freeze(
                featured, labels, stable, held_out_rater=args.hold_out_rater
            )
            summary["held_out"] = {
                "rater": args.hold_out_rater,
                "weights": frozen.as_dict(),
                **score_frozen(frozen, featured, labels[args.hold_out_rater]),
            }

        if args.held_at_equal:
            counterfactual = {}
            head = found.head()
            for pair_id, pair in featured.items():
                difference = pair.difference(stable)
                if difference is None:
                    continue
                counterfactual[pair_id] = head.prefers(
                    head.held(difference, args.held_at_equal)
                )
            moved = sum(
                1
                for pair_id, side in counterfactual.items()
                if found.preferred.get(pair_id) not in (None, side)
            )
            summary["held_equal"] = {
                "feature": args.held_at_equal,
                "pairs": len(counterfactual),
                "changed_side": moved,
            }

        target = guard_new(
            output_path(args, "evaluation", "preference_head.jsonl"), overwrite=args.overwrite
        )
        write_jsonl(
            target,
            [
                {
                    "pair_id": pair_id,
                    "preferred": side,
                    "margin": round(found.margin[pair_id], 6),
                    "fold": found.fold_of[pair_id],
                }
                for pair_id, side in sorted(found.preferred.items())
            ],
        )
        write_json(target.with_suffix(".json"), summary)
        report({"written": str(target), **summary}, path=None)
        return 0
