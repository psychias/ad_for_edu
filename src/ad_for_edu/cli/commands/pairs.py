"""Commands that build the pairs and order them."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from ...core.io import read_jsonl, write_jsonl
from ...core.outputs import guard_new
from ..common import COMMANDS, Command, output_path, report


@COMMANDS.register("build-pairs")
class BuildPairs(Command):
    name = "build-pairs"
    help = "pair the candidate descriptions of each moment against one another"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--candidates", type=Path, required=True)

    def run(self, args: argparse.Namespace) -> int:
        from ...metrics import CharacterOverlap
        from ...pairs import Offered, load_pair_set_settings, pairs_of_moment

        settings = load_pair_set_settings()
        overlap = CharacterOverlap()
        by_moment: dict[str, list[Offered]] = {}
        for row in read_jsonl(args.candidates):
            by_moment.setdefault(str(row["moment_id"]), []).append(
                Offered(
                    text=str(row["ad_text"]),
                    writer=str(row.get("family") or row.get("writer") or ""),
                    focus=row.get("directive"),
                    rung=row.get("rung"),
                )
            )
        generator = random.Random(settings.pairing_seed)
        rows = []
        for moment, offered in sorted(by_moment.items()):
            for index, pairing in enumerate(
                pairs_of_moment(
                    offered,
                    overlap.one,
                    generator,
                    cap=settings.maximum_per_moment,
                    minimum_distance=settings.minimum_distance,
                    per_description=settings.maximum_per_description,
                ),
                start=1,
            ):
                rows.append(
                    {
                        "pair_id": f"{moment}@{index}",
                        "moment_id": moment,
                        "lecture": moment.split("#")[0],
                        "a": pairing.first.text,
                        "b": pairing.second.text,
                        "a_family": pairing.first.writer,
                        "b_family": pairing.second.writer,
                        "distance": round(pairing.distance, 3),
                        "cross_family": pairing.across_writers,
                        "source": "natural",
                    }
                )
        target = guard_new(output_path(args, "pair_sets", "pairs.jsonl"), overwrite=args.overwrite)
        write_jsonl(target, rows)
        report({"written": str(target), "pairs": len(rows), "moments": len(by_moment)})
        return 0


@COMMANDS.register("draw-rating-pool")
class DrawRatingPool(Command):
    name = "draw-rating-pool"
    help = "draw the set of pairs people rate, spread over the categories and lectures"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--pairs", type=Path, required=True)

    def run(self, args: argparse.Namespace) -> int:
        from ...pairs import draw, load_pair_set_settings

        settings = load_pair_set_settings()
        rows = read_jsonl(args.pairs)
        by_kind: dict[str, list] = {}
        for row in rows:
            kind = str(row.get("stratum") or row.get("source") or "natural")
            by_kind.setdefault(kind, []).append(row)
        drawn = draw(by_kind, settings.rated_pool)
        flat = [row for kind in drawn.values() for row in kind]
        target = guard_new(
            output_path(args, "pair_sets", "rated_pool.jsonl"), overwrite=args.overwrite
        )
        write_jsonl(target, flat)
        report(
            {
                "written": str(target),
                "pairs": len(flat),
                "per_kind": {kind: len(rows) for kind, rows in drawn.items()},
                "moments": len({row["moment_id"] for row in flat}),
            }
        )
        return 0


@COMMANDS.register("hold-out-dev-pairs")
class HoldOutDevelopmentPairs(Command):
    name = "hold-out-dev-pairs"
    help = "split the training pairs by lecture, holding some out"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--pairs", type=Path, required=True)

    def run(self, args: argparse.Namespace) -> int:
        from ...pairs import hold_out_by_lecture, load_pair_set_settings

        settings = load_pair_set_settings()
        rows = read_jsonl(args.pairs)
        kept, held = hold_out_by_lecture(
            rows, settings.development_share, settings.development_seed
        )
        base = output_path(args, "pair_sets", "training_pairs.jsonl").parent
        train = guard_new(base / "training_pairs.jsonl", overwrite=args.overwrite)
        development = guard_new(base / "development_pairs.jsonl", overwrite=args.overwrite)
        write_jsonl(train, kept)
        write_jsonl(development, held)
        report(
            {
                "training": {"path": str(train), "pairs": len(kept)},
                "development": {"path": str(development), "pairs": len(held)},
                "lectures_held_out": sorted({str(row.get("lecture")) for row in held}),
            }
        )
        return 0


@COMMANDS.register("draw-head-to-head")
class DrawHeadToHead(Command):
    name = "draw-head-to-head"
    help = "draw the moments two systems are compared on"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--moments", type=Path, required=True)

    def run(self, args: argparse.Namespace) -> int:
        from ...evaluation import draw_moments
        from ...pairs import load_pair_set_settings

        settings = load_pair_set_settings()
        plan = settings.head_to_head
        available = sorted({str(row["moment_id"]) for row in read_jsonl(args.moments)})
        drawn = draw_moments(available, int(plan.get("moments", 150)), int(plan.get("seed", 17)))
        target = guard_new(
            output_path(args, "pair_sets", "head_to_head.jsonl"), overwrite=args.overwrite
        )
        write_jsonl(target, [{"moment_id": moment} for moment in drawn])
        report(
            {
                "written": str(target),
                "moments": len(drawn),
                "lectures": len({moment.split("#")[0] for moment in drawn}),
            }
        )
        return 0
