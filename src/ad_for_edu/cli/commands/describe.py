"""The command that asks a system to describe the moments of the evaluation set.

Every system is asked about every moment, whether or not it would have chosen to
speak. A system scored only on the moments it chose is scored on an easier set.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...core.errors import MissingSourceError
from ...core.io import read_jsonl
from ...core.outputs import guard_new
from ...data.catalog import DataCatalog
from ..common import COMMANDS, Command, output_path, report

#: How far either side of a moment a keyframe still shows what was on screen.
BEFORE, AFTER, MOST = 2.0, 5.0, 3


def eval_moments(
    rows: Sequence[Mapping[str, Any]], catalog: DataCatalog, *, with_pictures: bool = True
):
    """The moments of the evaluation set, with what a describer is shown about each.

    The pictures are attached only where they are: a machine that holds no
    recordings still builds the moments, and a describer that needs pictures is run
    where they are rather than sent them.
    """
    from ...preprocessing.build import media_layout
    from ...systems import EvalMoment

    layout = media_layout(catalog) if with_pictures else None
    frames: dict[str, tuple] = {}
    out = []
    for row in rows:
        moment_id = str(row.get("moment_id") or row.get("id"))
        lecture = str(row.get("lecture") or moment_id.split("#")[0])
        time = float(row.get("t") or row.get("time") or 0.0)
        pictures: tuple[Path, ...] = ()
        if layout is not None:
            if lecture not in frames:
                try:
                    frames[lecture] = layout.lecture(lecture).keyframes
                except MissingSourceError:
                    frames[lecture] = ()
            nearby = [
                frame.path
                for frame in frames[lecture]
                if time - BEFORE <= frame.time <= time + AFTER
            ]
            pictures = tuple(nearby[:MOST])
        out.append(
            EvalMoment(
                moment_id=moment_id,
                type=str(row.get("type") or ""),
                slide_ocr=str(row.get("slide_ocr") or ""),
                what_on_screen=str(row.get("what_on_screen") or ""),
                transcript_window=str(row.get("transcript_window") or ""),
                pause_after=float(row.get("pause_after") or 0.0),
                time=time,
                keyframes=pictures,
            )
        )
    return out


@COMMANDS.register("describe")
class Describe(Command):
    name = "describe"
    help = "ask one system for a description of every moment of the evaluation set"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--moments", type=Path, required=True)
        parser.add_argument(
            "--system",
            required=True,
            help="a registered system; run list-strategies to see them",
        )
        parser.add_argument(
            "--run-dir",
            type=Path,
            default=None,
            help="the run whose adapter is asked, for a trained describer",
        )
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument(
            "--references",
            type=Path,
            default=None,
            help="the reference rows, for the system that replays one writer's answers",
        )
        parser.add_argument("--writer", default=None, help="which writer's answers to replay")
        parser.add_argument(
            "--no-pictures",
            action="store_true",
            help="build the moments without keyframes, where the recordings are not held",
        )

    def run(self, args: argparse.Namespace) -> int:
        from ...inference import load_inference_settings, summarise, write_predictions
        from ...systems import DESCRIPTION_SYSTEMS, ReferenceWriter, SlideTitleReadout

        known = list(DESCRIPTION_SYSTEMS.names())
        if args.system not in known:
            raise SystemExit(f"unknown system {args.system!r}; known: {known}")
        catalog = DataCatalog.load()
        settings = load_inference_settings()
        rows = read_jsonl(args.moments)
        moments = eval_moments(rows, catalog, with_pictures=not args.no_pictures)

        if args.system == "slide_title":
            system = SlideTitleReadout()
        elif args.system == "reference_writer":
            if not args.writer:
                raise SystemExit("--writer says whose answers to replay")
            source = args.references or catalog.dataset_file("reference_rows")
            replayed = {
                str(row.get("moment_id") or str(row.get("output_id", "")).split("::")[0]): str(
                    row.get("ad_text") or ""
                )
                for row in read_jsonl(source)
                if str(row.get("family") or row.get("writer") or "") == args.writer
                and row.get("emit")
            }
            system = ReferenceWriter(args.writer, replayed)
        else:
            report(
                {
                    "system": args.system,
                    "moments": len(moments),
                    "with_pictures": sum(1 for moment in moments if moment.keyframes),
                    "checkpoint": settings.checkpoint,
                    "run_dir": str(args.run_dir) if args.run_dir else None,
                    "attempts": [spec.name for spec in settings.attempts],
                    "salvage": [spec.name for spec in settings.salvage],
                    "note": (
                        "a describer with weights is asked where those weights are; this "
                        "command reports the moments and how it would be asked"
                    ),
                },
                path=args.out,
            )
            return 0

        predictions = system.describe(moments)
        target = guard_new(
            output_path(args, "evaluation", f"predictions_{args.system}.jsonl"),
            overwrite=args.overwrite,
        )
        write_predictions(target, predictions)
        report({"written": str(target), **summarise(predictions)})
        return 0
