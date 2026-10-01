"""Commands that find the moments of a lecture and say what each one is."""

from __future__ import annotations

import argparse
from pathlib import Path

from ...core.io import write_jsonl
from ...core.outputs import guard_new
from ...core.spend import require_approval
from ...data.catalog import DataCatalog
from ...llm import LLMClient, ModelCatalog, PriceTable
from ..common import COMMANDS, Command, approved, output_path, report
from .media import chosen, lectures_of


@COMMANDS.register("detect-moments")
class DetectMoments(Command):
    name = "detect-moments"
    help = "propose the candidate events of each lecture, from its artefacts alone"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--lectures", nargs="*", default=None)

    def run(self, args: argparse.Namespace) -> int:
        from ...moments import build_detector, load_detection_settings
        from ...preprocessing.build import media_layout

        catalog = DataCatalog.load()
        manifest = lectures_of(args, catalog)
        layout = media_layout(catalog)
        detector = build_detector(load_detection_settings())
        target = guard_new(
            output_path(args, "events", "candidates.jsonl"), overwrite=args.overwrite
        )
        rows, summaries = [], {}
        for name in chosen(args, manifest):
            found = detector.detect(layout.lecture(name))
            rows += [event.to_row() for event in found]
            summaries[name] = detector.summarise(found)
        write_jsonl(target, rows)
        report(
            {
                "written": str(target),
                "candidates": len(rows),
                "per_lecture": summaries,
            },
        )
        return 0


@COMMANDS.register("classify-moments")
class ClassifyMoments(Command):
    name = "classify-moments"
    help = "say what each candidate event is, or reject it"
    paid = True

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--events", type=Path, required=True)
        parser.add_argument(
            "--prepare-stills",
            action="store_true",
            help="extract the stills and stop, without calling anything",
        )

    def run(self, args: argparse.Namespace) -> int:
        from ...core.io import read_jsonl
        from ...data.schema import CandidateEvent
        from ...moments import (
            batch_by_lecture,
            build_still_policy,
            load_classification_settings,
        )
        from ...preprocessing.build import media_layout

        catalog = DataCatalog.load()
        settings = load_classification_settings()
        events = [CandidateEvent.from_row(row) for row in read_jsonl(args.events)]
        visual = [event for event in events if event.visual]
        layout = media_layout(catalog)
        stills = build_still_policy(settings.still_policy)

        if args.prepare_stills:
            made = 0
            for event in visual:
                video = layout.lecture(event.lecture).video
                if video is None:
                    continue
                made += len(
                    stills.stills(
                        event,
                        video,
                        catalog.output_path("stills") / event.lecture / event.id.split("#")[1],
                        width=settings.still_width,
                        font_file=settings.font_file,
                    )
                )
            report({"stills": made, "candidates": len(visual)})
            return 0

        models = ModelCatalog.load()
        prices = PriceTable.load(models)
        batches = batch_by_lecture(visual, settings.batch_size)
        estimate = prices.estimate(
            "classify_moments",
            {"all lectures": {settings.model: len(batches)}},
            free_columns=("stills",),
        )
        approval = require_approval(estimate, approved(args))
        client = LLMClient(approval, models)
        report(
            {
                "candidates": len(visual),
                "batches": len(batches),
                "model": settings.model,
                "note": (
                    "the call loop runs on the machine that holds the recordings; "
                    "prepare the stills there first with --prepare-stills"
                ),
                "client": type(client).__name__,
            },
            path=args.out,
        )
        return 0
