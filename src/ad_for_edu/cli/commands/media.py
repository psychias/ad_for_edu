"""Commands that prepare the recordings."""

from __future__ import annotations

import argparse

from ...core.io import write_json
from ...core.spend import SpendEstimate, require_approval
from ...data.catalog import DataCatalog
from ...data.manifest import LectureManifest
from ..common import COMMANDS, Command, approved, report


def lectures_of(args: argparse.Namespace, catalog: DataCatalog) -> LectureManifest:
    manifest = LectureManifest.load(catalog.local_path("lecture_manifest"))
    if getattr(args, "lectures", None):
        unknown = [name for name in args.lectures if name not in manifest]
        if unknown:
            raise SystemExit(f"not in the manifest: {unknown}")
    return manifest


def chosen(args: argparse.Namespace, manifest: LectureManifest) -> list[str]:
    if getattr(args, "lectures", None):
        return list(args.lectures)
    return [entry.lecture_id for entry in manifest]


@COMMANDS.register("preprocess")
class Preprocess(Command):
    name = "preprocess"
    help = "audio, transcript, speech gaps, keyframes and slide text for each lecture"
    paid = True

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--lectures", nargs="*", default=None)
        parser.add_argument(
            "--steps",
            default=None,
            help="a comma-separated subset of the steps; all of them by default",
        )

    def run(self, args: argparse.Namespace) -> int:
        from ...preprocessing import STEPS, build_preprocessor
        from ...preprocessing.build import media_layout

        catalog = DataCatalog.load()
        manifest = lectures_of(args, catalog)
        names = chosen(args, manifest)
        steps = tuple(args.steps.split(",")) if args.steps else STEPS
        preprocessor = build_preprocessor(catalog)

        if preprocessor.paid_steps(steps):
            estimate = SpendEstimate("preprocess", free_columns=("keyframes", "slide text", "gaps"))
            for name in names:
                # Transcription is priced by the length of the recording, not by tokens,
                # so the estimate names the lectures and the service prices them.
                estimate.add(name, "transcription", 1, 0.0, measured=False,
                             note="priced by the length of the recording")
            require_approval(estimate, approved(args))

        layout = media_layout(catalog)
        done = {}
        for name in names:
            video = layout.lecture(name).video
            if video is None:
                done[name] = "no recording"
                continue
            done[name] = {
                result.step: result.detail for result in preprocessor.run(name, video, steps)
            }
        report(done, path=args.out)
        return 0


@COMMANDS.register("probe-cursor")
class ProbeCursor(Command):
    name = "probe-cursor"
    help = "measure whether each recording draws a pointer, from its keyframes"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--lectures", nargs="*", default=None)
        parser.add_argument(
            "--samples",
            type=int,
            default=40,
            help="points sampled per lecture, away from the candidate events",
        )

    def run(self, args: argparse.Namespace) -> int:
        from ...preprocessing.build import media_layout
        from ...preprocessing.cursor import load_grey, probe_frames, renders_cursor

        catalog = DataCatalog.load()
        manifest = lectures_of(args, catalog)
        layout = media_layout(catalog)
        facts = {}
        for name in chosen(args, manifest):
            frames = layout.lecture(name).keyframes
            step = max(1, len(frames) // max(1, args.samples))
            sampled = frames[::step][: args.samples]
            verdicts = []
            for earlier, later in zip(sampled, sampled[1:], strict=False):
                verdicts.append(
                    probe_frames([load_grey(earlier.path), load_grey(later.path)]).verdict()
                )
            facts[name] = {
                "draws_pointer": renders_cursor(verdicts),
                "sampled": len(verdicts),
            }
        path = args.out or (catalog.local_path("lecture_manifest").parent / "pointer.json")
        write_json(path, {name: found["draws_pointer"] for name, found in facts.items()})
        report({"written": str(path), "lectures": facts})
        return 0
