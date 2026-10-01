"""Commands that turn the corpus into training material, and set up a cell of the grid."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ...core.io import read_jsonl, write_jsonl
from ...core.outputs import guard_new
from ...core.runs import RunKey
from ...core.settings import work_dir
from ...data.catalog import DataCatalog
from ..common import COMMANDS, Command, output_path, report


def cell_of(key: RunKey) -> dict[str, Any]:
    return {
        "backbone": key.backbone,
        "arm": key.arm,
        "method": key.method,
        "seed": key.seed,
        "row": key.row,
        "name": key.name,
    }


@COMMANDS.register("build-training-examples")
class BuildTrainingExamples(Command):
    name = "build-training-examples"
    help = "the reference rows as conversations, one file per input arm and side"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--references",
            type=Path,
            default=None,
            help="the reference rows; taken from the dataset when not given",
        )
        parser.add_argument(
            "--arms",
            nargs="*",
            default=None,
            help="which input arms to write; all of the registered ones by default",
        )
        parser.add_argument(
            "--split",
            default=None,
            help="which split policy decides the side of a lecture",
        )

    def run(self, args: argparse.Namespace) -> int:
        from ...data.build import build_split_policy
        from ...data.schema import ReferenceRow
        from ...training import INPUT_ARMS, build_arm, example_for

        catalog = DataCatalog.load()
        source = args.references or catalog.dataset_file("reference_rows")
        rows = [ReferenceRow.from_row(row) for row in read_jsonl(source)]
        policy = build_split_policy(catalog, args.split)
        known = list(INPUT_ARMS.names())
        arms = list(args.arms) if args.arms else known
        unknown = [name for name in arms if name not in known]
        if unknown:
            raise SystemExit(f"unknown input arms {unknown}; known: {known}")

        base = output_path(args, "examples", "examples").parent
        written = {}
        for name in arms:
            arm = build_arm(name)
            per_side: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                side = policy.side_of_row(row.to_row())
                per_side.setdefault(side, []).append(example_for(row, arm))
            for side, examples in sorted(per_side.items()):
                suffix = "" if name == "multimodal" else f"_{name}"
                target = guard_new(
                    base / f"examples_{side}{suffix}.jsonl", overwrite=args.overwrite
                )
                write_jsonl(target, examples)
                written[f"{name}/{side}"] = {"path": str(target), "examples": len(examples)}
        report(
            {
                "read": str(source),
                "rows": len(rows),
                "written": written,
                "note": (
                    "the side of a lecture comes from the split policy, which holds out "
                    "whole courses; no lecture is named here"
                ),
            }
        )
        return 0


@COMMANDS.register("train")
class Train(Command):
    name = "train"
    help = "set up and run one cell of the grid: a backbone, an arm, a method and a seed"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--backbone", required=True)
        parser.add_argument("--arm", required=True)
        parser.add_argument("--method", required=True)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument(
            "--examples",
            type=Path,
            default=None,
            help="the training material; taken from the dataset when not given",
        )
        parser.add_argument(
            "--settings-only",
            action="store_true",
            help="print the layered settings of the cell and stop, without reading material",
        )
        parser.add_argument(
            "--smoke",
            action="store_true",
            help="write into a separate run directory, so a trial cannot overwrite a run",
        )

    def run(self, args: argparse.Namespace) -> int:
        from ...core.runs import write_record
        from ...data.build import build_split_policy
        from ...training import build_arm, build_method, load_grid, prepare, settings_of

        key = RunKey(args.backbone, args.arm, args.method, args.seed)
        grid = load_grid()
        settings, lora, backbone = settings_of(key)
        held = grid.development_for(key.method)
        if args.settings_only:
            report(
                {
                    "cell": cell_of(key),
                    "backbone": backbone.model_id,
                    "settings": dict(settings.__dict__),
                    "lora": dict(lora.__dict__),
                    "development": held,
                },
                path=args.out,
            )
            return 0

        catalog = DataCatalog.load()
        role = "training_pairs" if key.method == "dpo" else "train_examples"
        source = args.examples or catalog.dataset_file(role)
        rows = read_jsonl(source)
        data = prepare(
            rows,
            build_split_policy(catalog),
            share=float(held["share"]),
            seed=int(held.get("seed", key.seed)),
            by=str(held.get("by", "moment")),
        )
        arm = build_arm(key.arm)
        method = build_method(key.method)
        target = key.directory(work_dir(), smoke=args.smoke)
        write_record(
            target,
            key,
            {
                **dict(settings.__dict__),
                "lora": dict(lora.__dict__),
                "model_id": backbone.model_id,
            },
            extra={"material": {"read": str(source), **data.sizes}},
        )
        report(
            {
                "cell": cell_of(key),
                "read": str(source),
                "sizes": data.sizes,
                "arm": type(arm).__name__,
                "method": type(method).__name__,
                "run_dir": str(target),
                "note": (
                    "the fit itself runs where the accelerators are; this command checks the "
                    "settings, takes the development slice and opens the run"
                ),
            },
            path=args.out,
        )
        return 0
