"""Commands that look at the repository rather than at the data."""

from __future__ import annotations

import argparse

from ...core.registry import catalogue
from ..common import COMMANDS, Command, report


@COMMANDS.register("list-strategies")
class ListStrategies(Command):
    name = "list-strategies"
    help = "the interchangeable parts, by family, with the names a settings file can use"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--family", default=None, help="one family rather than all of them")

    def run(self, args: argparse.Namespace) -> int:
        # Importing the packages is what registers their parts.
        from ... import (  # noqa: F401
            compliance,
            data,
            evaluation,
            inference,
            judges,
            llm,
            metrics,
            moments,
            pairs,
            preprocessing,
            references,
            stats,
            systems,
            training,
        )
        from ...cli import commands  # noqa: F401
        from ...judges import pairwise, rating  # noqa: F401

        families = catalogue()
        if args.family is not None:
            if args.family not in families:
                report(f"no family {args.family!r}; known: {sorted(families)}")
                return 1
            families = {args.family: families[args.family]}
        lines = []
        for name, registry in families.items():
            lines.append(f"{name}:")
            for strategy in registry.names():
                lines.append(f"  {strategy}")
        report("\n".join(lines))
        return 0


@COMMANDS.register("check-standard")
class CheckStandard(Command):
    name = "check-standard"
    help = "read the rule book, check it against its table, and render every rule prompt"

    def run(self, args: argparse.Namespace) -> int:
        from ...standard import check_standard

        report(check_standard(), path=args.out)
        return 0
