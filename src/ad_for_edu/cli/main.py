"""The entry point.

The parser is built from the command registry, so the list of commands and what the
program accepts cannot drift apart. A command that can spend money is given its
`--approve-spend` flag here, from one place, rather than each remembering to add it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from ..core.errors import AdForEduError
from ..core.settings import set_config_dir
from . import commands as _commands  # noqa: F401  (registers every command)
from .common import COMMANDS, add_shared

PROGRAM = "ad-for-edu"
DESCRIPTION = (
    "Audio description for slide-based lecture recordings: find the moments that need "
    "describing, write reference descriptions, build preference pairs, train a describer "
    "and score it against the rule book."
)
EPILOGUE = (
    "A command that can call a hosted model prints what the work would cost, per row and "
    "per model, and stops. Pass --approve-spend to let it make the calls."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=DESCRIPTION,
        epilog=EPILOGUE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")
    for name in COMMANDS.names():
        command = COMMANDS.create(name)
        under = subparsers.add_parser(
            name,
            help=command.help,
            description=command.__doc__ or command.help,
        )
        add_shared(under, paid=command.paid)
        command.add_arguments(under)
        under.set_defaults(_command=command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    command = getattr(args, "_command", None)
    if command is None:
        parser.print_help()
        return 2
    if args.config_dir is not None:
        set_config_dir(args.config_dir)
    try:
        return int(command.run(args))
    except AdForEduError as error:
        # The message says what is wrong and what the allowed values are; a traceback
        # of this package's own guards would bury that.
        print(f"{PROGRAM} {args.command}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
