"""What every command shares.

A command is a small object: a name, a line of help, the arguments it takes, and
what it does. It is registered by name, so the list of commands and the parser are
built from one place and cannot disagree.

Every command that would call a hosted model prints what it would cost and stops,
unless it is told to spend. That is not a convention each command keeps: the
client cannot be built without the token that asking produces.
"""

from __future__ import annotations

import argparse
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from ..core.registry import Registry
from ..core.settings import work_dir
from ..core.spend import APPROVE_DEST, APPROVE_FLAG


class Command(ABC):
    """One thing the program can be asked to do."""

    name: ClassVar[str]
    help: ClassVar[str]
    #: Whether it can call a hosted model.
    paid: ClassVar[bool] = False

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:  # noqa: B027
        """The arguments beyond the shared ones.

        Not abstract, and deliberately so: a command that takes only the shared
        options should not have to declare an empty method to say so.
        """

    @abstractmethod
    def run(self, args: argparse.Namespace) -> int: ...


COMMANDS: Registry[Command] = Registry("command", Command)


def add_shared(parser: argparse.ArgumentParser, *, paid: bool) -> None:
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="where the settings files are; the shipped ones by default",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where to write; under the work directory by default",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="write over an output that already exists, instead of refusing",
    )
    if paid:
        parser.add_argument(
            APPROVE_FLAG,
            dest=APPROVE_DEST,
            action="store_true",
            help=(
                "make the calls. Without it the command prints what it would cost, "
                "per row and per model, and stops without building a client."
            ),
        )


def approved(args: argparse.Namespace) -> bool:
    return bool(getattr(args, APPROVE_DEST, False))


def output_path(args: argparse.Namespace, role: str, name: str) -> Path:
    """Where a command writes: what it was told, or the place for its stage."""
    if args.out is not None:
        return args.out
    return work_dir() / role / name


def report(value: Any, *, path: Path | None = None) -> None:
    """Print a result, and write it where asked."""
    text = value if isinstance(value, str) else json.dumps(value, indent=1, ensure_ascii=False)
    print(text)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


def counted(label: str, rows: Sequence[Any] | Mapping[Any, Any]) -> str:
    return f"{label}: {len(rows)}"
