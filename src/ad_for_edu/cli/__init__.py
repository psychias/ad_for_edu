"""The command-line program."""

from .common import COMMANDS, Command
from .main import build_parser, main

__all__ = ["COMMANDS", "Command", "build_parser", "main"]
