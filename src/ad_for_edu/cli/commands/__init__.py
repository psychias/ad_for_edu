"""The commands. Importing this package is what puts them in the registry.

Every module is imported here rather than found by scanning, so a command that fails
to import is an error at start-up instead of a name that quietly does not exist.
"""

from . import (  # noqa: F401
    agreement,
    describe,
    evaluate,
    generate,
    inspect,
    media,
    moments,
    pairs,
    training,
)

MODULES = (
    "inspect",
    "media",
    "moments",
    "generate",
    "pairs",
    "training",
    "describe",
    "evaluate",
    "agreement",
)
