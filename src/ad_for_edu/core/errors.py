"""Error types raised across the package.

Each error also derives from the built-in exception a caller would expect, so
`except LookupError` or `except ValueError` keeps working.
"""

from __future__ import annotations


class AdForEduError(Exception):
    """Base class of every error this package raises on purpose."""


class UnknownStrategyError(AdForEduError, LookupError):
    """A strategy name that is not registered in its family."""


class DuplicateStrategyError(AdForEduError, ValueError):
    """A strategy name registered twice in one family."""


class StrategyConfigError(AdForEduError, ValueError):
    """Parameters that a strategy's constructor does not accept."""


class SettingsError(AdForEduError, ValueError):
    """A settings file with an unknown key, a missing key, or an unresolved variable."""


class MissingSourceError(AdForEduError, FileNotFoundError):
    """A per-lecture source file that a stage depends on is absent."""


class ProcessError(AdForEduError, RuntimeError):
    """An external program failed."""


class ContractError(AdForEduError, ValueError):
    """Data that breaks a stated contract: wrong ids, wrong count, value outside a closed set."""


class SpendNotApprovedError(AdForEduError, RuntimeError):
    """A paid client was requested without an approval token."""


class OutputExistsError(AdForEduError, FileExistsError):
    """A stage was asked to write over an output that already exists."""
