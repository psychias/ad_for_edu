"""The shared prefix of the writing and judging prompts.

The prefix states the task and the operating principles and carries the rule
book verbatim. It is a function of two texts only, the prefix body and the rule
book, so it is identical across every call of a run and can be served from a
prompt cache. The classifier does not receive it: it decides what an event is,
not whether to describe it, and must not see the standard.
"""

from __future__ import annotations

from ..core.errors import ContractError
from ..standard.rulebook import RuleBook
from .templates import read_text

PRINCIPLES_HEADING = "\n\nOPERATING PRINCIPLES"
RULE_BOOK_NAME = "rules_for_slides.yaml"


def shared_prefix(book: RuleBook | None = None) -> str:
    """The prefix with the rule book spliced in between the task and the principles."""
    book = book or RuleBook.load()
    body = read_text("shared_prefix")
    head, separator, tail = body.partition(PRINCIPLES_HEADING)
    if not separator:
        raise ContractError("the prefix body has no OPERATING PRINCIPLES section")
    return (
        f"{head}\n\n[{RULE_BOOK_NAME} — {len(book)} rules, verbatim]\n"
        f"{book.text}{PRINCIPLES_HEADING}{tail}"
    )
