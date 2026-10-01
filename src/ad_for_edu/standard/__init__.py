"""The rule book, its applicability table and the per-rule judge prompts."""

from .applicability import Applicability, ApplicabilityTable
from .rulebook import Rule, RuleBook
from .templates import WITHOUT_PROMPT, bind_moment, render_all, render_rule_prompt


def check_standard() -> dict[str, object]:
    """Load the shipped standard, check it, and return its summary."""
    book = RuleBook.load()
    book.check()
    table = ApplicabilityTable.load()
    table.check_against(book)
    prompts = render_all(book)
    summary = book.summary()
    summary["judge_prompts"] = len(prompts)
    summary["without_prompt"] = sorted(WITHOUT_PROMPT)
    return summary


__all__ = [
    "WITHOUT_PROMPT",
    "Applicability",
    "ApplicabilityTable",
    "Rule",
    "RuleBook",
    "bind_moment",
    "check_standard",
    "render_all",
    "render_rule_prompt",
]
