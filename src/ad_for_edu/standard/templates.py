"""The per-rule prompt of the compliance judge.

One template serves every rule. The rule's own fields are filled in here, once
per rule; the fields of the moment are left as tokens and filled by the judge
when it scores a description.

Three rules have no prompt, because a judge that reads one description cannot
decide them: one governs the decision to describe at all, one constrains how a
pair is constructed, and one is enforced by a deterministic check.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core.errors import ContractError
from .rulebook import RESOURCES, Rule, RuleBook

TEMPLATE = RESOURCES / "rule_compliance_template.txt"

WITHOUT_PROMPT = frozenset({"visual_content_001", "slide_text_003", "pair_perceptibility_001"})

#: Filled from the rule, when the prompt of a rule is rendered.
RULE_TOKENS: tuple[str, ...] = (
    "rule_id",
    "section",
    "element",
    "rule_type",
    "rule_body",
    "grounding",
)
#: Filled from the moment and the description, when the judge scores.
MOMENT_TOKENS: tuple[str, ...] = (
    "type",
    "what_on_screen",
    "transcript_window",
    "redundant",
    "pointing_clear",
    "emit",
    "ad_text",
)
#: Moment tokens that have no source in the context of a moment. They are bound to an
#: empty string, so the prompt states nothing about them.
UNSOURCED_TOKENS = frozenset({"redundant", "pointing_clear"})


def load_template() -> str:
    text = TEMPLATE.read_text(encoding="utf-8")
    absent = [name for name in (*RULE_TOKENS, *MOMENT_TOKENS) if "{" + name + "}" not in text]
    if absent:
        raise ContractError(f"the rule template lacks the tokens {absent}")
    return text


def render_rule_prompt(rule: Rule, template: str | None = None) -> str:
    """The prompt of one rule, with the moment tokens still open."""
    if rule.id in WITHOUT_PROMPT:
        raise ContractError(f"rule {rule.id!r} has no judge prompt by design")
    text = template if template is not None else load_template()
    values = {
        "rule_id": rule.id,
        "section": rule.section,
        "element": rule.element,
        "rule_type": rule.type,
        "rule_body": rule.text,
        "grounding": rule.grounding(),
    }
    for name in RULE_TOKENS:
        text = text.replace("{" + name + "}", values[name])
    return text


def render_all(book: RuleBook) -> dict[str, str]:
    """The prompt of every rule that has one, by rule id."""
    missing = WITHOUT_PROMPT - set(book.ids)
    if missing:
        raise ContractError(
            f"rules named as having no prompt are not in the book: {sorted(missing)}"
        )
    template = load_template()
    return {
        rule.id: render_rule_prompt(rule, template)
        for rule in book
        if rule.id not in WITHOUT_PROMPT
    }


def bind_moment(prompt: str, values: Mapping[str, Any]) -> str:
    """Fill the moment tokens of a rendered rule prompt.

    Every sourced token must be given; an unsourced one defaults to an empty string.
    """
    required = set(MOMENT_TOKENS) - UNSOURCED_TOKENS
    missing = sorted(required - set(values))
    if missing:
        raise ContractError(f"the rule prompt needs values for {missing}")
    unknown = sorted(set(values) - set(MOMENT_TOKENS))
    if unknown:
        raise ContractError(f"the rule prompt has no tokens {unknown}; it has {MOMENT_TOKENS}")
    bound = prompt
    for name in MOMENT_TOKENS:
        value = values.get(name, "")
        bound = bound.replace("{" + name + "}", "" if value is None else str(value))
    return bound
