"""The ways of asking a model for a reference description.

Every one of them asks the same three things in order: what of the visual the
lecturer's words already convey, whether a description is needed at all, and, if
it is, the description itself within the words the delivery rung allows.

They differ in the evidence they read and in what they must state about it.

    fixed_window          a stated stretch of speech before and after the moment
    next_event_window     the stretch runs to the next event, since a lecturer may
                          name what is on screen well after it appears
    visual_condition      as above, and it must also say whether the visual carries
                          a form a description could convey, or only words

A claim that the lecturer already said something must be evidenced by a quotation
from the window. The check is not part of the prompt: a requirement nothing tests
is decoration, so the quotation is checked against the window after the reply.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from ..core.errors import ContractError
from ..core.registry import Registry
from ..llm.replies import extract_json
from ..prompts import PromptContract, check_inputs, check_output, template
from ..prompts.contracts import REFERENCE, REFERENCE_VISUAL

#: What the reply says about how much of the visual the words already convey.
COVERAGE: tuple[str, ...] = ("FULLY_COVERED", "PARTLY_COVERED", "UNCOVERED")


@dataclass(frozen=True)
class ReferenceDecision:
    """One writer's answer for one moment."""

    coverage: str
    emit: bool
    ad_text: str | None = None
    rung: int | None = None
    rationale: str = ""
    covering_quote: str | None = None
    uncovered_content: str | None = None
    #: Only the way of asking that requires it fills these.
    content: str | None = None
    unspoken_form: str | None = None

    @property
    def describes(self) -> bool:
        return self.emit and bool((self.ad_text or "").strip())


class ReferencePrompt(ABC):
    """One way of asking for a reference description."""

    #: Which template of the prompt catalogue this way of asking uses.
    prompt_name: ClassVar[str]
    #: Which window of speech it reads, by the name of a window strategy.
    window: ClassVar[str]
    contract: ClassVar[PromptContract]

    def render(self, values: Mapping[str, Any]) -> str:
        """The prompt for one moment, with its context bound in."""
        check_inputs(self.contract, values)
        return template(self.prompt_name).bind(values)

    def parse(self, reply: str | None) -> ReferenceDecision | None:
        """The reply as a decision, or None when it carries no object."""
        parsed = extract_json(reply)
        if not isinstance(parsed, dict):
            return None
        coverage = str(parsed.get("coverage") or "")
        if coverage not in COVERAGE:
            return None
        if "emit" not in parsed:
            return None
        return self.decision(parsed)

    def decision(self, parsed: Mapping[str, Any]) -> ReferenceDecision:
        rung = parsed.get("rung")
        return ReferenceDecision(
            coverage=str(parsed["coverage"]),
            emit=bool(parsed.get("emit")),
            ad_text=parsed.get("ad_text"),
            rung=int(rung) if isinstance(rung, int) else None,
            rationale=str(parsed.get("rationale") or ""),
            covering_quote=parsed.get("covering_quote"),
            uncovered_content=parsed.get("uncovered_content"),
        )

    def violations(
        self, decision: ReferenceDecision, values: Mapping[str, Any]
    ) -> list[str]:
        """The ways the answer breaks its contract. Empty when it keeps it."""
        return check_output(self.contract, self.as_reply(decision), values)

    @staticmethod
    def as_reply(decision: ReferenceDecision) -> dict[str, Any]:
        reply: dict[str, Any] = {
            "coverage": decision.coverage,
            "emit": decision.emit,
            "covering_quote": decision.covering_quote,
        }
        if decision.content is not None:
            reply["content"] = decision.content
        if decision.unspoken_form is not None:
            reply["unspoken_form"] = decision.unspoken_form
        return reply


REFERENCE_PROMPTS: Registry[ReferencePrompt] = Registry("reference prompt", ReferencePrompt)


@REFERENCE_PROMPTS.register("fixed_window")
class FixedWindow(ReferencePrompt):
    """Reads a stated stretch of speech before and after the moment."""

    prompt_name = "reference_fixed_window"
    window = "fixed"
    contract = REFERENCE


@REFERENCE_PROMPTS.register("next_event_window")
class NextEventWindow(ReferencePrompt):
    """Reads speech until the next event of the lecture, or a cap."""

    prompt_name = "reference_next_event_window"
    window = "next_event"
    contract = REFERENCE


@REFERENCE_PROMPTS.register("visual_condition")
class VisualCondition(ReferencePrompt):
    """As the window above, and must say whether the visual has a form to convey.

    Both extra answers belong to closed sets, so a reply that claims a visual
    carries form without naming which form, or that invents a kind of content, is a
    violation that can be counted rather than a sentence nobody reads.
    """

    prompt_name = "reference_visual_condition"
    window = "next_event"
    contract = REFERENCE_VISUAL

    def decision(self, parsed: Mapping[str, Any]) -> ReferenceDecision:
        base = super().decision(parsed)
        return ReferenceDecision(
            coverage=base.coverage,
            emit=base.emit,
            ad_text=base.ad_text,
            rung=base.rung,
            rationale=base.rationale,
            covering_quote=base.covering_quote,
            uncovered_content=base.uncovered_content,
            content=parsed.get("content"),
            unspoken_form=parsed.get("unspoken_form"),
        )

    def parse(self, reply: str | None) -> ReferenceDecision | None:
        decision = super().parse(reply)
        if decision is None:
            return None
        if decision.content is None:
            raise ContractError(
                "this way of asking requires the kind of content, and the reply names none"
            )
        return decision
