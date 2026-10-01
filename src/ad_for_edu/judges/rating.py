"""Judges that grade one description rather than choose between two.

    rubric                  grades the six rule categories and gives one grade overall
    reference_rating        rates a description against one reference
    reference_rating_pooled rates it against every reference of its moment
    rule_compliance         says whether a description keeps one named rule

The rubric's grades are read from fixed lines, one per category. A reply that
misses a category is not read as a lower grade: it is not read at all, and the
failure is recorded.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..data.schema import CATEGORIES
from ..llm.base import LLMRequest
from ..llm.replies import parse_scale, parse_verdict
from ..prompts import template
from ..standard.templates import bind_moment
from .base import JUDGES, Judge

#: The scale each category is graded on.
CATEGORY_SCALE: tuple[int, int] = (1, 5)
#: The scale the grade overall is given on.
OVERALL_SCALE: tuple[int, int] = (1, 10)

_LINE = re.compile(r"^\s*([a-z_]+)\s*:\s*(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class DescriptionToGrade:
    """One description, with what the judge needs to grade it."""

    moment_id: str
    text: str
    references: tuple[str, ...] = ()
    standard: str = ""
    context: Mapping[str, Any] = None  # type: ignore[assignment]
    system: str = ""

    @property
    def item_id(self) -> str:
        return f"{self.moment_id}::{self.system}" if self.system else self.moment_id


@dataclass(frozen=True)
class RubricGrades:
    """What the rubric judge answered."""

    overall: float
    categories: Mapping[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {"overall": self.overall, **dict(self.categories)}


@JUDGES.register("rubric")
class Rubric(Judge[DescriptionToGrade, RubricGrades]):
    """Grades the six categories and gives one grade overall."""

    stage = "score_rubric"

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 900,
        reasoning_effort: str | None = "low",
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

    def request(self, item: DescriptionToGrade) -> LLMRequest:
        context = dict(item.context or {})
        return LLMRequest(
            model=self.model,
            prompt=template("judge_rubric").bind(
                {
                    "standard": item.standard,
                    "on_screen": context.get("what_on_screen", ""),
                    "ocr": context.get("slide_ocr", ""),
                    "transcript": context.get("transcript_window", ""),
                    "refs": "\n".join(f"- {reference}" for reference in item.references),
                    "cand": item.text,
                }
            ),
            temperature=self.temperature,
            # The limit covers the reasoning and the answer together, so a judge that
            # thinks at length has its lines cut off unless the limit is generous.
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            tag=item.item_id,
        )

    def parse(self, reply: str | None) -> RubricGrades | None:
        """The grades, or None when any of the seven is missing or out of range."""
        if not reply:
            return None
        lines = {name: value for name, value in _LINE.findall(reply)}
        grades: dict[str, float] = {}
        for category in CATEGORIES:
            value = parse_scale(lines.get(category), *CATEGORY_SCALE)
            if value is None:
                return None
            grades[category] = float(value)
        overall = parse_scale(lines.get("overall"), *OVERALL_SCALE)
        if overall is None:
            return None
        return RubricGrades(float(overall), grades)

    def identify(self, item: DescriptionToGrade) -> str:
        return item.item_id


class ReferenceRating(Judge[DescriptionToGrade, int]):
    """Rates a description against the reference descriptions of its moment."""

    stage = "rate_against_references"
    scale: tuple[int, int] = CATEGORY_SCALE

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 16) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def parse(self, reply: str | None) -> int | None:
        return parse_scale(reply, *self.scale)

    def identify(self, item: DescriptionToGrade) -> str:
        return item.item_id


@JUDGES.register("reference_rating")
class AgainstOneReference(ReferenceRating):
    """Rates the description against one reference."""

    def request(self, item: DescriptionToGrade) -> LLMRequest:
        reference = item.references[0] if item.references else ""
        return LLMRequest(
            model=self.model,
            prompt=template("judge_reference_rating").bind(ref=reference, cand=item.text),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tag=item.item_id,
        )


@JUDGES.register("reference_rating_pooled")
class AgainstEveryReference(ReferenceRating):
    """Rates the description against every reference of its moment.

    A moment has several references written independently, and a description that
    resembles one of them is not worse for resembling the others less.
    """

    def request(self, item: DescriptionToGrade) -> LLMRequest:
        return LLMRequest(
            model=self.model,
            prompt=template("judge_reference_rating_pooled").bind(
                refs="\n".join(f"- {reference}" for reference in item.references),
                cand=item.text,
            ),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tag=item.item_id,
        )


@dataclass(frozen=True)
class RuleToCheck:
    """One description, one rule, and the moment they belong to."""

    moment_id: str
    rule_id: str
    prompt: str
    values: Mapping[str, Any]

    @property
    def item_id(self) -> str:
        return f"{self.moment_id}::{self.rule_id}"


@JUDGES.register("rule_compliance")
class RuleCompliance(Judge[RuleToCheck, str]):
    """Says whether a description keeps one named rule of the standard."""

    stage = "judge_rules"

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 200) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def request(self, item: RuleToCheck) -> LLMRequest:
        return LLMRequest(
            model=self.model,
            prompt=bind_moment(item.prompt, item.values),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tag=item.item_id,
        )

    def parse(self, reply: str | None) -> str | None:
        verdict, _reasoning = parse_verdict(reply)
        return verdict

    def identify(self, item: RuleToCheck) -> str:
        return item.item_id


def compliance_per_rule(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """The share of descriptions that keep each rule, by rule.

    A rule nothing was judged against is left out rather than counted as kept.
    """
    counts: dict[str, list[int]] = {}
    for row in rows:
        verdict = row.get("answer")
        if verdict not in ("PASS", "FAIL"):
            continue
        rule = str(row.get("rule_id") or "")
        kept, total = counts.setdefault(rule, [0, 0])
        counts[rule] = [kept + (verdict == "PASS"), total + 1]
    return {
        rule: kept / total for rule, (kept, total) in sorted(counts.items()) if total
    }
