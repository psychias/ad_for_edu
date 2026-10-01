"""Judges that choose between two descriptions of one moment.

Three of them, differing in what they are shown.

    video    a clip of the moment. What tells two descriptions apart is often
             movement, which a single still cannot carry.
    text     the moment as text: its type, the words spoken, the slide text.
    sheet    what a person rating the pair is shown, so that a model's answers can
             stand beside theirs on the same evidence.

Every one of them is asked about a pair twice, once each way round. Which side an
answer names therefore depends on the order it was asked in, and resolving that is
not the judge's business: it hands back the option it was told, and the ordering
step turns that into a side.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..llm.base import LLMRequest
from ..llm.replies import parse_choice
from ..pairs.ordering import side_shown_first
from ..prompts import template
from .base import JUDGES, Judge


@dataclass(frozen=True)
class PairToJudge:
    """A pair as a judge is asked about it, in one presentation order."""

    pair_id: str
    order: int
    first: str
    second: str
    moment_id: str = ""
    clip: Path | None = None
    still: Path | None = None
    context: Mapping[str, Any] = None  # type: ignore[assignment]

    @classmethod
    def from_pair(
        cls,
        pair: Mapping[str, Any],
        order: int,
        *,
        clip: Path | None = None,
        still: Path | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> PairToJudge:
        """The pair as shown in `order`: in the second order the sides are swapped."""
        shown_first = side_shown_first(order)
        other = "b" if shown_first == "a" else "a"
        return cls(
            pair_id=str(pair["pair_id"]),
            order=order,
            first=str(pair[shown_first]),
            second=str(pair[other]),
            moment_id=str(pair.get("moment_id") or ""),
            clip=clip,
            still=still,
            context=dict(context or {}),
        )

    @property
    def item_id(self) -> str:
        return f"{self.pair_id}#order{self.order}"


class PairwiseJudge(Judge[PairToJudge, str]):
    """Chooses between the two descriptions it is shown."""

    def parse(self, reply: str | None) -> str | None:
        """The option the reply names, or None. Which side that is depends on the order."""
        return parse_choice(reply)

    def identify(self, item: PairToJudge) -> str:
        return item.item_id


@JUDGES.register("video_pairwise")
class VideoPairwise(PairwiseJudge):
    """Watches a clip of the moment and chooses."""

    stage = "judge_pairs"
    watches = True

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1200) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def request(self, item: PairToJudge) -> LLMRequest:
        return LLMRequest(
            model=self.model,
            prompt=template("judge_video_pairwise").bind(d1=item.first, d2=item.second),
            videos=(item.clip,) if item.clip else (),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tag=item.item_id,
        )


@JUDGES.register("text_pairwise")
class TextPairwise(PairwiseJudge):
    """Reads the moment as text and chooses."""

    stage = "judge_head_to_head"

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1200) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def request(self, item: PairToJudge) -> LLMRequest:
        context = dict(item.context or {})
        return LLMRequest(
            model=self.model,
            prompt=template("judge_text_pairwise").bind(
                {
                    "type": context.get("type", ""),
                    "transcript_window": context.get("transcript_window", ""),
                    "slide_ocr": context.get("slide_ocr", ""),
                    "pause_after": context.get("pause_after", 0),
                    "ad1": item.first,
                    "ad2": item.second,
                }
            ),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tag=item.item_id,
        )


@JUDGES.register("sheet_pairwise")
class SheetPairwise(PairwiseJudge):
    """Is shown what a person rating the pair is shown, and chooses."""

    stage = "annotate_rated_pairs"

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 900) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def request(self, item: PairToJudge) -> LLMRequest:
        context = dict(item.context or {})
        return LLMRequest(
            model=self.model,
            prompt=template("judge_sheet_pairwise").bind(
                {
                    "subject": context.get("subject", ""),
                    "moment_type": context.get("type", ""),
                    "on_screen": context.get("what_on_screen", ""),
                    "slide_ocr": context.get("slide_ocr", ""),
                    "transcript_window": context.get("transcript_window", ""),
                    "ad1": item.first,
                    "ad2": item.second,
                }
            ),
            images=(item.still,) if item.still else (),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tag=item.item_id,
        )
