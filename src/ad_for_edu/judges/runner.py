"""Running a judge over many items.

The runner owns everything the judge does not: the client, the estimate and its
approval, resuming, and what a failure becomes. It works for every judge, which is
why a judge builds a request instead of making a call.

Resuming reads the rows already written and asks again only for what has no usable
answer. A row recording a failure is not an answer, so it is asked again; a row
recording a reply that could not be read is asked again too.

A failure that no retry will clear stops the run. Retrying such a failure spends
the attempts of every remaining item and ends a run that did nothing, looking like
a run that merely met a lot of trouble.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.io import append_jsonl
from ..core.outputs import completed_keys
from ..llm.base import LLMReply
from ..llm.client import LLMClient
from ..llm.replies import keep_raw
from .base import Judge, JudgeRow


@dataclass
class RunSummary:
    """What a judging run did."""

    asked: int = 0
    answered: int = 0
    unread: int = 0
    failed: int = 0
    skipped: int = 0

    @property
    def share_answered(self) -> float:
        return self.answered / self.asked if self.asked else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "asked": self.asked,
            "answered": self.answered,
            "unread": self.unread,
            "failed": self.failed,
            "already_done": self.skipped,
            "share_answered": round(self.share_answered, 4),
        }


class JudgeRunner:
    """Asks a judge about each item, writing every answer as it arrives."""

    def __init__(
        self,
        judge: Judge,
        client: LLMClient,
        *,
        role: str,
        fields_of: Callable[[Any], Mapping[str, Any]] = lambda item: {},
    ) -> None:
        self.judge = judge
        self.client = client
        self.role = role
        self.fields_of = fields_of

    def already_done(self, output: Path) -> set[Hashable]:
        """The items of `output` that have a usable answer."""
        return completed_keys(
            output,
            key=lambda row: row.get("item_id"),
            is_complete=lambda row: row.get("error") is None,
        )

    def run(
        self,
        items: Sequence[Any],
        output: Path,
        *,
        resume: bool = True,
        batch_size: int = 1,
    ) -> RunSummary:
        """Ask about every item and append each answer to `output`."""
        summary = RunSummary()
        done = self.already_done(output) if resume else set()
        remaining = [item for item in items if self.judge.identify(item) not in done]
        summary.skipped = len(items) - len(remaining)
        for start in range(0, len(remaining), max(1, batch_size)):
            batch = remaining[start : start + max(1, batch_size)]
            requests = [self.judge.request(item) for item in batch]
            replies = self.client.generate_many(requests, role=self.role)
            for item, reply in zip(batch, replies, strict=True):
                append_jsonl(output, self._row(item, reply).as_dict())
                summary.asked += 1
                if isinstance(reply, Exception):
                    summary.failed += 1
                elif self.judge.parse(reply.text) is None:
                    summary.unread += 1
                else:
                    summary.answered += 1
        return summary

    def _row(self, item: Any, reply: LLMReply | Exception) -> JudgeRow:
        name = type(self.judge).strategy_name  # type: ignore[attr-defined]
        identifier = self.judge.identify(item)
        fields = dict(self.fields_of(item))
        if isinstance(reply, Exception):
            return JudgeRow(
                identifier,
                name,
                getattr(self.judge, "model", ""),
                error=f"{type(reply).__name__}: {reply}",
                fields=fields,
            )
        answer = self.judge.parse(reply.text)
        if answer is None:
            # The reply is kept: a failure with no reply can be counted and never
            # diagnosed, and a better reader cannot be written without one.
            return JudgeRow(
                identifier,
                name,
                getattr(self.judge, "model", ""),
                error="the reply could not be read",
                raw=keep_raw({}, reply.text)["raw"],
                fields=fields,
            )
        recorded = answer.as_dict() if hasattr(answer, "as_dict") else answer
        return JudgeRow(identifier, name, getattr(self.judge, "model", ""), recorded, fields=fields)


def estimate_for(
    judge: Judge,
    items: Iterable[Any],
    prices: Any,
    *,
    row_of: Callable[[Any], str] = lambda item: "all",
    model: str | None = None,
) -> Any:
    """What asking this judge about these items would cost."""
    calls: dict[str, dict[str, int]] = {}
    named = model or getattr(judge, "model", "")
    for item in items:
        row = calls.setdefault(row_of(item), {})
        row[named] = row.get(named, 0) + 1
    return prices.estimate(judge.stage, calls)
