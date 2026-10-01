"""The judge family: build a request, read a reply.

A judge never calls anything. It turns one item into a request and one reply into
an answer, and the stage that runs it owns the client, the spend gate, the resuming
and the writing. That separation is what lets a judge be tested without a service
and lets one runner drive every judge.

A reply a judge cannot read yields None, never a value. Every judge here answers
on a scale or from a small set, so a failure that became a plausible answer would
be indistinguishable from a real one: a tie, the top of a scale, a zero.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, TypeVar

from ..core.registry import Registry
from ..llm.base import LLMRequest

Item = TypeVar("Item")
Answer = TypeVar("Answer")


@dataclass(frozen=True)
class JudgeRow:
    """One judged item, as it is written out.

    A row carries what was asked as well as what came back, so a stored answer can
    be read without the code that produced it. A row that failed carries the reply
    that could not be read: a failure with no reply can be counted and never
    diagnosed.
    """

    item_id: str
    judge: str
    model: str
    answer: Any = None
    error: str | None = None
    raw: str | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)

    @property
    def answered(self) -> bool:
        return self.error is None and self.answer is not None

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "item_id": self.item_id,
            "judge": self.judge,
            "model": self.model,
            **dict(self.fields),
        }
        if self.error is not None:
            row["error"] = self.error
        if self.raw is not None:
            row["raw"] = self.raw
        if isinstance(self.answer, Mapping):
            row.update(self.answer)
        elif self.answer is not None:
            row["answer"] = self.answer
        return row


class Judge(ABC, Generic[Item, Answer]):
    """Asks one question about one item."""

    #: The stage this judge belongs to, which is how its cost is priced.
    stage: ClassVar[str]
    #: Whether it needs to watch the moment rather than look at a still of it.
    watches: ClassVar[bool] = False

    @abstractmethod
    def request(self, item: Item) -> LLMRequest: ...

    @abstractmethod
    def parse(self, reply: str | None) -> Answer | None: ...

    def identify(self, item: Item) -> str:
        """What to record this item as. By default its own identifier."""
        for name in ("pair_id", "item_id", "moment_id", "id"):
            found = getattr(item, name, None) or (
                item.get(name) if isinstance(item, Mapping) else None
            )
            if found:
                return str(found)
        raise KeyError("the item carries no identifier to record it by")


JUDGES: Registry[Judge] = Registry("judge", Judge)
