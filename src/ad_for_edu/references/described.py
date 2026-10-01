"""What has already been described in a lecture.

A rule of the standard forbids describing again what an earlier description of the
same lecture already introduced. Keeping that rule needs memory, and the memory is
per lecture.

This is where the writing stage cannot fan out freely: the prompt for a moment
carries the elements named by the moments before it, so the moments of one lecture
are written in order while the lectures run alongside one another. That is a
property of the rule, not of the code.

What goes into the prompt is names, not sentences. Feeding back whole descriptions
would fill the prompt on a long lecture and invite a model to copy the phrasing,
which produces the near-duplicate candidates a pair set then has to throw away. So
the memory holds an ordered set of element names, and the prompt asks for a
name-only reference or for silence.

An element is only recorded when the slide it was named on actually carried it: a
passing adjective in a description would otherwise become a thing "already
described" and silence a later moment for good.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..core.text import content_terms

#: How many of the most recent elements the prompt carries.
DEFAULT_WINDOW = 40
#: The shortest an element name can be.
MINIMUM_LENGTH = 4

#: Words that name where a thing is or what kind of thing it is, rather than naming
#: the thing. Recording these would silence every later moment about a figure or a
#: table anywhere on any slide.
STRUCTURAL = frozenset(
    {
        "figure", "figures", "slide", "slides", "table", "tables", "chart", "charts",
        "diagram", "image", "picture", "graph", "text", "title", "page", "section",
        "part", "chapter", "left", "right", "top", "bottom", "centre", "center",
        "middle", "above", "below", "next", "example", "overview", "summary",
        "introduction", "conclusion", "outline", "lecture",
    }
)

NOTHING_YET = "(nothing yet — this is the first described moment of the lecture)"


@dataclass
class DescribedSoFar:
    """The elements described in one lecture, in the order they were introduced."""

    lecture: str
    window: int = DEFAULT_WINDOW
    _order: list[str] = field(default_factory=list)
    _seen: set[str] = field(default_factory=set)

    def add(self, description: str | None, slide: str = "") -> list[str]:
        """Record the elements a description named. Returns those that are new.

        `slide` is what was on the slide, which restricts what can be recorded to
        what was there to be described.
        """
        if not description or not description.strip():
            return []
        on_slide = (
            {term.lower() for term in content_terms(slide, cap=80)} if slide else None
        )
        added = []
        for term in content_terms(description, cap=40):
            key = term.lower().strip()
            if len(key) < MINIMUM_LENGTH or key in STRUCTURAL or key in self._seen:
                continue
            if on_slide is not None and key not in on_slide:
                continue
            self._seen.add(key)
            self._order.append(term)
            added.append(term)
        return added

    def prompt_value(self) -> str:
        """The string the prompt carries.

        A stated sentence rather than an empty value, so that a model is never handed
        a bare label and left to guess whether the field failed to fill.
        """
        if not self._order:
            return NOTHING_YET
        return "; ".join(self._order[-self.window :])

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, term: object) -> bool:
        return str(term).lower().strip() in self._seen

    @property
    def elements(self) -> tuple[str, ...]:
        return tuple(self._order)

    def as_dict(self) -> dict[str, Any]:
        return {"lecture": self.lecture, "window": self.window, "elements": list(self._order)}

    @classmethod
    def from_dict(cls, stored: dict[str, Any]) -> DescribedSoFar:
        memory = cls(lecture=stored["lecture"], window=stored.get("window", DEFAULT_WINDOW))
        for term in stored.get("elements", []):
            memory._order.append(term)
            memory._seen.add(term.lower().strip())
        return memory


def in_order(
    moments: Sequence[Any],
    write: Callable[[Any, str], dict[str, Any]],
    *,
    lecture: str,
    slide_of: Callable[[Any], str] | None = None,
    window: int = DEFAULT_WINDOW,
) -> tuple[list[dict[str, Any]], DescribedSoFar]:
    """Write the moments of one lecture in order, carrying the memory through.

    `moments` must already be in the order they occur. The memory means nothing
    otherwise, and sorting by identifier rather than by time is an easy mistake:
    identifiers are padded to a fixed width, so they sort by time only by luck.

    Only a moment that was described adds to the memory. A moment passed over in
    silence described nothing.
    """
    memory = DescribedSoFar(lecture=lecture, window=window)
    written = []
    for moment in moments:
        result = write(moment, memory.prompt_value())
        described = result.get("ad_text") if result.get("emit") else None
        added = memory.add(described, slide_of(moment) if slide_of else "")
        written.append({**result, "new_elements": added, "elements_so_far": len(memory)})
    return written, memory


def by_lecture(moments: Iterable[Any], lecture_of: Callable[[Any], str]) -> dict[str, list[Any]]:
    """The moments grouped by lecture, each group keeping the order it came in."""
    grouped: dict[str, list[Any]] = {}
    for moment in moments:
        grouped.setdefault(lecture_of(moment), []).append(moment)
    return grouped
