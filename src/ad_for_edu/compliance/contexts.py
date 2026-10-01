"""Where the context of a moment comes from when a description is scored.

The scorer needs five fields per moment and, optionally, the fact whether the
lecture shows a pointer. A context source turns files into those contexts. It
scores nothing and does not know which mode will read what it returns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..core.errors import ContractError
from ..core.ids import lecture_of, moment_of
from ..core.io import iter_jsonl, read_json
from ..core.registry import Registry
from ..core.timecodes import as_seconds
from ..data.schema import MomentContext

#: Pointer facts per lecture: `{lecture: True | False}`.
CursorFacts = Mapping[str, bool]


class ContextSource(ABC):
    """Supplies the context of each moment."""

    @abstractmethod
    def contexts(self) -> dict[str, MomentContext]:
        """Every context this source holds, by moment id."""

    def context(self, moment_id: str) -> MomentContext:
        found = self.contexts().get(moment_id)
        if found is None:
            raise ContractError(
                f"moment {moment_id!r} has no context. Check that the descriptions and the "
                "context rows come from the same set of moments."
            )
        return found


CONTEXT_SOURCES: Registry[ContextSource] = Registry("context source", ContextSource)


def load_cursor_facts(path: str | Path) -> dict[str, bool]:
    """Pointer facts from a JSON file `{lecture: {"pointing_allowed": bool}}` or `{lecture: bool}`.

    Every lecture must state the fact. A missing fact would mean "score deixis",
    so a malformed file would switch the abstention off without notice.
    """
    raw = read_json(path)
    if not isinstance(raw, dict) or not raw:
        raise ContractError(f"{path}: expected a non-empty mapping of lectures")
    facts: dict[str, bool] = {}
    unreadable: list[str] = []
    for lecture, entry in raw.items():
        if isinstance(entry, bool):
            facts[lecture] = entry
        elif isinstance(entry, dict) and isinstance(entry.get("pointing_allowed"), bool):
            facts[lecture] = entry["pointing_allowed"]
        else:
            unreadable.append(lecture)
    if unreadable:
        raise ContractError(
            f"{path}: {len(unreadable)} lectures do not state the pointer fact, "
            f"for example {unreadable[:3]}"
        )
    return facts


def contexts_from_rows(
    rows: Iterable[Mapping[str, Any]],
    cursor_facts: CursorFacts | None = None,
) -> dict[str, MomentContext]:
    """One context per moment, taken from the first row of that moment.

    All rows are read, not only the ones that carry a description: a moment on
    which every writer stayed silent can still be the moment of a pair.
    """
    contexts: dict[str, MomentContext] = {}
    for row in rows:
        moment = moment_of(row)
        if not moment or moment in contexts:
            continue
        fact = None if cursor_facts is None else cursor_facts.get(lecture_of(moment))
        contexts[moment] = MomentContext.from_row(row, moment_id=moment, renders_cursor=fact)
    return contexts


def times_from_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """The time of each moment, from the first row of that moment; 0 when unreadable."""
    times: dict[str, float] = {}
    for row in rows:
        moment = moment_of(row)
        if not moment or moment in times:
            continue
        seconds = as_seconds(row.get("time") if row.get("time") is not None else row.get("t"))
        times[moment] = seconds if seconds >= 0 else 0.0
    return times


@CONTEXT_SOURCES.register("reference_rows")
class ReferenceRowContexts(ContextSource):
    """Contexts read from a file of reference rows."""

    def __init__(self, path: str | Path, cursor_facts: str | Path | None = None) -> None:
        self.path = Path(path)
        self.cursor_facts_path = Path(cursor_facts) if cursor_facts else None
        self._contexts: dict[str, MomentContext] | None = None
        self._times: dict[str, float] | None = None

    def contexts(self) -> dict[str, MomentContext]:
        if self._contexts is None:
            facts = load_cursor_facts(self.cursor_facts_path) if self.cursor_facts_path else None
            self._contexts = contexts_from_rows(iter_jsonl(self.path), facts)
            if not self._contexts:
                raise ContractError(f"{self.path}: no rows")
        return self._contexts

    def times(self) -> dict[str, float]:
        if self._times is None:
            self._times = times_from_rows(iter_jsonl(self.path))
        return self._times


@CONTEXT_SOURCES.register("in_memory")
class InMemoryContexts(ContextSource):
    """Contexts handed over directly, for callers that already hold the rows."""

    def __init__(
        self,
        rows: Iterable[Mapping[str, Any]],
        cursor_facts: CursorFacts | None = None,
    ) -> None:
        materialised = list(rows)
        self._contexts = contexts_from_rows(materialised, cursor_facts)
        self._times = times_from_rows(materialised)

    def contexts(self) -> dict[str, MomentContext]:
        return self._contexts

    def times(self) -> dict[str, float]:
        return self._times
