"""Asking several writers for a reference description of every moment.

Each writer answers every moment independently, so a moment carries one row per
writer: several competent describers of one moment disagree, and a metric that
scores a candidate against the best of them is measuring whether it describes the
moment, not whether it wrote one particular sentence.

Two rules govern the work.

**A moment of the test side carries every writer, or it carries none.** The best
value over four references and the best over three are not the same quantity, and
they differ by a different amount than the spread between the writers does. A
moment with a writer missing is therefore left out of the scored set and recorded
as left out, rather than scored against whichever writers answered.

**The moments of one lecture are written in order.** The prompt for a moment names
what earlier moments described, so the order is part of the input.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import ContractError
from ..core.ids import description_id
from ..data.schema import ReferenceRow
from .described import DescribedSoFar
from .prompts import ReferenceDecision, ReferencePrompt


@dataclass
class WriterAnswer:
    """What one writer answered for one moment, before it becomes a row."""

    moment_id: str
    writer: str
    decision: ReferenceDecision | None
    violations: tuple[str, ...] = ()
    error: str | None = None
    raw: str | None = None

    @property
    def usable(self) -> bool:
        return self.error is None and self.decision is not None


def to_row(
    answer: WriterAnswer,
    context: Mapping[str, Any],
    *,
    window: str,
    time: Any = None,
) -> dict[str, Any]:
    """One writer's answer as a row of the reference set.

    The row carries the context the answer was given, so a later stage reads the
    same slide text and the same words the writer read. `window` names which stretch
    of speech that was: a row whose window is not stated cannot be compared with one
    written under another.
    """
    if not answer.usable:
        return {
            "output_id": description_id(answer.moment_id, answer.writer),
            "family": answer.writer,
            "error": answer.error or "unparsed",
            "raw": answer.raw,
        }
    decision = answer.decision
    row = ReferenceRow(
        output_id=description_id(answer.moment_id, answer.writer),
        time=time,
        type=str(context.get("type") or ""),
        emit=decision.emit,
        ad_text=decision.ad_text,
        rung=decision.rung,
        rationale=decision.rationale,
        family=answer.writer,
        transcript_window=str(context.get("transcript_window") or ""),
        slide_ocr=str(context.get("slide_ocr") or ""),
        what_on_screen=str(context.get("what_on_screen") or ""),
        pause_after=float(context.get("pause_after") or 0.0),
        emit_window=window,
    ).to_row()
    row["coverage"] = decision.coverage
    row["covering_quote"] = decision.covering_quote
    row["uncovered_content"] = decision.uncovered_content
    if decision.content is not None:
        row["visual_content"] = decision.content
        row["unspoken_form"] = decision.unspoken_form
    if answer.violations:
        row["violations"] = list(answer.violations)
    return row


@dataclass
class ReferenceSet:
    """The rows of the reference set, and what was left out of the scored part."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    left_out: dict[str, list[str]] = field(default_factory=dict)

    def moments(self) -> set[str]:
        from ..core.ids import moment_of

        return {moment_of(row) for row in self.rows}


def writers_per_moment(rows: Iterable[Mapping[str, Any]]) -> dict[str, set[str]]:
    """Which writers answered each moment."""
    from ..core.ids import moment_of

    answered: dict[str, set[str]] = {}
    for row in rows:
        if row.get("error"):
            continue
        answered.setdefault(moment_of(row), set()).add(str(row.get("family") or ""))
    return answered


def incomplete_moments(
    rows: Iterable[Mapping[str, Any]], writers: Sequence[str]
) -> dict[str, list[str]]:
    """The moments a writer is missing from, and which writers those are."""
    wanted = set(writers)
    answered = writers_per_moment(rows)
    return {
        moment: sorted(wanted - found)
        for moment, found in sorted(answered.items())
        if not wanted <= found
    }


def complete_only(
    rows: Sequence[Mapping[str, Any]], writers: Sequence[str]
) -> tuple[list[Mapping[str, Any]], dict[str, list[str]]]:
    """The rows of the moments every writer answered, and what was left out.

    Scoring a moment against a partial set of writers moves the score of a system
    and the spread between the writers by different amounts, so the comparison a
    complete set licenses no longer holds for it.
    """
    from ..core.ids import moment_of

    missing = incomplete_moments(rows, writers)
    kept = [row for row in rows if moment_of(row) not in missing]
    return kept, missing


class ReferenceGenerator:
    """Asks the writers for a description of every moment of a lecture, in order."""

    def __init__(
        self,
        prompt: ReferencePrompt,
        writers: Sequence[str],
        ask: Callable[[str, str, Sequence[Any]], WriterAnswer],
    ) -> None:
        """`ask(writer, prompt_text, stills)` makes one call and reads the reply."""
        if not writers:
            raise ContractError("a reference set needs at least one writer")
        self.prompt = prompt
        self.writers = tuple(writers)
        self.ask = ask

    def write_lecture(
        self,
        lecture: str,
        moments: Sequence[Mapping[str, Any]],
        *,
        context_of: Callable[[Mapping[str, Any], str], Mapping[str, Any]],
        stills_of: Callable[[Mapping[str, Any]], Sequence[Any]] = lambda moment: (),
        window: int | None = None,
    ) -> list[dict[str, Any]]:
        """Every writer's answer for every moment of one lecture, in order.

        Each writer carries its own memory of what it has described, because what one
        writer already introduced says nothing about what another has.
        """
        memories = {
            writer: DescribedSoFar(lecture=lecture, **({"window": window} if window else {}))
            for writer in self.writers
        }
        rows: list[dict[str, Any]] = []
        for moment in moments:
            for writer in self.writers:
                memory = memories[writer]
                context = dict(context_of(moment, memory.prompt_value()))
                text = self.prompt.render(context)
                answer = self.ask(writer, text, stills_of(moment))
                if answer.usable:
                    answer.violations = tuple(self.prompt.violations(answer.decision, context))
                    if answer.decision.describes:
                        memory.add(answer.decision.ad_text, str(context.get("slide_ocr") or ""))
                rows.append(
                    to_row(
                        answer,
                        context,
                        window=self.prompt.window,
                        time=moment.get("t"),
                    )
                )
        return rows
