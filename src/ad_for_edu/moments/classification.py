"""Deciding what each candidate event is, or rejecting it.

A batch of candidates goes out in one call and comes back as one reply object per
candidate. Four guards stand between that reply and the classified moments, and
each exists because the thing it checks can fail without looking like a failure.

**The reply is read line by line, one object per line.** A reader that takes the
first structure of a whole reply would collapse a batch of ten to one, and the
count guard below depends on the line-to-object correspondence.

**A batch is complete and correctly identified, or it is rejected.** A reply that
stops early cannot be told from a correctly short one by its content. Counting
alone is not enough either: a reply that repeats one identifier and drops another
has the right count.

**Every label belongs to the set the prompt offered, and that set is the intended
one.** Both halves are needed. A guard that checks only the labels passes when the
prompt failed to compose, because the model then answers in a vocabulary of its
own and the label the guard looks for is absent from both sides. So the offered
set is read back out of the prompt that is about to be sent.

**A batch covers one lecture.** The label set is bound per batch and depends on
the lecture, so a batch spanning two would offer one lecture's labels to another.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..core.errors import ContractError
from ..core.timecodes import minutes_to_seconds
from ..data.schema import MOMENT_TYPES, REJECT, CandidateEvent, Moment
from ..llm.replies import extract_json
from ..preprocessing.cursor import allowed_types
from ..prompts import read_data, template

#: How the prompt lays out the labels it offers: two spaces, the label, then two more.
_OFFERED = re.compile(r"^\s{2}([a-z_]+)\s{2,}")
_PLACEHOLDER = re.compile(r"\{[a-z_]+\}")
#: Fewer offered labels than this means the prompt did not compose.
MINIMUM_OFFERED = 5


def parse_objects(reply: str | None) -> list[dict[str, Any]]:
    """One object per line of the reply.

    A line that is not an object is dropped rather than guessed at; a dropped line
    then shows up as a short batch, which the next guard catches. A line carrying
    trailing commentary still parses, since the object in it is read out.
    """
    found = []
    for line in (reply or "").splitlines():
        line = line.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        if not line.startswith("{"):
            continue
        parsed = extract_json(line)
        if isinstance(parsed, dict):
            found.append(parsed)
    return found


def batch_fault(wanted: Sequence[str], got: Sequence[Mapping[str, Any]]) -> str | None:
    """Why a batch must be rejected, or None when it is complete and correct."""
    returned = [answer.get("id") for answer in got]
    if len(got) != len(wanted):
        return f"{len(wanted)} answers were asked for and {len(got)} came back"
    repeated = [name for name, count in Counter(returned).items() if count > 1]
    if repeated:
        return f"identifiers answered more than once: {repeated[:3]}"
    missing = sorted(set(wanted) - set(returned))
    if missing:
        return f"identifiers not answered: {missing[:3]}"
    return None


def offered_types(prompt: str) -> set[str]:
    """The labels the prompt offers, read back out of the prompt itself.

    Read back rather than passed in: the point is to compare what the model was
    offered with what it returned, and a set handed to both sides would agree with
    itself even where the prompt said something else.
    """
    found: set[str] = set()
    inside = False
    for line in prompt.splitlines():
        if line.startswith("TYPES"):
            inside = True
            continue
        if inside:
            if not line.strip():
                break
            match = _OFFERED.match(line)
            if match:
                found.add(match.group(1))
    return found


def vocabulary_fault(prompt: str, got: Sequence[Mapping[str, Any]]) -> str | None:
    """Why the labels of a reply cannot be trusted, or None."""
    offered = offered_types(prompt)
    if len(offered) < MINIMUM_OFFERED:
        return (
            f"the prompt offers only {sorted(offered)}, so its label list did not compose "
            "and any agreement with the reply means nothing"
        )
    outside = sorted({str(answer.get("type")) for answer in got} - offered)
    if outside:
        return (
            f"labels outside the {len(offered)} offered: {outside[:4]}; the model is not using "
            "the set it was given"
        )
    return None


def assert_ready_to_send(prompt: str) -> None:
    """Raise unless the prompt carries a label list and no unfilled placeholder."""
    unfilled = _PLACEHOLDER.findall(prompt)
    if unfilled:
        raise ContractError(f"the prompt still carries {unfilled}; it will not be sent")
    if "TYPES" not in prompt or f"{REJECT} " not in prompt:
        raise ContractError("the prompt carries no label list; it will not be sent")


def types_block(labels: Sequence[str]) -> str:
    """The label section of the prompt, with the withheld labels absent."""
    data = read_data("classify_types")
    lines = [data["types"][name] for name in data["order"] if name in set(labels)]
    note = (data["pointing_note"] if "pointing" in set(labels) else "") + data["reject_note"]
    return "TYPES - use exactly one:\n  " + "\n  ".join(lines) + "\n\n" + note


def prompt_for(*, draws_pointer: bool) -> str:
    """The classifier prompt, with the label set this recording can support bound in.

    A recording that draws no pointer is not offered the pointing label. Asking a
    model not to invent a pointer is asking; not offering the label removes the
    opportunity.
    """
    data = read_data("classify_types")
    labels = allowed_types(tuple(data["order"]), draws_pointer)
    bound = template("classify_moments").bind(types=types_block(labels))
    assert_ready_to_send(bound)
    return bound


def batch_by_lecture(
    candidates: Sequence[CandidateEvent], size: int
) -> list[list[CandidateEvent]]:
    """Batches of at most `size`, none spanning two lectures."""
    if size < 1:
        raise ValueError(f"a batch holds at least one candidate, got {size}")
    by_lecture: dict[str, list[CandidateEvent]] = {}
    for candidate in candidates:
        by_lecture.setdefault(candidate.lecture, []).append(candidate)
    batches = []
    for group in by_lecture.values():
        batches += [group[start : start + size] for start in range(0, len(group), size)]
    return batches


def batch_request_text(
    candidates: Sequence[CandidateEvent], transcripts: Mapping[str, str]
) -> str:
    """The part of the call that names this batch's candidates."""
    lines = ["CANDIDATES IN THIS BATCH:"]
    for candidate in candidates:
        lines.append(
            f"- id: {candidate.id} | detector channel: {candidate.channel} | "
            f"transcript: {transcripts[candidate.id]}"
        )
    lines.append("")
    lines.append(
        f"Return exactly {len(candidates)} JSON objects, one per candidate, in this order."
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class Classification:
    """One answer, read and checked."""

    candidate_id: str
    type: str
    time: float
    what_on_screen: str
    revisit: bool = False

    @property
    def rejected(self) -> bool:
        return self.type == REJECT


def read_answer(answer: Mapping[str, Any], candidate: CandidateEvent) -> Classification:
    """One answer as a classification. The time is read off the answer, not estimated.

    The time is read in the form the stills carry it and the prompt asks for it,
    minutes and seconds with no hour field. Where the answer carries no readable
    time the time of the candidate stands: the detector knows when the event was,
    and a time invented from nothing would move the moment.
    """
    label = str(answer.get("type") or "")
    if label not in (*MOMENT_TYPES, REJECT):
        raise ContractError(f"{candidate.id}: label {label!r} is outside the closed set")
    stated = minutes_to_seconds(answer.get("t"))
    return Classification(
        candidate_id=candidate.id,
        type=label,
        time=stated if stated >= 0 else float(candidate.t),
        what_on_screen=str(answer.get("what_on_screen") or "").strip(),
        revisit=bool(answer.get("revisit")),
    )


def to_moment(classification: Classification, candidate: CandidateEvent, model: str) -> Moment:
    """A classification that is not a rejection, as a moment."""
    if classification.rejected:
        raise ContractError(f"{candidate.id} was rejected and is not a moment")
    return Moment(
        id=candidate.id,
        lecture=candidate.lecture,
        t=classification.time,
        type=classification.type,
        channel=candidate.channel,
        revisit=classification.revisit,
        what_on_screen=classification.what_on_screen,
        model=model,
    )
