"""Asking for several descriptions of one moment, to pair against each other.

Two ways of asking. One asks for several descriptions in a single call, which is
the way of record; the other asks once per description. Both are given a focus
directive, which names what the description should attend to, so that two
descriptions of one moment differ in what they say rather than only in wording.

A description is usable when it falls inside the word band the call asked for and
names a rung it could be delivered at. The bands are told to the model, not only
used to check its answer: a model graded against a target it was never given
misses it, and the failures are then all in one direction.

The band a moment is given follows how much of the visual the words leave
unsaid and which rung the gap allows. A moment whose visual is wholly unsaid and
whose gap allows a full delivery may take the widest band; its ceiling is
computed for that moment, because a fixed ceiling above what the moment can
deliver is a target nothing can reach.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from ..core.registry import Registry
from ..core.text import word_count
from ..llm.replies import extract_json
from ..moments.rungs import REACHABLE, deliverable_words
from ..prompts import check_inputs, template
from ..prompts.contracts import CANDIDATES, FOCUS_DIRECTIVES

#: The word bands, by the name a call refers to them by.
BANDS: Mapping[str, tuple[int, int]] = {
    "S": (12, 15),
    "M": (16, 25),
    "L": (23, 25),
    "XL": (40, 60),
}
#: Which band a rung is given, when the widest is not available.
BAND_FOR_RUNG: Mapping[int, str] = {1: "L", 2: "S", 3: "M", 4: "S"}
#: The band for a moment the words leave wholly unsaid, where the rung allows it.
WIDEST = "XL"
#: The rungs the widest band is offered at: those that can deliver a long description.
WIDEST_RUNGS: tuple[int, ...] = (1, 3)
UNCOVERED = "UNCOVERED"


def band_bounds(
    band: str, pause: float, rung: int, bands: Mapping[str, tuple[int, int]] | None = None
) -> tuple[int, int] | None:
    """The word bounds of `band` for this moment, or None when it cannot support it.

    The floor of the widest band is fixed, since it is what distinguishes that band
    from the next. Its ceiling is what the moment can actually deliver: a ceiling
    declared once, above what most moments can deliver, is a target the model is
    graded against and cannot reach.
    """
    table = bands or BANDS
    low, high = table[band]
    if band != WIDEST:
        return (low, high)
    ceiling = int(deliverable_words(rung, pause))
    if ceiling < low:
        return None
    return (low, min(high, ceiling))


def band_for(coverage: str, rung: int, pause: float) -> tuple[str, int, int]:
    """The band a moment is given, and its bounds."""
    if (coverage or "").upper() == UNCOVERED and rung in WIDEST_RUNGS:
        widest = band_bounds(WIDEST, pause, rung)
        if widest is not None:
            return (WIDEST, widest[0], widest[1])
    band = BAND_FOR_RUNG.get(rung, "M")
    low, high = band_bounds(band, pause, rung)
    return (band, low, high)


@dataclass(frozen=True)
class Candidate:
    """One description offered for a moment."""

    ad_text: str
    rung: int | None = None
    focus: str | None = None

    @property
    def words(self) -> int:
        return word_count(self.ad_text)

    def usable(self, low: int, high: int) -> bool:
        """Whether it falls in the band asked for and names a rung it could be delivered at."""
        return bool(self.ad_text.strip()) and low <= self.words <= high and self.rung in REACHABLE


class CandidateMode(ABC):
    """One way of asking for candidate descriptions."""

    prompt_name: ClassVar[str]
    #: How many descriptions one call asks for.
    per_call: ClassVar[int]

    def render(self, values: Mapping[str, Any]) -> str:
        check_inputs(CANDIDATES, values)
        return template(self.prompt_name).bind(values)

    @abstractmethod
    def parse(self, reply: str | None) -> list[Candidate]: ...


CANDIDATE_MODES: Registry[CandidateMode] = Registry("candidate mode", CandidateMode)


def _one(parsed: Mapping[str, Any]) -> Candidate | None:
    text = parsed.get("ad_text")
    if not isinstance(text, str) or not text.strip():
        return None
    rung = parsed.get("rung")
    return Candidate(
        ad_text=text.strip(),
        rung=int(rung) if isinstance(rung, int) else None,
        focus=parsed.get("focus") or parsed.get("directive"),
    )


@CANDIDATE_MODES.register("verbalized_sampling")
class VerbalizedSampling(CandidateMode):
    """Several descriptions in one call, which is cheaper per description.

    The reply is a list. A reply that is one object is one description, not a list of
    one, and is read as such.
    """

    prompt_name = "candidates_verbalized_sampling"
    per_call = 5

    def parse(self, reply: str | None) -> list[Candidate]:
        parsed = extract_json(reply)
        if isinstance(parsed, Mapping):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return []
        found = [_one(entry) for entry in parsed if isinstance(entry, Mapping)]
        return [candidate for candidate in found if candidate is not None]


@CANDIDATE_MODES.register("per_call")
class PerCall(CandidateMode):
    """One description per call."""

    prompt_name = "candidates_per_call"
    per_call = 1

    def parse(self, reply: str | None) -> list[Candidate]:
        parsed = extract_json(reply)
        if not isinstance(parsed, Mapping):
            return []
        candidate = _one(parsed)
        return [candidate] if candidate is not None else []


def directives(names: Sequence[str] | None = None) -> dict[str, str]:
    """The focus directives, by name. Every name must be one of the declared set."""
    from ..prompts import read_data

    declared = read_data("focus_directives")
    if names is None:
        return dict(declared)
    unknown = sorted(set(names) - set(declared))
    if unknown:
        raise ValueError(f"unknown focus directives {unknown}; known: {sorted(declared)}")
    return {name: declared[name] for name in names}


def check_directives(names: Sequence[str]) -> None:
    """Raise unless every name is a declared directive."""
    unknown = sorted(set(names) - set(FOCUS_DIRECTIVES))
    if unknown:
        raise ValueError(f"unknown focus directives {unknown}; known: {sorted(FOCUS_DIRECTIVES)}")
