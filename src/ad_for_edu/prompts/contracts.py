"""What must hold for the inputs of a prompt before it is sent, and for its reply.

A helper that builds a value can be correct while the value that reaches the
model is empty. The contract is therefore checked on the values themselves,
immediately before the call. An empty string is what a failed lookup produces,
so an input may be empty only where its contract says so.

Violations found in a reply are returned, not raised: a quotation that is not in
the transcript is a finding to count over a run, not a reason to lose the row.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..core.errors import ContractError
from ..core.text import NO_SPEECH
from ..data.schema import CATEGORIES


@dataclass(frozen=True)
class FieldSpec:
    """One declared input of a prompt."""

    name: str
    non_empty: bool = True
    #: The values the input may take, when the set is closed.
    allowed: tuple[str, ...] = ()
    #: Markers that are legitimate values, such as the marker of a silent window.
    sentinels: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromptContract:
    """The declared inputs and outputs of one prompt."""

    name: str
    inputs: tuple[FieldSpec, ...]
    #: Keys the parsed reply must contain.
    required_out: tuple[str, ...] = ()
    #: `(reply key, input name)`: the reply value must be a verbatim span of the input.
    quotes_from: tuple[tuple[str, str], ...] = ()
    #: `(reply key, allowed values)`: a reply value outside the set is a violation;
    #: a null value is accepted.
    closed_out: tuple[tuple[str, tuple[str, ...]], ...] = ()
    notes: str = ""

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.inputs)


FOCUS_DIRECTIVES: tuple[str, ...] = ("REFERENT", "STRUCTURE", "CHANGE", "DATA", "SPATIAL")
LENGTH_BANDS: tuple[str, ...] = ("S", "M", "L", "XL")
RUNGS: tuple[str, ...] = ("1", "2", "3", "4")
CONTENT_CLASSES: tuple[str, ...] = ("NONE", "TEXT_ONLY", "VISUAL")
FORM_PROPERTIES: tuple[str, ...] = (
    "which_element",
    "shape_trend",
    "spatial_arrangement",
    "relative_magnitude",
    "count",
    "value",
    "colour_code",
)

#: The six fields every writing prompt takes about the moment.
MOMENT_INPUTS: tuple[FieldSpec, ...] = (
    FieldSpec("type"),
    FieldSpec("transcript_window", sentinels=(NO_SPEECH,)),
    FieldSpec("slide_ocr", non_empty=False),
    FieldSpec("pause_after"),
    FieldSpec("reachable_rung", allowed=RUNGS),
    FieldSpec("described_so_far", non_empty=False),
)

REFERENCE = PromptContract(
    name="reference",
    inputs=MOMENT_INPUTS,
    required_out=("emit", "coverage"),
    quotes_from=(("covering_quote", "transcript_window"),),
    notes="the redundancy decision reads the transcript window and cannot run on an empty one",
)

REFERENCE_VISUAL = PromptContract(
    name="reference_visual_condition",
    inputs=MOMENT_INPUTS,
    required_out=("emit", "coverage", "content"),
    quotes_from=(("covering_quote", "transcript_window"),),
    closed_out=(("content", CONTENT_CLASSES), ("unspoken_form", FORM_PROPERTIES)),
    notes="the visual-content decision is a closed output, as coverage has its quotation",
)

CANDIDATES = PromptContract(
    name="candidates",
    inputs=MOMENT_INPUTS
    + (
        FieldSpec("directive", allowed=FOCUS_DIRECTIVES),
        FieldSpec("band", allowed=LENGTH_BANDS),
        FieldSpec("band_min"),
        FieldSpec("band_max"),
    ),
    required_out=("ad_text", "rung"),
    notes="the word bounds must reach the model, since candidates are checked against them",
)

CONTROLLED_PAIR = PromptContract(
    name="controlled_pair",
    inputs=MOMENT_INPUTS + (FieldSpec("axis", allowed=CATEGORIES),),
    required_out=("axis", "a", "b", "what_differs"),
    notes="the axis is the only claim a controlled pair makes",
)

_ELLIPSIS = re.compile(r"\s*(?:…|\.\.\.)\s*")


def _normalise(text: str) -> str:
    """Collapse whitespace and case, so a quotation is not rejected for re-wrapping."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def quote_is_grounded(quote: str, source: str) -> bool:
    """Whether `quote` is a verbatim span of `source`, allowing elision.

    "X ... Y" is accepted when X and Y both occur in the source, in that order:
    eliding is what quotation has always permitted. A paraphrase is not accepted,
    and neither are parts quoted in the reverse of their order in the source.
    """
    wanted, available = _normalise(quote), _normalise(source)
    if not wanted:
        return False
    if wanted in available:
        return True
    parts = [part for part in (piece.strip() for piece in _ELLIPSIS.split(wanted)) if part]
    if len(parts) < 2:
        return False
    position = 0
    for part in parts:
        found = available.find(part, position)
        if found < 0:
            return False
        position = found + len(part)
    return True


def check_inputs(contract: PromptContract, values: Mapping[str, Any]) -> None:
    """Raise unless every declared input is present and satisfies its specification."""
    problems = []
    for spec in contract.inputs:
        if spec.name not in values:
            problems.append(f"{spec.name}: missing")
            continue
        value = values[spec.name]
        text = "" if value is None else str(value)
        if spec.non_empty and not text.strip():
            problems.append(f"{spec.name}: empty")
        if spec.allowed and text.strip() and text.strip() not in spec.allowed:
            problems.append(f"{spec.name}: {text.strip()!r} not in {list(spec.allowed)}")
    if problems:
        note = f" -- {contract.notes}" if contract.notes else ""
        raise ContractError(f"inputs of {contract.name} break its contract: {problems}{note}")


def check_output(
    contract: PromptContract, reply: Mapping[str, Any], inputs: Mapping[str, Any]
) -> list[str]:
    """The ways a parsed reply breaks the contract. Empty when it does not."""
    violations = []
    for key in contract.required_out:
        if key not in reply:
            violations.append(f"missing output field {key!r}")
    for reply_key, input_name in contract.quotes_from:
        quote = reply.get(reply_key)
        if not quote:
            continue
        if not quote_is_grounded(quote, inputs.get(input_name) or ""):
            violations.append(
                f"{reply_key} is not a verbatim span of {input_name} ({str(quote)[:48]!r})"
            )
    for key, allowed in contract.closed_out:
        value = reply.get(key)
        if value is None:
            continue
        if str(value) not in allowed:
            violations.append(
                f"{key} outside its closed set ({str(value)[:32]!r} not in {allowed})"
            )
    return violations
