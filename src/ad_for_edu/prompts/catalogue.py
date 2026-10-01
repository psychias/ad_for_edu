"""The prompts of the pipeline, by name.

The set is closed. Each entry names the text file of a template, how the
template is bound, and the contract of its inputs where it has one.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.errors import ContractError
from .contracts import (
    CANDIDATES,
    CONTROLLED_PAIR,
    REFERENCE,
    REFERENCE_VISUAL,
    PromptContract,
)
from .templates import PromptTemplate, read_text


@dataclass(frozen=True)
class PromptEntry:
    name: str
    purpose: str
    style: str = "format"
    tokens: tuple[str, ...] = ()
    contract: PromptContract | None = None
    #: Whether the shared prefix precedes this prompt.
    prefixed: bool = False


ENTRIES: tuple[PromptEntry, ...] = (
    PromptEntry(
        "classify_moments",
        "assign a moment type to each candidate event, or reject it",
        style="replace",
        tokens=("types",),
    ),
    PromptEntry(
        "reference_fixed_window",
        "write a reference description, or stay silent; transcript window of fixed length",
        contract=REFERENCE,
        prefixed=True,
    ),
    PromptEntry(
        "reference_next_event_window",
        "as above, with the window running to the next event",
        contract=REFERENCE,
        prefixed=True,
    ),
    PromptEntry(
        "reference_visual_condition",
        "as above, with the decision whether the moment has visual content to describe",
        contract=REFERENCE_VISUAL,
        prefixed=True,
    ),
    PromptEntry(
        "candidates_per_call",
        "write one candidate description under a focus directive",
        contract=CANDIDATES,
        prefixed=True,
    ),
    PromptEntry(
        "candidates_verbalized_sampling",
        "write five candidate descriptions in one call under a focus directive",
        contract=CANDIDATES,
        prefixed=True,
    ),
    PromptEntry(
        "controlled_pair",
        "write two descriptions that differ in exactly one rule category",
        contract=CONTROLLED_PAIR,
        prefixed=True,
    ),
    PromptEntry(
        "judge_video_pairwise",
        "choose between two descriptions after watching a clip of the moment",
        prefixed=True,
    ),
    PromptEntry("judge_text_pairwise", "choose between two descriptions from the text context"),
    PromptEntry("judge_sheet_pairwise", "choose between two descriptions as a rater would"),
    PromptEntry("judge_rubric", "grade a description in six categories and overall"),
    PromptEntry("judge_reference_rating", "rate a description against one reference"),
    PromptEntry(
        "judge_reference_rating_pooled", "rate a description against all references of a moment"
    ),
    # Sent as it is: the braces in it show the output format and are not placeholders.
    PromptEntry("student_system", "the system prompt of the trained describer", style="replace"),
)

_BY_NAME = {entry.name: entry for entry in ENTRIES}


def names() -> tuple[str, ...]:
    return tuple(entry.name for entry in ENTRIES)


def entry(name: str) -> PromptEntry:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ContractError(f"unknown prompt {name!r}; known: {sorted(_BY_NAME)}") from None


def template(name: str) -> PromptTemplate:
    """The template called `name`, read from its text file."""
    found = entry(name)
    return PromptTemplate(found.name, read_text(found.name), found.style, found.tokens)


def check_catalogue() -> None:
    """Raise unless every template loads and every contract input is a placeholder of
    its template, and the other way round."""
    for found in ENTRIES:
        loaded = template(found.name)
        if found.contract is None:
            continue
        placeholders, declared = set(loaded.placeholders), set(found.contract.input_names)
        if placeholders != declared:
            raise ContractError(
                f"prompt {found.name!r}: placeholders and contract differ: only in template "
                f"{sorted(placeholders - declared)}, only in contract "
                f"{sorted(declared - placeholders)}"
            )
