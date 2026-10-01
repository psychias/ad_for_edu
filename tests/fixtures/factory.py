"""Synthetic lectures, rows and pairs for the tests.

Everything here is invented: two short fictional lectures about cell biology and
about sorting algorithms. No text comes from a real recording.
"""

from __future__ import annotations

from typing import Any

from ad_for_edu.compliance.models import LocalModels
from ad_for_edu.core.text import NO_SLIDE_TEXT, NO_SPEECH, token_set
from ad_for_edu.data.schema import MomentContext

LECTURE_A = "lec-a"
LECTURE_B = "lec-b"
WRITERS = ("writer-x", "writer-y", "writer-z")


def context(**fields: Any) -> MomentContext:
    """A moment context with sensible defaults; any field can be overridden."""
    values: dict[str, Any] = {
        "moment_id": f"{LECTURE_A}#0001",
        "type": "figure",
        "pause_after": 1.5,
        "slide_ocr": "Mitochondrion: outer membrane, inner membrane, cristae, matrix",
        "what_on_screen": "A labelled cross-section of a mitochondrion",
        "transcript_window": "so the energy of the cell is produced in this organelle",
        "renders_cursor": None,
    }
    values.update(fields)
    return MomentContext(**values)


def empty_context(**fields: Any) -> MomentContext:
    """A moment whose slide offered no text and whose window was silent."""
    return context(
        slide_ocr=NO_SLIDE_TEXT, what_on_screen="", transcript_window=NO_SPEECH, **fields
    )


def reference_rows() -> list[dict[str, Any]]:
    """Reference rows for six moments over two lectures, three writers each."""
    moments = [
        (LECTURE_A, 1, 12.0, "slide", "Cell structure", "Title slide with an outline"),
        (LECTURE_A, 2, 47.5, "figure", "Mitochondrion cristae matrix", "A labelled organelle"),
        (LECTURE_A, 3, 90.0, "pointing", "Nucleus nucleolus envelope", "A cell with its nucleus"),
        (LECTURE_B, 1, 8.0, "slide", "Sorting algorithms", "Title slide"),
        (LECTURE_B, 2, 33.0, "code", "def merge(left, right)", "A code listing"),
        (LECTURE_B, 3, 71.0, "chart", "Runtime comparison n log n", "A line chart of runtimes"),
    ]
    texts = {
        "writer-x": "The {topic} appears with its labels.",
        "writer-y": "A diagram introduces the {topic} and its parts.",
        "writer-z": None,
    }
    rows: list[dict[str, Any]] = []
    for lecture, index, time, kind, slide_text, on_screen in moments:
        moment = f"{lecture}#{index:04d}"
        for writer in WRITERS:
            template = texts[writer]
            rows.append(
                {
                    "output_id": f"{moment}::{writer}",
                    "time": str(time),
                    "type": kind,
                    "emit": template is not None,
                    "ad_text": template.format(topic=slide_text.split()[0]) if template else None,
                    "rung": 3 if template else None,
                    "rationale": "fixture",
                    "family": writer,
                    "transcript_window": "and now we turn to the next part",
                    "slide_ocr": slide_text,
                    "what_on_screen": on_screen,
                    "pause_after": 1.0,
                }
            )
    return rows


def predictions(texts: dict[str, str | None]) -> list[dict[str, Any]]:
    """Prediction rows for `{moment_id: text}`."""
    return [
        {"output_id": moment, "moment_id": moment, "emit": True, "ad_text": text, "forced": True}
        for moment, text in texts.items()
    ]


class FakeModels(LocalModels):
    """Deterministic stand-ins for the two local models.

    Contradiction is 1 when the hypothesis contains the word `not` and the premise
    does not, else 0. Similarity is the token overlap of the two texts.
    """

    def __init__(self) -> None:
        self.contradiction_calls: list[tuple[str, str]] = []
        self.similarity_calls: list[tuple[str, str]] = []

    def contradiction(self, premise: str, hypothesis: str) -> float:
        self.contradiction_calls.append((premise, hypothesis))
        denies = "not" in token_set(hypothesis) and "not" not in token_set(premise)
        return 1.0 if denies else 0.0

    def similarity(self, left: str, right: str) -> float:
        self.similarity_calls.append((left, right))
        first, second = token_set(left), token_set(right)
        if not first or not second:
            return 0.0
        return len(first & second) / len(first | second)
