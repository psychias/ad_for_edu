"""Turning reference rows into the examples a describer is trained on.

An example is a conversation: a system message carrying the compressed standard, a
user message carrying what the describer will have at inference (the keyframes, the
slide text, the words spoken, the type of the moment and the gap after it), and an
answer that is the reference row's own decision as structured text.

Two things the reference rows carry are deliberately kept out of the input.

**What the writer concluded.** A row records what was on screen, whether the words
already conveyed it and whether the pointing was clear. Those are conclusions, and
one of them is the answer restated: a describer given them would be told what to
decide. In the text-only arm the summary of what is on screen returns, because
there it stands in for the picture.

**The reference of another writer.** Each row is its own example.

The answer is one object with the decision, the description, the rung and the
reason. One head, not two: the decision whether to describe is never optimised on
its own, and a separate head would add an objective nothing measures.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from ..core.registry import Registry
from ..data.schema import ReferenceRow
from ..prompts import read_text

#: Where the keyframes of an example are named relative to.
DEFAULT_MEDIA_ROOT = "keyframes"


def system_prompt() -> str:
    """The compressed standard, the same text at training and at inference."""
    return read_text("student_system")


class InputArm(ABC):
    """What a describer is given about a moment."""

    carries_images: ClassVar[bool]

    @abstractmethod
    def user_content(
        self, row: ReferenceRow, keyframes: Sequence[str]
    ) -> list[dict[str, Any]]: ...

    def fields(self, row: ReferenceRow, *, with_screen: bool) -> str:
        """The text part: the moment as the describer will meet it."""
        lines = [f"Moment type: {row.type}"]
        lines.append(f"Lecturer said (+/-8s): {(row.transcript_window or '').strip()}")
        lines.append(f"On-screen text (OCR): {(row.slide_ocr or '').strip()}")
        if with_screen:
            lines.append(f"What is on screen: {(row.what_on_screen or '').strip()}")
        lines.append(f"Pause available after the moment: {row.pause_after or 0}s")
        return "\n".join(lines)


INPUT_ARMS: Registry[InputArm] = Registry("input arm", InputArm)


@INPUT_ARMS.register("multimodal")
class Multimodal(InputArm):
    """The keyframes, and the moment as text."""

    carries_images = True

    def user_content(self, row: ReferenceRow, keyframes: Sequence[str]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {"type": "image", "image": reference} for reference in keyframes
        ]
        content.append({"type": "text", "text": self.fields(row, with_screen=False)})
        return content


@INPUT_ARMS.register("text_only")
class TextOnly(InputArm):
    """No picture. The summary of what is on screen stands in for it."""

    carries_images = False

    def user_content(self, row: ReferenceRow, keyframes: Sequence[str]) -> list[dict[str, Any]]:
        return [{"type": "text", "text": self.fields(row, with_screen=True)}]


def answer_for(row: ReferenceRow) -> str:
    """The reference row's own decision, as the text the describer is trained to produce."""
    return json.dumps(
        {
            "emit": bool(row.emit),
            "ad_text": row.ad_text,
            "rung": row.rung,
            "rationale": (row.rationale or "").strip() or None,
        },
        ensure_ascii=False,
    )


def example_for(
    row: ReferenceRow,
    arm: InputArm,
    keyframes: Sequence[str] = (),
) -> dict[str, Any]:
    """One reference row as one training example."""
    return {
        "output_id": row.output_id,
        "moment_id": row.moment_id,
        "lecture": row.lecture,
        "type": row.type,
        "family": row.writer,
        "messages": [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": arm.user_content(row, keyframes)},
            {"role": "assistant", "content": answer_for(row)},
        ],
    }


def keyframe_references(
    lecture: str,
    available: Sequence[tuple[int, Path]],
    time: float,
    *,
    before: float = 2.0,
    after: float = 5.0,
    most: int = 3,
    media_root: str = DEFAULT_MEDIA_ROOT,
) -> list[str]:
    """How the keyframes of a moment are named in an example.

    Names, not pictures: the recordings cannot be passed on, so an example carries
    where its keyframes are and a reader of the set fetches them.
    """
    inside = [
        path for seconds, path in sorted(available) if time - before <= seconds <= time + after
    ]
    chosen = inside[:most]
    if not chosen and available:
        chosen = [min(available, key=lambda entry: abs(entry[0] - time))[1]]
    return [f"{media_root}/{lecture}/{path.name}" for path in chosen]


def as_typed(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The conversation with every message's content as a list of parts.

    Chat templates disagree: some accept a plain string, others require the list.
    The list form renders correctly under both.
    """
    typed = json.loads(json.dumps(list(messages)))
    for message in typed:
        content = message["content"]
        if isinstance(content, str):
            message["content"] = [{"type": "text", "text": content}]
        elif isinstance(content, dict):
            message["content"] = [
                {"type": "text", "text": json.dumps(content, ensure_ascii=False)}
            ]
    return typed
