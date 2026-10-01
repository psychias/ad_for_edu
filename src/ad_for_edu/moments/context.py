"""Everything a writer is told about a moment, computed from its time.

Six fields, each from one source and each with its own reader:

    type              the classifier's label for the moment
    transcript_window the words spoken around it, by the window of the stage
    slide_ocr         the text of the nearest keyframe
    pause_after       the longest usable gap beginning soon after it
    reachable_rung    the rung that gap allows
    described_so_far  what has already been described in this lecture

Every reader raises on a missing source or returns a stated marker. None of them
falls back to an empty string or to zero: a lookup that fails and returns a
plausible value produces a run in which the decision whether a description is
needed was made without the words it is supposed to read, and nothing looks wrong.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..core.text import NO_SLIDE_TEXT, NOTHING_DESCRIBED_YET
from ..data.media import LectureArtefacts
from .rungs import RungPolicy
from .windows import TranscriptWindow

#: A keyframe further from the moment than this shows a different slide.
MAX_SLIDE_SKEW = 8.0
#: How far after the moment to look for a usable gap.
GAP_HORIZON = 12.0


def slide_text_at(
    lecture: LectureArtefacts, time: float, maximum_skew: float = MAX_SLIDE_SKEW
) -> str:
    """The text of the keyframe nearest the moment, or the marker for none.

    A keyframe far from the moment shows another slide, and handing that over as
    context is worse than handing over nothing.
    """
    readings = lecture.slide_text
    if not readings:
        return NO_SLIDE_TEXT
    nearest = min(readings, key=lambda reading: abs(reading.time - time))
    if abs(nearest.time - time) > maximum_skew or not nearest.text:
        return NO_SLIDE_TEXT
    return nearest.text


def pause_after(
    lecture: LectureArtefacts, time: float, horizon: float = GAP_HORIZON
) -> float:
    """The longest unbroken silence beginning within the horizon after the moment.

    Unbroken, not the total: three short breaths are not a gap a description can be
    spoken into. Zero is a real answer here, since most moments offer no gap, which
    is why it must be computed rather than reached by a lookup that failed.
    """
    longest = 0.0
    for gap in lecture.gaps:
        if time - 0.5 <= gap.start <= time + horizon:
            longest = max(longest, gap.length)
    return round(longest, 2)


@dataclass(frozen=True)
class WriterContext:
    """The six fields, ready to fill a prompt with."""

    moment_id: str
    type: str
    transcript_window: str
    slide_ocr: str
    pause_after: float
    reachable_rung: int
    described_so_far: str

    def as_prompt_values(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "transcript_window": self.transcript_window,
            "slide_ocr": self.slide_ocr,
            "pause_after": self.pause_after,
            "reachable_rung": self.reachable_rung,
            "described_so_far": self.described_so_far,
        }


class ContextBuilder:
    """Builds the context of a moment from the artefacts of its lecture."""

    def __init__(
        self,
        window: TranscriptWindow,
        rung_policy: RungPolicy,
        *,
        maximum_slide_skew: float = MAX_SLIDE_SKEW,
        gap_horizon: float = GAP_HORIZON,
    ) -> None:
        self.window = window
        self.rung_policy = rung_policy
        self.maximum_slide_skew = maximum_slide_skew
        self.gap_horizon = gap_horizon

    def build(
        self,
        moment_id: str,
        moment_type: str,
        lecture: LectureArtefacts,
        time: float,
        *,
        next_event: float | None = None,
        described_so_far: Sequence[str] = (),
    ) -> WriterContext:
        pause = pause_after(lecture, time, self.gap_horizon)
        return WriterContext(
            moment_id=moment_id,
            type=moment_type,
            transcript_window=self.window.words(lecture.words, time, next_event),
            slide_ocr=slide_text_at(lecture, time, self.maximum_slide_skew),
            pause_after=pause,
            reachable_rung=self.rung_policy.reachable(pause),
            described_so_far=(
                "; ".join(described_so_far) if described_so_far else NOTHING_DESCRIBED_YET
            ),
        )
