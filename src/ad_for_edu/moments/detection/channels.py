"""The channels that propose candidate events.

A channel reads the artefacts of a lecture and says "something happened here".
It does not say what happened, and it cannot: a channel sees that the slide text
turned over or that pixels moved, not whether a teaching event took place. That
judgement belongs to the classifier, and whether a moment deserves a description
is decided later still, against the rule book.

Frame channels compare one keyframe with the next and are tried in order; the
first that matches names the candidate, and the rest are not consulted. The order
runs from the most specific signal to the least: a whole new slide, then an
increment added to the slide, then movement without a change of text.

The gap channel proposes the stretches without speech that no visual event
claimed. Those are moments a describer would rightly pass over, and a detector
that proposes only visual events cannot supply them.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from ...core.registry import Registry
from ...data.media import Gap, SlideTextFrame

_WORD = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")
#: Words too common to count as content of a slide.
_COMMON = frozenset(
    {
        "the", "and", "for", "are", "with", "that", "this", "from", "not", "but", "you",
        "can", "has", "have", "will", "was", "were", "its", "their", "which", "when",
    }
)


def content_words(text: str | None) -> set[str]:
    """The content words of a piece of slide text, lower-cased."""
    return {word.lower() for word in _WORD.findall(text or "")} - _COMMON


@dataclass
class FramePair:
    """Two consecutive keyframes of one lecture, and what they show."""

    earlier: SlideTextFrame
    later: SlideTextFrame
    earlier_image: Path | None = None
    later_image: Path | None = None

    @property
    def apart(self) -> float:
        """Seconds between the two keyframes."""
        return self.later.time - self.earlier.time

    @property
    def words_before(self) -> set[str]:
        return content_words(self.earlier.text)

    @property
    def words_after(self) -> set[str]:
        return content_words(self.later.text)

    @property
    def added_words(self) -> set[str]:
        return self.words_after - self.words_before

    @property
    def text_overlap(self) -> float:
        """How much of the slide text the two keyframes share; 1 when both are empty."""
        both = self.words_before | self.words_after
        if not both:
            return 1.0
        return len(self.words_before & self.words_after) / len(both)


@dataclass
class ChannelHit:
    """One channel's answer for one pair of keyframes."""

    channel: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class LectureSignals:
    """What the channels of one lecture share.

    `heartbeat` is the interval at which keyframes are taken while nothing changes.
    A pair that far apart was produced by the interval rather than by a change, so a
    channel that looks for movement leaves it alone.
    """

    heartbeat_seconds: float = 30.0
    #: The signature of the keyframe a movement channel looked at last.
    last_signature: Any = None


class FrameChannel(ABC):
    """Proposes a candidate from two consecutive keyframes."""

    name: ClassVar[str]

    @abstractmethod
    def evaluate(self, pair: FramePair, signals: LectureSignals) -> ChannelHit | None: ...


class GapChannel(ABC):
    """Proposes candidates from the stretches without speech."""

    name: ClassVar[str]

    @abstractmethod
    def propose(self, gaps: Sequence[Gap], claimed: set[float]) -> list[ChannelHit]: ...


FRAME_CHANNELS: Registry[FrameChannel] = Registry("frame channel", FrameChannel)
GAP_CHANNELS: Registry[GapChannel] = Registry("gap channel", GapChannel)


@FRAME_CHANNELS.register("slide_turnover")
class SlideTurnover(FrameChannel):
    """Most of the slide text has changed: a new slide, not an addition to one."""

    name = "slide_turnover"

    def __init__(self, overlap_below: float = 0.55) -> None:
        self.overlap_below = overlap_below

    def evaluate(self, pair: FramePair, signals: LectureSignals) -> ChannelHit | None:
        if not (pair.words_before | pair.words_after):
            return None
        if pair.text_overlap < self.overlap_below:
            return ChannelHit(
                self.name,
                {"text_overlap": round(pair.text_overlap, 3), "added_words": len(pair.added_words)},
            )
        return None


@FRAME_CHANNELS.register("added_words")
class AddedWords(FrameChannel):
    """Words have appeared on the slide: a build, a bullet, an overlay.

    A threshold, because a text reader disagrees with itself by a word or two
    between two readings of the same slide.
    """

    name = "added_words"

    def __init__(self, at_least: int = 3) -> None:
        self.at_least = at_least

    def evaluate(self, pair: FramePair, signals: LectureSignals) -> ChannelHit | None:
        added = pair.added_words
        if len(added) >= self.at_least:
            return ChannelHit(
                self.name,
                {"added_words": len(added), "text_overlap": round(pair.text_overlap, 3)},
            )
        return None


@FRAME_CHANNELS.register("pixel_motion")
class PixelMotion(FrameChannel):
    """The picture has changed while the slide text has not: a pointer, ink, an animation.

    Two properties of this channel are worth knowing.

    It leaves alone a pair of keyframes taken at the interval that runs while nothing
    changes: that pair is far apart because nothing happened, so any difference in it
    is the difference between two unrelated slides.

    The keyframe it compares against is the one this channel looked at last, which is
    not always the keyframe immediately before. Where an earlier channel claimed the
    intervening pairs, this channel did not read them, and the comparison reaches
    further back than one interval. Where this channel declined to look at all, it
    forgets what it recorded, so the next comparison it makes starts from the keyframe
    immediately before it.
    """

    name = "pixel_motion"

    def __init__(self, difference_at_least: float = 0.06, signature_size: int = 64) -> None:
        self.difference_at_least = difference_at_least
        self.signature_size = signature_size

    def signature(self, image: Path) -> list[int]:
        """The keyframe reduced to a small square of grey values."""
        from PIL import Image

        with Image.open(image) as opened:
            reduced = opened.convert("L").resize((self.signature_size, self.signature_size))
            # One byte per pixel in row order, which is what a grey image holds.
            return list(reduced.tobytes())

    @staticmethod
    def difference(first: Sequence[int], second: Sequence[int]) -> float:
        """The mean difference of two signatures, between 0 and 1."""
        if not first or not second or len(first) != len(second):
            return 0.0
        total = sum(abs(one - other) for one, other in zip(first, second, strict=True))
        return total / (len(first) * 255.0)

    def evaluate(self, pair: FramePair, signals: LectureSignals) -> ChannelHit | None:
        if pair.apart >= signals.heartbeat_seconds or None in (
            pair.later_image,
            pair.earlier_image,
        ):
            signals.last_signature = None
            return None
        later = self.signature(pair.later_image)
        earlier = signals.last_signature
        if earlier is None:
            earlier = self.signature(pair.earlier_image)
        signals.last_signature = later
        difference = self.difference(earlier, later)
        if difference >= self.difference_at_least:
            return ChannelHit(self.name, {"picture_difference": round(difference, 4)})
        return None


@GAP_CHANNELS.register("speech_gap")
class SpeechGapChannel(GapChannel):
    """The stretches without speech that no visual event claimed."""

    name = "speech_gap"

    def propose(self, gaps: Sequence[Gap], claimed: set[float]) -> list[ChannelHit]:
        proposed = []
        for gap in gaps:
            start = round(gap.start, 2)
            if start in claimed:
                continue
            proposed.append(
                ChannelHit(
                    self.name,
                    {"t": start, "gap_start": start, "gap_length": round(gap.length, 2)},
                )
            )
        return proposed
