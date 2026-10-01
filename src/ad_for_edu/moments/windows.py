"""The words spoken around a moment.

The window is recomputed wherever it is needed, from one function, with each
stage naming its own bounds. It is not stored: the stages disagree about how wide
the window should be, and a stored window would hand one stage the bounds of
another while its prompt said otherwise. A value that is plausible and wrong is
worse than one that is missing, because nothing looks broken.

A window with no words in it returns a marker rather than an empty string, so
that "the lecturer said nothing here" is distinguishable from "nobody looked".
The decision whether a description is needed reads the words spoken; an empty
string read as silence would answer that question with no evidence at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from ..core.registry import Registry
from ..core.text import NO_SPEECH
from ..data.media import Word


@dataclass(frozen=True)
class Span:
    """The bounds of a window, in seconds from the start of the recording."""

    start: float
    end: float

    @property
    def length(self) -> float:
        return self.end - self.start


def words_in(words: Sequence[Word], span: Span) -> str:
    """The words whose start falls in the span, or the marker for a silent window."""
    spoken = [word.text for word in words if span.start <= word.start < span.end and word.text]
    return " ".join(spoken) if spoken else NO_SPEECH


class TranscriptWindow(ABC):
    """The bounds of the window of one stage."""

    @abstractmethod
    def span(self, time: float, next_event: float | None = None) -> Span: ...

    def words(
        self, words: Sequence[Word], time: float, next_event: float | None = None
    ) -> str:
        return words_in(words, self.span(time, next_event))


TRANSCRIPT_WINDOWS: Registry[TranscriptWindow] = Registry("transcript window", TranscriptWindow)


@TRANSCRIPT_WINDOWS.register("symmetric")
class Symmetric(TranscriptWindow):
    """As much before the moment as after it. What the classifier reads."""

    def __init__(self, seconds: float = 8.0) -> None:
        self.seconds = seconds

    def span(self, time: float, next_event: float | None = None) -> Span:
        return Span(time - self.seconds, time + self.seconds)


@TRANSCRIPT_WINDOWS.register("fixed")
class Fixed(TranscriptWindow):
    """A stated stretch before the moment and a stated one after.

    What a writer reads. More after than before, because whether a description
    would repeat the lecturer depends most on what is said once the visual appears.
    """

    def __init__(self, before: float = 5.0, after: float = 8.0) -> None:
        self.before = before
        self.after = after

    def span(self, time: float, next_event: float | None = None) -> Span:
        return Span(time - self.before, time + self.after)


@TRANSCRIPT_WINDOWS.register("next_event")
class UntilNextEvent(TranscriptWindow):
    """From a stated stretch before the moment until the next event, or a cap.

    A lecturer may name what is on screen well after it appears. The window
    therefore runs to the next thing that happens, since anything said after that
    belongs to the next moment.
    """

    def __init__(self, before: float = 5.0, cap: float = 45.0) -> None:
        self.before = before
        self.cap = cap

    def span(self, time: float, next_event: float | None = None) -> Span:
        end = time + self.cap
        if next_event is not None and time < next_event < end:
            end = next_event
        return Span(time - self.before, end)
