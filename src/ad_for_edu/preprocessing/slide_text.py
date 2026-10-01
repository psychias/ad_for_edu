"""Reading the text off the slides.

One reading per keyframe: the text the reader found, how sure it was, and where
each piece sat. The text of a keyframe is what a moment near that time is
grounded against, and the boxes are kept because the layout of a slide is
sometimes what a description has to convey.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.registry import Registry
from ..core.timecodes import seconds_to_clock


@dataclass(frozen=True)
class SlideReading:
    """What was read off one keyframe."""

    time: float
    text: str
    confidence: float = 0.0
    boxes: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": seconds_to_clock(self.time),
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "boxes": list(self.boxes),
        }


class SlideTextReader(ABC):
    """Reads the text of keyframes."""

    paid: bool = False

    @abstractmethod
    def read(self, frames: Sequence[tuple[float, Path]]) -> list[SlideReading]: ...


SLIDE_TEXT_READERS: Registry[SlideTextReader] = Registry("slide text reader", SlideTextReader)


@SLIDE_TEXT_READERS.register("easyocr")
class EasyOcrReader(SlideTextReader):
    """A local text reader over the keyframes, one image at a time."""

    paid = False

    def __init__(
        self,
        languages: Sequence[str] = ("en",),
        use_gpu: bool = False,
        reader_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.languages = tuple(languages)
        self.use_gpu = use_gpu
        self._reader_factory = reader_factory
        self._reader: Any = None

    def _load(self) -> Any:
        if self._reader is None:
            if self._reader_factory is not None:
                self._reader = self._reader_factory()
            else:
                import easyocr

                self._reader = easyocr.Reader(
                    list(self.languages), gpu=self.use_gpu, verbose=False
                )
        return self._reader

    def read(self, frames: Sequence[tuple[float, Path]]) -> list[SlideReading]:
        reader = self._load()
        readings = []
        for time, path in frames:
            # A frame the reader cannot open yields no text, not a failed lecture:
            # the reading of one keyframe is not worth losing the others over.
            try:
                found = reader.readtext(str(path))
            except Exception:  # noqa: BLE001
                found = []
            pieces = [
                {
                    "text": piece[1],
                    "confidence": round(float(piece[2]), 3),
                    "box": [[int(x), int(y)] for x, y in piece[0]],
                }
                for piece in found
            ]
            confidence = (
                round(sum(piece["confidence"] for piece in pieces) / len(pieces), 3)
                if pieces
                else 0.0
            )
            readings.append(
                SlideReading(
                    time=time,
                    text=" ".join(piece["text"] for piece in pieces),
                    confidence=confidence,
                    boxes=tuple(pieces),
                )
            )
        return readings
