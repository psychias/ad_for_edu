"""Finding the stretches of a lecture in which nobody speaks.

A description is delivered in a gap, so the gaps decide what is deliverable. A
gap is the time between one stretch of speech and the next, and only gaps of at
least a stated length are kept: shorter ones carry no description.

The detector reports the stretches that contain speech; the gaps are what lies
between them, including before the first and after the last.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..core.registry import Registry
from ..data.media import Gap
from .media import read_audio_samples

SAMPLE_RATE = 16000


class SpeechGapDetector(ABC):
    """Finds the gaps between speech in one lecture's audio."""

    paid: bool = False

    @abstractmethod
    def gaps(self, audio: Path, duration: float) -> list[Gap]: ...


GAP_DETECTORS: Registry[SpeechGapDetector] = Registry("speech gap detector", SpeechGapDetector)


def gaps_between(
    speech: Sequence[tuple[float, float]], duration: float, minimum: float
) -> list[Gap]:
    """The gaps of at least `minimum` seconds around the given stretches of speech.

    The stretch before the first speech and the one after the last are gaps too.
    """
    found: list[Gap] = []
    previous = 0.0
    for start, end in sorted(speech):
        if start - previous >= minimum:
            found.append(Gap(round(previous, 2), round(start, 2)))
        previous = max(previous, end)
    if duration - previous >= minimum:
        found.append(Gap(round(previous, 2), round(duration, 2)))
    return found


@GAP_DETECTORS.register("silero")
class SileroGaps(SpeechGapDetector):
    """A local voice-activity model reports the speech; the gaps are the rest."""

    paid = False

    def __init__(
        self,
        minimum_gap: float = 0.5,
        model_factory: Callable[[], Any] | None = None,
        detector: Callable[..., Any] | None = None,
    ) -> None:
        self.minimum_gap = minimum_gap
        self._model_factory = model_factory
        self._detector = detector
        self._model: Any = None

    def _load(self) -> tuple[Any, Callable[..., Any]]:
        if self._model is None:
            if self._model_factory is not None:
                self._model = self._model_factory()
            else:
                from silero_vad import load_silero_vad

                self._model = load_silero_vad()
        if self._detector is None:
            from silero_vad import get_speech_timestamps

            self._detector = get_speech_timestamps
        return self._model, self._detector

    def gaps(self, audio: Path, duration: float) -> list[Gap]:
        model, detect = self._load()
        samples = read_audio_samples(audio)
        try:
            import torch

            samples = torch.from_numpy(samples)
        except ImportError:
            pass
        stretches = detect(samples, model, sampling_rate=SAMPLE_RATE, return_seconds=True)
        speech = [
            (float(stretch["start"]), float(stretch["end"]))
            for stretch in stretches
            if stretch.get("start") is not None and stretch.get("end") is not None
        ]
        return gaps_between(speech, duration, self.minimum_gap)
