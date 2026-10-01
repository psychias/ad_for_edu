"""Preparing one lecture, step by step.

Five steps produce what every later stage reads. Each writes one file or one
directory and is skipped when its output is already there, so preparing a corpus
can be stopped and resumed.

    audio        the sound, as one channel at 16 kHz
    transcript   the words, each with the second it begins at
    gaps         the stretches without speech
    keyframes    one image per scene change and per point of a regular grid
    slide text   the text read off each keyframe

The context holds the strategies; it does not choose them. Which transcriber, gap
detector and text reader to use is settled by the settings and passed in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.io import write_json
from ..data.media import MediaLayout
from . import media
from .slide_text import SlideTextReader
from .speech_gaps import SpeechGapDetector
from .transcription import Transcriber

STEPS: tuple[str, ...] = ("audio", "transcript", "gaps", "keyframes", "slide_text")


@dataclass(frozen=True)
class PreprocessingSettings:
    """What the steps need beyond the strategies."""

    grid_seconds: float = 30.0
    minimum_keyframe_gap: float = 1.0
    scene_threshold: float = 0.3


@dataclass
class StepResult:
    """What one step did."""

    step: str
    done: bool
    detail: dict[str, Any]


class LecturePreprocessor:
    """Runs the steps for one lecture at a time."""

    def __init__(
        self,
        layout: MediaLayout,
        audio_dir: Path,
        transcriber: Transcriber,
        gap_detector: SpeechGapDetector,
        slide_text_reader: SlideTextReader,
        settings: PreprocessingSettings | None = None,
    ) -> None:
        self.layout = layout
        self.audio_dir = audio_dir
        self.transcriber = transcriber
        self.gap_detector = gap_detector
        self.slide_text_reader = slide_text_reader
        self.settings = settings or PreprocessingSettings()

    # ------------------------------------------------------------ steps
    def audio(self, lecture_id: str, video: Path) -> StepResult:
        target = self.audio_dir / f"{lecture_id}.mp3"
        existed = target.is_file()
        media.extract_audio(video, target)
        return StepResult("audio", not existed, {"path": str(target)})

    def transcript(self, lecture_id: str, video: Path) -> StepResult:
        target = self.layout.transcripts / f"{lecture_id}.json"
        if target.is_file():
            return StepResult("transcript", False, {"path": str(target)})
        audio = self.audio_dir / f"{lecture_id}.mp3"
        transcript = self.transcriber.transcribe(lecture_id, audio, media.duration(video))
        write_json(target, transcript.as_dict())
        return StepResult("transcript", True, {"words": len(transcript.words)})

    def gaps(self, lecture_id: str, video: Path) -> StepResult:
        target = self.layout.speech_gaps / f"{lecture_id}.json"
        if target.is_file():
            return StepResult("gaps", False, {"path": str(target)})
        audio = self.audio_dir / f"{lecture_id}.mp3"
        found = self.gap_detector.gaps(audio, media.duration(video))
        write_json(
            target,
            {
                "lecture_id": lecture_id,
                "silences": [
                    {"start": gap.start, "end": gap.end, "length": round(gap.length, 2)}
                    for gap in found
                ],
            },
        )
        return StepResult("gaps", True, {"gaps": len(found)})

    def keyframes(self, lecture_id: str, video: Path) -> StepResult:
        target = self.layout.keyframes / lecture_id
        before = len(list(target.glob("*.jpg"))) if target.is_dir() else 0
        times = media.keyframe_times(
            media.duration(video),
            media.scene_changes(video, self.settings.scene_threshold),
            grid_seconds=self.settings.grid_seconds,
            minimum_gap=self.settings.minimum_keyframe_gap,
        )
        made = media.extract_keyframes(video, times, target)
        return StepResult("keyframes", len(made) > before, {"keyframes": len(made)})

    def slide_text(self, lecture_id: str, video: Path) -> StepResult:
        target = self.layout.slide_text / f"{lecture_id}.json"
        if target.is_file():
            return StepResult("slide_text", False, {"path": str(target)})
        frames = [
            (float(frame.time), frame.path)
            for frame in self.layout.lecture(lecture_id).keyframes
        ]
        readings = self.slide_text_reader.read(frames)
        write_json(
            target,
            {
                "lecture_id": lecture_id,
                "frames": [reading.as_dict() for reading in readings],
            },
        )
        return StepResult("slide_text", True, {"frames": len(readings)})

    # ------------------------------------------------------------ running
    def run(self, lecture_id: str, video: Path, steps: tuple[str, ...] = STEPS) -> list[StepResult]:
        """Run the named steps for one lecture, in the order they are listed here."""
        unknown = [step for step in steps if step not in STEPS]
        if unknown:
            raise ValueError(f"unknown steps {unknown}; known: {STEPS}")
        return [getattr(self, step)(lecture_id, video) for step in STEPS if step in steps]

    def paid_steps(self, steps: tuple[str, ...] = STEPS) -> tuple[str, ...]:
        """The named steps that cost money."""
        paid = {"transcript": self.transcriber.paid}
        return tuple(step for step in steps if paid.get(step, False))
