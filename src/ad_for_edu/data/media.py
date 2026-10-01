"""The artefacts of one lecture on disk, and how they are read.

Preprocessing writes four artefacts per lecture. Each has one reader here, and
each reader raises when its file is absent. A reader that returned an empty
value for a missing file would let a whole run proceed without speech or
without slide text and still look complete.

    transcript   <transcripts>/<lecture>.json   {"words": [{"start": s, "word": w}, ...]}
    speech gaps  <gaps>/<lecture>.json          {"silences": [{"start": s, "end": s}, ...]}
    slide text   <slide text>/<lecture>.json
                 {"frames": [{"timestamp": "hh:mm:ss", "text": t}, ...]}
    keyframes    <keyframes>/<lecture>/HHMMSS.jpg
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from ..core.errors import MissingSourceError
from ..core.timecodes import clock_to_seconds, packed_to_seconds


@dataclass(frozen=True)
class Word:
    start: float
    text: str


@dataclass(frozen=True)
class Gap:
    """A stretch without speech."""

    start: float
    end: float

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class SlideTextFrame:
    """The text read off one keyframe."""

    time: float
    text: str


@dataclass(frozen=True)
class Keyframe:
    time: int
    path: Path


@dataclass(frozen=True)
class MediaLayout:
    """The directories that hold the artefacts of all lectures."""

    transcripts: Path
    speech_gaps: Path
    slide_text: Path
    keyframes: Path
    videos: Path | None = None

    def lecture(self, lecture_id: str) -> LectureArtefacts:
        return LectureArtefacts(lecture_id, self)


def _read(path: Path, what: str, lecture: str) -> dict:
    if not path.is_file():
        raise MissingSourceError(f"no {what} for lecture {lecture!r} at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


class LectureArtefacts:
    """The artefacts of one lecture, read on first use and kept."""

    def __init__(self, lecture_id: str, layout: MediaLayout) -> None:
        self.lecture_id = lecture_id
        self.layout = layout

    @cached_property
    def words(self) -> tuple[Word, ...]:
        path = self.layout.transcripts / f"{self.lecture_id}.json"
        raw = _read(path, "transcript", self.lecture_id)
        return tuple(
            Word(float(entry["start"]), str(entry.get("word", "")).strip())
            for entry in raw.get("words", [])
            if entry.get("start") is not None
        )

    @cached_property
    def gaps(self) -> tuple[Gap, ...]:
        path = self.layout.speech_gaps / f"{self.lecture_id}.json"
        raw = _read(path, "speech gaps", self.lecture_id)
        return tuple(
            Gap(float(entry["start"]), float(entry["end"]))
            for entry in raw.get("silences", [])
            if entry.get("start") is not None and entry.get("end") is not None
        )

    @cached_property
    def slide_text(self) -> tuple[SlideTextFrame, ...]:
        path = self.layout.slide_text / f"{self.lecture_id}.json"
        raw = _read(path, "slide text", self.lecture_id)
        frames = []
        for entry in raw.get("frames", []):
            seconds = clock_to_seconds(entry.get("timestamp", ""))
            if seconds >= 0:
                frames.append(SlideTextFrame(seconds, (entry.get("text") or "").strip()))
        return tuple(sorted(frames, key=lambda frame: frame.time))

    @cached_property
    def keyframes(self) -> tuple[Keyframe, ...]:
        directory = self.layout.keyframes / self.lecture_id
        if not directory.is_dir():
            raise MissingSourceError(
                f"no keyframes for lecture {self.lecture_id!r} at {directory}"
            )
        frames = []
        for path in directory.iterdir():
            seconds = packed_to_seconds(path.name)
            if seconds is not None:
                frames.append(Keyframe(seconds, path))
        return tuple(sorted(frames, key=lambda frame: frame.time))

    @property
    def video(self) -> Path | None:
        """The recording, when the layout has a video directory and the file is there."""
        if self.layout.videos is None:
            return None
        for extension in ("mp4", "mkv", "webm", "mov"):
            candidate = self.layout.videos / f"{self.lecture_id}.{extension}"
            if candidate.is_file():
                return candidate
        return None
