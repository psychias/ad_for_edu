"""Turning the audio of a lecture into words with times.

A transcript is a list of words, each with the second it begins at. The times are
what every later stage needs: the words spoken around a moment are selected by
time, so a transcript without word times cannot serve the pipeline.

Long audio is sent in stretches and the times of each stretch are shifted back to
the time in the lecture, so the result is one transcript of the whole recording.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.errors import ProcessError
from ..core.registry import Registry
from ..core.secrets import OPENAI, credential
from .media import _run


@dataclass(frozen=True)
class Transcript:
    """The words of one lecture, with the second each begins at."""

    lecture_id: str
    words: tuple[dict[str, Any], ...]
    segments: tuple[dict[str, Any], ...] = ()
    language: str | None = None
    duration: float = 0.0
    model: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "lecture_id": self.lecture_id,
            "duration": round(self.duration, 3),
            "language": self.language,
            "model": self.model,
            "segments": list(self.segments),
            "words": list(self.words),
        }


class Transcriber(ABC):
    """Produces a transcript of one lecture's audio."""

    #: Whether transcribing costs money.
    paid: bool = True

    @abstractmethod
    def transcribe(self, lecture_id: str, audio: Path, duration: float) -> Transcript: ...


TRANSCRIBERS: Registry[Transcriber] = Registry("transcriber", Transcriber)

_RETRY_MARKERS = ("429", "500", "502", "503", "timeout", "Timeout")


@TRANSCRIBERS.register("hosted_whisper")
class HostedWhisper(Transcriber):
    """A hosted speech recognition service, asked for word times.

    The audio is cut into stretches short enough for one request, transcribed one
    at a time, and the times are put back on the clock of the lecture.
    """

    paid = True

    def __init__(
        self,
        model: str = "whisper-1",
        language: str = "en",
        chunk_seconds: float = 600.0,
        attempts: int = 5,
        credential_variable: str = OPENAI,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.model = model
        self.language = language
        self.chunk_seconds = chunk_seconds
        self.attempts = max(1, int(attempts))
        self.credential_variable = credential_variable
        self._client_factory = client_factory
        self._client: Any = None
        self._sleep: Callable[[float], None] = __import__("time").sleep

    def _connect(self) -> Any:
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                from openai import OpenAI

                self._client = OpenAI(api_key=credential(self.credential_variable))
        return self._client

    def _cut(self, audio: Path, start: float, target: Path) -> Path:
        _run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{self.chunk_seconds:.3f}",
                "-i",
                str(audio),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-b:a",
                "32k",
                str(target),
            ],
            what=f"cut {audio.name} at {start:.0f} s",
        )
        return target

    def _ask(self, part: Path) -> dict[str, Any]:
        client = self._connect()
        last: Exception | None = None
        for attempt in range(self.attempts):
            try:
                with part.open("rb") as handle:
                    reply = client.audio.transcriptions.create(
                        model=self.model,
                        file=handle,
                        response_format="verbose_json",
                        timestamp_granularities=["word", "segment"],
                        language=self.language,
                    )
            except Exception as error:  # noqa: BLE001 - classified below
                last = error
                if any(marker in str(error) for marker in _RETRY_MARKERS):
                    self._sleep(min(60.0, 10.0 * (attempt + 1)))
                    continue
                raise
            return reply.model_dump() if hasattr(reply, "model_dump") else dict(reply)
        raise ProcessError(
            f"transcription failed after {self.attempts} attempts: {str(last)[:200]}"
        )

    def transcribe(self, lecture_id: str, audio: Path, duration: float) -> Transcript:
        parts = max(1, math.ceil(duration / self.chunk_seconds))
        words: list[dict[str, Any]] = []
        segments: list[dict[str, Any]] = []
        language: str | None = None
        temporary: list[Path] = []
        try:
            for index in range(parts):
                offset = index * self.chunk_seconds
                part = audio
                if parts > 1:
                    part = self._cut(audio, offset, audio.with_name(f"{audio.stem}_{index}.mp3"))
                    temporary.append(part)
                reply = self._ask(part)
                language = language or reply.get("language")
                for segment in reply.get("segments") or []:
                    segments.append(
                        {
                            "start": round(segment["start"] + offset, 3),
                            "end": round(segment["end"] + offset, 3),
                            "text": str(segment["text"]).strip(),
                        }
                    )
                for word in reply.get("words") or []:
                    words.append(
                        {
                            "start": round(word["start"] + offset, 3),
                            "end": round(word["end"] + offset, 3),
                            "word": word["word"],
                        }
                    )
        finally:
            for part in temporary:
                part.unlink(missing_ok=True)
        return Transcript(
            lecture_id=lecture_id,
            words=tuple(words),
            segments=tuple(segments),
            language=language,
            duration=duration,
            model=self.model,
        )
