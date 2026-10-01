"""Operations on the recording itself, through ffmpeg.

Two conventions hold throughout.

**Stills carry their time.** A still is extracted with the time burned into the
corner, so a model reads the time instead of estimating it. The burned form is
`t = MM:SS.SS`, with minutes allowed past 59: a three-field form would be read as
hours, minutes, seconds by one reader and as minutes, seconds, hundredths by
another, and the two differ by a factor of sixty.

**Work already done is not repeated.** Every operation that writes files returns
at once when its output is there, so a stage can be resumed.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..core.errors import MissingSourceError, ProcessError
from ..core.timecodes import seconds_to_packed

#: A still smaller than this is a failed write, not an image.
MIN_STILL_BYTES = 512
#: A clip smaller than this is a failed write, not a clip.
MIN_CLIP_BYTES = 2048


def _run(arguments: list[str], *, what: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(arguments, capture_output=True, text=True, check=True)
    except FileNotFoundError as error:
        raise MissingSourceError(
            f"{arguments[0]} is not installed; it is needed to {what}"
        ) from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "")[-400:]
        if "Fontconfig" in detail or "Cannot load default config file" in detail:
            raise ProcessError(
                f"could not {what}: ffmpeg found no font to draw the time with. "
                "Name a font file in the settings."
            ) from error
        raise ProcessError(f"could not {what}: {detail}") from error


def duration(video: Path) -> float:
    """The length of the recording in seconds."""
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(video),
        ],
        what=f"read the length of {video.name}",
    )
    return float(result.stdout.strip())


def escape_for_filter(path: str | Path) -> str:
    """A path as an ffmpeg filter value.

    Separators are written forward and the colon of a drive letter is escaped once:
    a colon separates the parts of a filter, and the program is called directly, so
    no shell consumes an escape on the way.
    """
    text = str(path).replace(chr(92), "/")
    return text.replace(":", chr(92) + ":")


def timestamp_overlay(offset_seconds: float = 0.0, font_file: str | Path | None = None) -> str:
    """An ffmpeg filter that burns `t = MM:SS.SS` into every frame.

    `offset_seconds` is where the extracted stretch begins in the recording, so
    the burned time is the time in the lecture and not in the stretch.

    `font_file` names the font to draw with. Without it ffmpeg looks for a system
    font configuration, which not every machine has; where that lookup fails the
    filter does not fall back to another font, it crashes, so the font belongs in
    the settings rather than in the environment.
    """
    time = f"(t+{offset_seconds:.3f})"
    minutes = rf"%{{eif\:trunc({time}/60)\:d\:2}}"
    seconds = rf"%{{eif\:trunc(mod({time}\,60))\:d\:2}}"
    hundredths = rf"%{{eif\:trunc(mod({time}*100\,100))\:d\:2}}"
    font = f"fontfile='{escape_for_filter(font_file)}':" if font_file else ""
    return (
        f"drawtext={font}text='t = {minutes}\\:{seconds}.{hundredths}'"
        ":fontcolor=white:fontsize=28:box=1:boxcolor=black@0.75:boxborderw=8:x=12:y=12"
    )


#: The area the burned time covers, as (left, top, right, bottom) in pixels. The
#: probe for pointer motion must leave it out: it changes in every frame by design.
TIMESTAMP_BOX: tuple[int, int, int, int] = (0, 0, 210, 64)


def extract_audio(video: Path, target: Path) -> Path:
    """One channel at 16 kHz, which is what the transcriber and the gap detector read."""
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    _run(
        ["ffmpeg", "-y", "-i", str(video), "-ac", "1", "-ar", "16000", "-b:a", "32k", str(target)],
        what=f"extract the audio of {video.name}",
    )
    return target


def read_audio_samples(audio: Path):
    """The audio as float samples at 16 kHz."""
    import numpy

    try:
        completed = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", str(audio),
                "-ac", "1", "-ar", "16000", "-f", "f32le", "-",
            ],
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as error:
        raise MissingSourceError("ffmpeg is not installed; it is needed to decode audio") from error
    except subprocess.CalledProcessError as error:
        raise ProcessError(f"could not decode {audio.name}: {error.stderr[-400:]!r}") from error
    return numpy.frombuffer(completed.stdout, dtype=numpy.float32).copy()


def scene_changes(video: Path, threshold: float = 0.3) -> list[float]:
    """The times at which the picture changes by more than `threshold`."""
    import re

    completed = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(video),
            "-vf",
            f"select='gt(scene,{threshold})',metadata=print:file=-",
            "-an",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    output = (completed.stderr or "") + (completed.stdout or "")
    return [float(found) for found in re.findall(r"pts_time:([0-9.]+)", output)]


@dataclass(frozen=True)
class StillRequest:
    """Stills to extract from one stretch of the recording."""

    video: Path
    start: float
    span: float
    count: int
    target: Path
    width: int = 960
    timestamp: bool = True
    font_file: str | None = None

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"a request must ask for at least one still, got {self.count}")
        if self.span <= 0:
            raise ValueError(f"a request must span time, got {self.span}")


def extract_stills(request: StillRequest) -> list[Path]:
    """`count` stills evenly spread over the stretch, in one pass over the recording.

    Going straight from the recording to images, rather than cutting a clip first
    and sampling it, saves a second process and an encode of a clip nobody watches.
    """
    target = request.target
    expected = [target / f"{index:02d}.jpg" for index in range(request.count)]
    if all(path.is_file() and path.stat().st_size > MIN_STILL_BYTES for path in expected):
        return expected
    target.mkdir(parents=True, exist_ok=True)
    for existing in target.glob("*.jpg"):
        existing.unlink(missing_ok=True)
    start = max(0.0, request.start)
    filters = f"fps={request.count / max(request.span, 0.1):.6f},scale={int(request.width)}:-2"
    if request.timestamp:
        filters += "," + timestamp_overlay(start, request.font_file)
    _run(
        [
            "ffmpeg",
            "-y",
            # The seek precedes the input, so ffmpeg jumps to the stretch instead of
            # decoding the recording from its start.
            "-ss",
            f"{start:.3f}",
            "-i",
            str(request.video),
            "-t",
            f"{request.span:.3f}",
            "-vf",
            filters,
            "-frames:v",
            str(request.count),
            "-q:v",
            "3",
            # Numbering from zero, so the first still is 00.jpg and a reader that
            # counts from zero finds every still it was told about.
            "-start_number",
            "0",
            str(target / "%02d.jpg"),
        ],
        what=f"extract stills of {request.video.name} at {start:.1f} s",
    )
    return [path for path in expected if path.is_file() and path.stat().st_size > MIN_STILL_BYTES]


def extract_clip(
    video: Path,
    start: float,
    length: float,
    target: Path,
    *,
    width: int = 640,
    frames_per_second: int = 2,
) -> Path:
    """A short, small, silent clip of one moment, for a judge that watches.

    Small on purpose: it is uploaded on every call, and what a judge needs from it
    is the motion a single still cannot carry.
    """
    if target.is_file() and target.stat().st_size >= MIN_CLIP_BYTES:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{max(0.0, start):.3f}",
            "-i",
            str(video),
            "-t",
            f"{length:.3f}",
            "-an",
            "-vf",
            f"scale={int(width)}:-2",
            "-r",
            str(int(frames_per_second)),
            "-c:v",
            "libx264",
            str(target),
        ],
        what=f"extract a clip of {video.name} at {start:.1f} s",
    )
    return target


def keyframe_times(
    length: float,
    scene_times: list[float],
    *,
    grid_seconds: float = 30.0,
    minimum_gap: float = 1.0,
) -> list[float]:
    """The times to extract keyframes at: the scene changes, plus a regular grid.

    The grid catches a slide that changes too little to register as a scene change.
    Times closer together than `minimum_gap` collapse to the first of them.
    """
    grid = [float(second) for second in range(0, int(length), int(grid_seconds))]
    kept: list[float] = []
    last = float("-inf")
    for time in sorted(grid + list(scene_times)):
        if time - last >= minimum_gap:
            kept.append(time)
            last = time
    return kept


def extract_keyframes(video: Path, times: list[float], target: Path) -> list[tuple[int, Path]]:
    """One image per time, named by that time. Times already extracted are kept."""
    target.mkdir(parents=True, exist_ok=True)
    made: list[tuple[int, Path]] = []
    for time in times:
        path = target / seconds_to_packed(time)
        if not path.is_file():
            _run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{time:.3f}",
                    "-i",
                    str(video),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "3",
                    str(path),
                ],
                what=f"extract a keyframe of {video.name} at {time:.1f} s",
            )
        if path.is_file():
            made.append((int(time), path))
    return made
