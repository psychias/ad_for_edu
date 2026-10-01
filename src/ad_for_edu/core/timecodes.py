"""Times, in one unit.

Every time inside the package is a number of seconds from the start of the
recording. Three textual forms occur at the edges and are converted here:

    clock     01:02:03  or  01:02:03.250     transcripts, slide-text frames
    packed    010203.jpg                     keyframe file names
    numeric   3723.25   or  "3723.25"        everything written by this package

A value that cannot be read returns `UNPARSEABLE` instead of zero, because zero is
a valid time and would place the value at the start of the recording.
"""

from __future__ import annotations

import re
from typing import Any

UNPARSEABLE = -1.0

_PACKED = re.compile(r"^(\d{2})(\d{2})(\d{2})(?:\.[A-Za-z0-9]+)?$")
_CLOCK = re.compile(r"^(\d{1,3}):(\d{1,2}):(\d{1,2}(?:\.\d+)?)$")
_MINUTES = re.compile(r"^(\d{1,4}):(\d{1,2}(?:\.\d+)?)$")


def clock_to_seconds(text: Any) -> float:
    """`hh:mm:ss` with optional fraction, to seconds."""
    match = _CLOCK.match(str(text).strip())
    if not match:
        return UNPARSEABLE
    hours, minutes, seconds = int(match.group(1)), int(match.group(2)), float(match.group(3))
    if minutes >= 60 or seconds >= 60:
        return UNPARSEABLE
    return hours * 3600 + minutes * 60 + seconds


def seconds_to_clock(seconds: float) -> str:
    """Seconds to `hh:mm:ss`, truncating the fraction."""
    whole = int(seconds)
    if whole < 0:
        raise ValueError(f"a time must not be negative, got {seconds}")
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}"


def minutes_to_seconds(text: Any) -> float:
    """`MM:SS`, with an optional fraction, to seconds.

    This is the form burned into the stills and the form a model answers with, and
    the minutes may run past 59: a lecture in its seventy-second minute reads
    `71:03.50`. There is no hour field, which is the point. A three-field time is
    read as hours, minutes, seconds by one reader and as minutes, seconds,
    hundredths by another, and the two differ by a factor of sixty.
    """
    match = _MINUTES.match(str(text).strip())
    if not match:
        return UNPARSEABLE
    minutes, seconds = int(match.group(1)), float(match.group(2))
    if seconds >= 60:
        return UNPARSEABLE
    return minutes * 60 + seconds


def packed_to_seconds(name: str) -> int | None:
    """`HHMMSS` or `HHMMSS.ext`, to seconds; None when the name has another shape."""
    match = _PACKED.match(str(name).strip())
    if not match:
        return None
    hours, minutes, seconds = (int(part) for part in match.groups())
    if minutes >= 60 or seconds >= 60:
        return None
    return hours * 3600 + minutes * 60 + seconds


def seconds_to_packed(seconds: float, extension: str = "jpg") -> str:
    """Seconds to the keyframe file name for that second."""
    whole = int(seconds)
    if whole < 0:
        raise ValueError(f"a time must not be negative, got {seconds}")
    stem = f"{whole // 3600:02d}{whole % 3600 // 60:02d}{whole % 60:02d}"
    return f"{stem}.{extension}" if extension else stem


def as_seconds(value: Any) -> float:
    """Read a time in any of the accepted forms."""
    if isinstance(value, bool) or value is None:
        return UNPARSEABLE
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else UNPARSEABLE
    text = str(value).strip()
    if not text:
        return UNPARSEABLE
    try:
        number = float(text)
    except ValueError:
        return clock_to_seconds(text)
    return number if number >= 0 else UNPARSEABLE


def is_time(value: float) -> bool:
    return value >= 0
