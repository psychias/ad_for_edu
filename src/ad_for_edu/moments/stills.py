"""Which stills of a moment a model is shown.

The event lies between the keyframe before it and the keyframe at it, so the
window opens before the event and closes after it. Two properties are built into
the window rather than hoped for.

**Part of the window falls after the event.** With a short fixed tail the event
sits near the end of a wide window, so only the last still shows what followed; a
model then reports the state that fills most of the window and rejects a real
change. The span is therefore widened until the part after the event is at least
one still interval long.

**The window covers enough time for slow movement to show.** A rotating model or
a playing animation changes little between two close stills, and a window too
narrow in elapsed time reads as static.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..core.errors import ContractError
from ..core.registry import Registry
from ..data.schema import CandidateEvent
from ..preprocessing.media import StillRequest, extract_stills


@dataclass(frozen=True)
class StillPlan:
    """The window of stills for one moment."""

    count: int
    start: float
    span: float
    #: How far the window opens before the event.
    back: float = 0.0

    @property
    def after_event(self) -> float:
        """How much of the span falls after the event."""
        return self.span - self.back

    @property
    def interval(self) -> float:
        """The longest a still interval can be: the span over one fewer than the stills."""
        return self.span / max(1, self.count - 1)

    @property
    def reaches_past_the_event(self) -> bool:
        return self.after_event >= self.interval - 1e-9


class StillPolicy(ABC):
    """Decides which stills of a moment a model sees."""

    @abstractmethod
    def plan(self, event: CandidateEvent) -> StillPlan: ...

    def stills(
        self,
        event: CandidateEvent,
        video: Path,
        target: Path,
        *,
        width: int = 960,
        font_file: str | None = None,
        require_count: bool = True,
    ) -> list[Path]:
        """Extract the stills of the plan. Fewer than planned is an error.

        A directory with fewer stills than the prompt describes would reach a model
        as evidence quietly smaller than what it is told it has, which is the same
        shape of failure as a reply cut short.
        """
        planned = self.plan(event)
        found = extract_stills(
            StillRequest(
                video=video,
                start=planned.start,
                span=planned.span,
                count=planned.count,
                target=target,
                width=width,
                font_file=font_file,
            )
        )
        if require_count and len(found) != planned.count:
            raise ContractError(
                f"{event.id}: {planned.count} stills were planned and {len(found)} were "
                f"written to {target}"
            )
        return found


STILL_POLICIES: Registry[StillPolicy] = Registry("still policy", StillPolicy)


@STILL_POLICIES.register("event_window")
class EventWindow(StillPolicy):
    """A window that opens before the event and reaches past it.

    How far back it opens follows the distance between the two keyframes that
    proposed the event, bounded below so a pair one second apart still gets a
    usable window, and above so the stills of a long pair are not so far apart
    that they show unrelated states.
    """

    def __init__(
        self,
        narrow_count: int = 3,
        wide_count: int = 5,
        wide_above_seconds: float = 3.0,
        minimum_back: float = 2.0,
        maximum_back: float = 20.0,
        minimum_span: float = 6.0,
        trail_seconds: float = 2.0,
        trail_fraction: float = 0.45,
    ) -> None:
        if min(narrow_count, wide_count) < 3:
            raise ValueError("a window needs at least three stills to have an interval after it")
        self.narrow_count = narrow_count
        self.wide_count = wide_count
        self.wide_above_seconds = wide_above_seconds
        self.minimum_back = minimum_back
        self.maximum_back = maximum_back
        self.minimum_span = minimum_span
        self.trail_seconds = trail_seconds
        self.trail_fraction = trail_fraction

    def plan(self, event: CandidateEvent) -> StillPlan:
        apart = max(0.0, float(event.detail.get("keyframes_apart") or 0.0))
        back = max(self.minimum_back, min(apart, self.maximum_back))
        count = self.wide_count if apart > self.wide_above_seconds else self.narrow_count
        # The part after the event must be at least one still interval, and the interval
        # is taken as the span over one fewer than the stills:
        #   span - back >= span / (count - 1)   holds from   span >= back (count - 1)/(count - 2)
        for_one_interval = back * (count - 1) / (count - 2)
        span = max(
            self.minimum_span,
            back + max(self.trail_seconds, self.trail_fraction * back),
            for_one_interval,
        )
        return StillPlan(count=count, start=max(0.0, float(event.t) - back), span=span, back=back)


@STILL_POLICIES.register("keyframe_window")
class KeyframeWindow(StillPolicy):
    """The keyframes already on disk near the moment, rather than new stills.

    This is what the trained describer is shown: keyframes of the lecture from a few
    seconds around the moment, without a burned-in time. When the window holds none,
    the nearest keyframe stands in, so a moment is never described from no picture at
    all.
    """

    def __init__(self, before: float = 2.0, after: float = 5.0, count: int = 3) -> None:
        self.before = before
        self.after = after
        self.count = count

    def plan(self, event: CandidateEvent) -> StillPlan:
        return StillPlan(
            count=self.count,
            start=max(0.0, float(event.t) - self.before),
            span=self.before + self.after,
            back=self.before,
        )

    def keyframes(
        self, event: CandidateEvent, available: list[tuple[int, Path]]
    ) -> list[Path]:
        """The keyframes in the window, or the nearest one when the window holds none."""
        low = float(event.t) - self.before
        high = float(event.t) + self.after
        inside = [path for time, path in sorted(available) if low <= time <= high]
        if inside:
            return inside[: self.count]
        if not available:
            return []
        nearest = min(available, key=lambda entry: abs(entry[0] - float(event.t)))
        return [nearest[1]]
