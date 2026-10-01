"""The detector: candidate events of one lecture, from its artefacts alone.

No model is called and the recording is not decoded. The detector walks the
keyframes in pairs, asks the frame channels in turn, and takes the first answer.
Then the gap channel adds the stretches without speech that no visual event
claimed.

Every visual candidate is matched with the next stretch of silence that begins
within a horizon, because a description that cannot be spoken has nowhere to go.
A visual candidate with no such stretch is kept and marked: the lecturer talks
straight through it, which is a fact about the lecture and not a failure of the
detector.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ...core.ids import moment_id
from ...data.media import Gap, LectureArtefacts
from ...data.schema import CandidateEvent
from .channels import FrameChannel, FramePair, GapChannel, LectureSignals


@dataclass(frozen=True)
class DetectionSettings:
    """What the detector needs beyond its channels."""

    #: The shortest stretch of silence a description could be spoken into.
    minimum_gap: float = 1.5
    #: How far after a visual event to look for such a stretch.
    gap_horizon: float = 12.0
    #: The interval at which keyframes are taken while nothing changes.
    heartbeat_seconds: float = 30.0


def usable_gaps(gaps: Sequence[Gap], minimum: float) -> list[Gap]:
    """The stretches of silence long enough to speak into, in order."""
    return sorted(
        (gap for gap in gaps if gap.length >= minimum), key=lambda gap: (gap.start, gap.end)
    )


def next_gap(gaps: Sequence[Gap], time: float, horizon: float) -> Gap | None:
    """The first usable stretch beginning within the horizon after `time`.

    The window opens slightly before the event, because a stretch of silence that
    begins as the slide changes is the one a description of it would use.
    """
    for gap in gaps:
        if time - 0.5 <= gap.start <= time + horizon:
            return gap
    return None


class MomentDetector:
    """Proposes the candidate events of a lecture."""

    def __init__(
        self,
        frame_channels: Sequence[FrameChannel],
        gap_channel: GapChannel,
        settings: DetectionSettings | None = None,
    ) -> None:
        if not frame_channels:
            raise ValueError("a detector needs at least one frame channel")
        self.frame_channels = tuple(frame_channels)
        self.gap_channel = gap_channel
        self.settings = settings or DetectionSettings()

    @property
    def channel_names(self) -> tuple[str, ...]:
        """The frame channels in the order they are asked, then the gap channel."""
        return tuple(channel.name for channel in self.frame_channels) + (self.gap_channel.name,)

    def detect(self, lecture: LectureArtefacts) -> list[CandidateEvent]:
        readings = lecture.slide_text
        gaps = usable_gaps(lecture.gaps, self.settings.minimum_gap)
        images = {frame.time: frame.path for frame in lecture.keyframes}
        signals = LectureSignals(heartbeat_seconds=self.settings.heartbeat_seconds)

        found: list[dict[str, Any]] = []
        for earlier, later in zip(readings, readings[1:], strict=False):
            pair = FramePair(
                earlier=earlier,
                later=later,
                earlier_image=images.get(int(earlier.time)),
                later_image=images.get(int(later.time)),
            )
            for channel in self.frame_channels:
                hit = channel.evaluate(pair, signals)
                if hit is None:
                    continue
                gap = next_gap(gaps, later.time, self.settings.gap_horizon)
                found.append(
                    {
                        "t": float(later.time),
                        "channel": hit.channel,
                        "visual": True,
                        "gap_start": round(gap.start, 2) if gap else None,
                        "gap_after": round(gap.length, 2) if gap else 0.0,
                        "keyframes_apart": round(pair.apart, 2),
                        "detail": hit.detail,
                    }
                )
                break

        claimed = {event["gap_start"] for event in found if event["gap_start"] is not None}
        for hit in self.gap_channel.propose(gaps, claimed):
            found.append(
                {
                    "t": float(hit.detail["t"]),
                    "channel": hit.channel,
                    "visual": False,
                    "gap_start": hit.detail["gap_start"],
                    "gap_after": hit.detail["gap_length"],
                    "keyframes_apart": None,
                    "detail": {},
                }
            )

        found.sort(key=lambda event: (event["t"], event["channel"]))
        return [
            CandidateEvent(
                id=moment_id(
                    lecture.lecture_id,
                    index,
                    from_gap=not event["visual"],
                ),
                lecture=lecture.lecture_id,
                t=event["t"],
                channel=event["channel"],
                visual=event["visual"],
                gap_after=event["gap_after"],
                detail={
                    "gap_start": event["gap_start"],
                    "keyframes_apart": event["keyframes_apart"],
                    **event["detail"],
                },
            )
            for index, event in enumerate(found, start=1)
        ]

    def summarise(self, events: Sequence[CandidateEvent]) -> dict[str, Any]:
        """How many candidates, by channel, and how many carry a stretch to speak into."""
        by_channel: dict[str, int] = {}
        for event in events:
            by_channel[event.channel] = by_channel.get(event.channel, 0) + 1
        visual = [event for event in events if event.visual]
        return {
            "candidates": len(events),
            "visual": len(visual),
            "speech_only": len(events) - len(visual),
            "by_channel": {name: by_channel.get(name, 0) for name in self.channel_names},
            "visual_without_a_gap": sum(
                1 for event in visual if event.detail.get("gap_start") is None
            ),
        }
