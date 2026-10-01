"""The detector and its channels. Importing the package registers the channels."""

from .channels import (
    FRAME_CHANNELS,
    GAP_CHANNELS,
    ChannelHit,
    FrameChannel,
    FramePair,
    GapChannel,
    LectureSignals,
    content_words,
)
from .detector import DetectionSettings, MomentDetector, next_gap, usable_gaps

__all__ = [
    "FRAME_CHANNELS",
    "GAP_CHANNELS",
    "ChannelHit",
    "DetectionSettings",
    "FrameChannel",
    "FramePair",
    "GapChannel",
    "LectureSignals",
    "MomentDetector",
    "content_words",
    "next_gap",
    "usable_gaps",
]
