"""Wiring: from the settings files to a detector, a still policy and a context builder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.settings import check_keys, config_dir, load_yaml
from ..core.strategy import StrategySpec
from .classification import MINIMUM_OFFERED
from .context import ContextBuilder
from .detection import FRAME_CHANNELS, GAP_CHANNELS, DetectionSettings, MomentDetector
from .rungs import RUNG_POLICIES, RungPolicy
from .stills import STILL_POLICIES, StillPolicy
from .windows import TRANSCRIPT_WINDOWS, TranscriptWindow

DETECTION_FILE = "detection.yaml"
CLASSIFICATION_FILE = "classification.yaml"


@dataclass(frozen=True)
class DetectionConfig:
    frame_channels: tuple[Any, ...]
    gap_channel: Any
    settings: DetectionSettings


@dataclass(frozen=True)
class ClassificationConfig:
    model: str
    batch_size: int
    temperature: float
    reasoning_effort: str | None
    max_tokens: int
    attempts: int
    still_policy: StrategySpec
    transcript_window: StrategySpec
    still_width: int
    font_file: str | None


def load_detection_settings(path: str | Path | None = None) -> DetectionConfig:
    source = Path(path) if path else config_dir() / DETECTION_FILE
    raw = load_yaml(source)
    check_keys(
        raw,
        ("frame_channels", "gap_channel", "minimum_gap", "gap_horizon", "heartbeat_seconds"),
        required=("frame_channels", "gap_channel"),
        where=source.name,
    )
    settings = DetectionSettings(
        minimum_gap=float(raw.get("minimum_gap", 1.5)),
        gap_horizon=float(raw.get("gap_horizon", 12.0)),
        heartbeat_seconds=float(raw.get("heartbeat_seconds", 30.0)),
    )
    return DetectionConfig(
        frame_channels=tuple(StrategySpec.parse(entry) for entry in raw["frame_channels"]),
        gap_channel=StrategySpec.parse(raw["gap_channel"]),
        settings=settings,
    )


def load_classification_settings(path: str | Path | None = None) -> ClassificationConfig:
    source = Path(path) if path else config_dir() / CLASSIFICATION_FILE
    raw = load_yaml(source)
    check_keys(
        raw,
        (
            "model",
            "batch_size",
            "temperature",
            "reasoning_effort",
            "max_tokens",
            "attempts",
            "still_policy",
            "transcript_window",
            "still_width",
            "font_file",
        ),
        required=("model", "batch_size", "still_policy", "transcript_window"),
        where=source.name,
    )
    return ClassificationConfig(
        model=str(raw["model"]),
        batch_size=int(raw["batch_size"]),
        temperature=float(raw.get("temperature", 0.0)),
        reasoning_effort=raw.get("reasoning_effort") or None,
        max_tokens=int(raw.get("max_tokens", 8000)),
        attempts=int(raw.get("attempts", 3)),
        still_policy=StrategySpec.parse(raw["still_policy"]),
        transcript_window=StrategySpec.parse(raw["transcript_window"]),
        still_width=int(raw.get("still_width", 960)),
        font_file=raw.get("font_file") or None,
    )


def build_detector(settings: DetectionConfig | None = None) -> MomentDetector:
    settings = settings or load_detection_settings()
    return MomentDetector(
        frame_channels=[FRAME_CHANNELS.create_from(spec) for spec in settings.frame_channels],
        gap_channel=GAP_CHANNELS.create_from(settings.gap_channel),
        settings=settings.settings,
    )


def build_still_policy(spec: StrategySpec) -> StillPolicy:
    return STILL_POLICIES.create_from(spec)


def build_window(spec: StrategySpec) -> TranscriptWindow:
    return TRANSCRIPT_WINDOWS.create_from(spec)


def build_rung_policy(spec: StrategySpec) -> RungPolicy:
    return RUNG_POLICIES.create_from(spec)


def build_context_builder(
    window: StrategySpec,
    rung_policy: StrategySpec,
    *,
    maximum_slide_skew: float = 8.0,
    gap_horizon: float = 12.0,
) -> ContextBuilder:
    return ContextBuilder(
        window=build_window(window),
        rung_policy=build_rung_policy(rung_policy),
        maximum_slide_skew=maximum_slide_skew,
        gap_horizon=gap_horizon,
    )


__all__ = [
    "MINIMUM_OFFERED",
    "ClassificationConfig",
    "DetectionConfig",
    "build_context_builder",
    "build_detector",
    "build_rung_policy",
    "build_still_policy",
    "build_window",
    "load_classification_settings",
    "load_detection_settings",
]
