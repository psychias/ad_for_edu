"""Wiring: from the settings file to a preprocessor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.settings import check_keys, config_dir, load_yaml
from ..core.strategy import StrategySpec
from ..data.catalog import DataCatalog
from ..data.media import MediaLayout
from .pipeline import LecturePreprocessor, PreprocessingSettings
from .slide_text import SLIDE_TEXT_READERS
from .speech_gaps import GAP_DETECTORS
from .transcription import TRANSCRIBERS

SETTINGS_FILE = "preprocessing.yaml"
_TOP_LEVEL = ("transcriber", "gap_detector", "slide_text_reader", "keyframes")


@dataclass(frozen=True)
class Settings:
    transcriber: StrategySpec
    gap_detector: StrategySpec
    slide_text_reader: StrategySpec
    keyframes: PreprocessingSettings


def load_preprocessing_settings(path: str | Path | None = None) -> Settings:
    source = Path(path) if path else config_dir() / SETTINGS_FILE
    raw = load_yaml(source)
    check_keys(raw, _TOP_LEVEL, required=_TOP_LEVEL, where=source.name)
    keyframes: dict[str, Any] = raw["keyframes"] or {}
    check_keys(
        keyframes,
        ("grid_seconds", "minimum_keyframe_gap", "scene_threshold"),
        where=f"{source.name}:keyframes",
    )
    return Settings(
        transcriber=StrategySpec.parse(raw["transcriber"]),
        gap_detector=StrategySpec.parse(raw["gap_detector"]),
        slide_text_reader=StrategySpec.parse(raw["slide_text_reader"]),
        keyframes=PreprocessingSettings(**keyframes),
    )


def media_layout(catalog: DataCatalog) -> MediaLayout:
    return MediaLayout(
        transcripts=catalog.local_path("transcripts"),
        speech_gaps=catalog.local_path("speech_gaps"),
        slide_text=catalog.local_path("slide_text"),
        keyframes=catalog.local_path("keyframes"),
        videos=catalog.local_path("videos"),
    )


def build_preprocessor(
    catalog: DataCatalog, settings: Settings | None = None
) -> LecturePreprocessor:
    settings = settings or load_preprocessing_settings()
    return LecturePreprocessor(
        layout=media_layout(catalog),
        audio_dir=catalog.local_path("audio"),
        transcriber=TRANSCRIBERS.create_from(settings.transcriber),
        gap_detector=GAP_DETECTORS.create_from(settings.gap_detector),
        slide_text_reader=SLIDE_TEXT_READERS.create_from(settings.slide_text_reader),
        settings=settings.keyframes,
    )
