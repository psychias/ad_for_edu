"""Preparing lectures: audio, transcript, speech gaps, keyframes, slide text, pointer probe."""

from .build import build_preprocessor, load_preprocessing_settings, media_layout
from .cursor import MotionProbe, allowed_types, probe_frames, renders_cursor
from .pipeline import STEPS, LecturePreprocessor, PreprocessingSettings, StepResult
from .slide_text import SLIDE_TEXT_READERS, SlideReading, SlideTextReader
from .speech_gaps import GAP_DETECTORS, SpeechGapDetector, gaps_between
from .transcription import TRANSCRIBERS, Transcriber, Transcript

__all__ = [
    "GAP_DETECTORS",
    "SLIDE_TEXT_READERS",
    "STEPS",
    "TRANSCRIBERS",
    "LecturePreprocessor",
    "MotionProbe",
    "PreprocessingSettings",
    "SlideReading",
    "SlideTextReader",
    "SpeechGapDetector",
    "StepResult",
    "Transcriber",
    "Transcript",
    "allowed_types",
    "build_preprocessor",
    "gaps_between",
    "load_preprocessing_settings",
    "media_layout",
    "probe_frames",
    "renders_cursor",
]
