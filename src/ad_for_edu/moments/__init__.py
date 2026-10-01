"""Finding the moments of a lecture, and what a writer is told about each."""

from .build import (
    build_context_builder,
    build_detector,
    build_rung_policy,
    build_still_policy,
    build_window,
    load_classification_settings,
    load_detection_settings,
)
from .classification import (
    Classification,
    batch_by_lecture,
    batch_fault,
    batch_request_text,
    offered_types,
    parse_objects,
    prompt_for,
    read_answer,
    to_moment,
    vocabulary_fault,
)
from .context import ContextBuilder, WriterContext, pause_after, slide_text_at
from .detection import DetectionSettings, MomentDetector
from .rungs import RUNG_POLICIES, RUNG_WORDS, RungPolicy, deliverable_words, fits
from .stills import STILL_POLICIES, StillPlan, StillPolicy
from .windows import TRANSCRIPT_WINDOWS, Span, TranscriptWindow, words_in

__all__ = [
    "RUNG_POLICIES",
    "RUNG_WORDS",
    "STILL_POLICIES",
    "TRANSCRIPT_WINDOWS",
    "Classification",
    "ContextBuilder",
    "DetectionSettings",
    "MomentDetector",
    "RungPolicy",
    "Span",
    "StillPlan",
    "StillPolicy",
    "TranscriptWindow",
    "WriterContext",
    "batch_by_lecture",
    "batch_fault",
    "batch_request_text",
    "build_context_builder",
    "build_detector",
    "build_rung_policy",
    "build_still_policy",
    "build_window",
    "deliverable_words",
    "fits",
    "load_classification_settings",
    "load_detection_settings",
    "offered_types",
    "parse_objects",
    "pause_after",
    "prompt_for",
    "read_answer",
    "slide_text_at",
    "to_moment",
    "vocabulary_fault",
    "words_in",
]
