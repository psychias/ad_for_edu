"""Training a describer: the examples, the input arms, the methods, the cells."""

from .backbones import Backbone, load_backbones, settings_for
from .build import Grid, build_arm, build_method, load_grid, settings_of
from .data import TrainingData, as_preference_rows, prepare, split_by_moment
from .examples import (
    INPUT_ARMS,
    InputArm,
    answer_for,
    as_typed,
    example_for,
    keyframe_references,
    system_prompt,
)
from .methods import (
    TRAINING_METHODS,
    LoraSettings,
    TrainingMethod,
    TrainingSettings,
    checked_settings,
)

__all__ = [
    "INPUT_ARMS",
    "TRAINING_METHODS",
    "Backbone",
    "Grid",
    "InputArm",
    "LoraSettings",
    "TrainingData",
    "TrainingMethod",
    "TrainingSettings",
    "answer_for",
    "as_preference_rows",
    "as_typed",
    "build_arm",
    "build_method",
    "checked_settings",
    "example_for",
    "keyframe_references",
    "load_backbones",
    "load_grid",
    "prepare",
    "settings_for",
    "settings_of",
    "split_by_moment",
    "system_prompt",
]
