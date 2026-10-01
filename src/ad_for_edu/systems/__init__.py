"""The things a table compares: trained describers, the untrained one, the readout,
and the writers of the references."""

from .adapter import TrainedDescriber, UntrainedDescriber, conversation
from .base import DESCRIPTION_SYSTEMS, DescriptionSystem, EvalMoment
from .simple import ReferenceWriter, SlideTitleReadout, slide_title

__all__ = [
    "DESCRIPTION_SYSTEMS",
    "DescriptionSystem",
    "EvalMoment",
    "ReferenceWriter",
    "SlideTitleReadout",
    "TrainedDescriber",
    "UntrainedDescriber",
    "conversation",
    "slide_title",
]
