"""The things a table compares: a system is anything that describes moments.

Four kinds.

    adapter        a describer trained by this pipeline
    zero_shot      the same describer before it was trained
    slide_title    reads the title of the slide, whatever the moment
    reference      one of the writers whose descriptions are the references

The last two are not competitors but measurements of the scale. The readout is
what a metric scores when nothing is described but the slide is copied; a metric
that ranks it well is measuring something other than description. The writers show
what the scale looks like at the top, and they answered only the moments they chose
to answer, so their number is over their own moments and not over the shared set.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..core.registry import Registry
from ..inference.predictions import Prediction


@dataclass(frozen=True)
class EvalMoment:
    """One moment a system is asked to describe."""

    moment_id: str
    type: str = ""
    slide_ocr: str = ""
    what_on_screen: str = ""
    transcript_window: str = ""
    pause_after: float = 0.0
    time: float = 0.0
    keyframes: tuple[Any, ...] = ()
    fields: dict[str, Any] = field(default_factory=dict)


class DescriptionSystem(ABC):
    """Describes moments."""

    #: Whether it answers every moment it is asked about, or chooses.
    forced: ClassVar[bool] = True
    #: Whether its numbers are over the shared set of moments.
    on_shared_moments: ClassVar[bool] = True

    @abstractmethod
    def describe(self, moments: Sequence[EvalMoment]) -> list[Prediction]: ...


DESCRIPTION_SYSTEMS: Registry[DescriptionSystem] = Registry(
    "description system", DescriptionSystem
)
