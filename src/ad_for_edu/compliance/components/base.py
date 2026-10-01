"""The component family: one scorer per rule category.

A component reads one description and the context of its moment and returns a
score in [0, 1], where 1 means the rules of its category are met. A component
whose rules cannot apply to a moment says so through `applies`, and the scorer
leaves it out of the mean instead of counting it as zero.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ...core.registry import Registry
from ...data.schema import CATEGORIES, MomentContext


class Component(ABC):
    """Scores one rule category of one description."""

    #: The rule category this component scores. One of `CATEGORIES`.
    category: ClassVar[str]
    #: True when the component needs a local model and is therefore absent from the
    #: mechanical mode.
    requires_models: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        category = cls.__dict__.get("category")
        if category is not None and category not in CATEGORIES:
            raise TypeError(f"{cls.__name__}: category {category!r} is not one of {CATEGORIES}")

    def applies(self, moment: MomentContext) -> bool:
        """Whether the rules of this category can apply to the moment at all."""
        return True

    @abstractmethod
    def score(self, text: str, moment: MomentContext) -> float:
        """Compliance of `text` with this category, in [0, 1]. `text` is never empty."""


COMPONENTS: Registry[Component] = Registry("compliance component", Component)
