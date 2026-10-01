"""Pairs built to differ on exactly one rule category.

A controlled pair states two descriptions of one moment that say the same things
and differ in one respect only: which rule category one of them breaks. That claim
is the whole value of the pair, so it is checked rather than trusted.

Two checks run on every pair, whatever its category.

**The sides are about as long as each other**, except on the category that is
about length, where they must not be. A difference in length is perceptible on its
own, so a pair that differs in length as well as in its category does not isolate
anything.

**The sides differ.** Two identical sides are not a pair.

The category that is about resolving what a lecturer points at needs a moment
where something is pointed at, so it is offered only on those. It also draws two
pairs from one moment rather than one, because too few such moments exist; the
consequence is that its pairs are not independent of one another, and an interval
over them has to treat the moment as the unit.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from ..core.registry import Registry
from ..core.text import word_count
from ..data.schema import CATEGORIES
from ..llm.replies import extract_json
from ..prompts import check_inputs, check_output, template
from ..prompts.contracts import CONTROLLED_PAIR

#: How much longer one side may be than the other, off the category about length.
MAXIMUM_WORD_GAP = 3


class ControlledAxis(ABC):
    """One rule category a controlled pair can differ on."""

    category: ClassVar[str]
    #: Moment types this category can be built on. Empty means any.
    requires_types: ClassVar[frozenset[str]] = frozenset()
    #: How many pairs one moment may supply on this category.
    pairs_per_moment: ClassVar[int] = 1

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        category = cls.__dict__.get("category")
        if category is not None and category not in CATEGORIES:
            raise TypeError(f"{cls.__name__}: category {category!r} is not one of {CATEGORIES}")

    def applies_to(self, moment_type: str | None) -> bool:
        return not self.requires_types or moment_type in self.requires_types

    def render(self, values: Mapping[str, Any]) -> str:
        check_inputs(CONTROLLED_PAIR, {**values, "axis": self.category})
        return template("controlled_pair").bind({**values, "axis": self.category})

    def parse(self, reply: str | None) -> dict[str, Any] | None:
        parsed = extract_json(reply)
        return dict(parsed) if isinstance(parsed, Mapping) else None

    def violations(self, pair: Mapping[str, Any]) -> list[str]:
        """The ways a pair fails what it claims about itself.

        Returned rather than raised: a pair that breaks the length rule is a finding
        to count over a run, and counting them is how "the category is what differs"
        stops being an assumption.
        """
        faults = check_output(CONTROLLED_PAIR, pair, {})
        stated = pair.get("axis")
        if stated != self.category:
            faults.append(f"the pair says its category is {stated!r}, not {self.category!r}")
        compliant = (pair.get("a") or "").strip()
        defective = (pair.get("b") or "").strip()
        if not compliant or not defective:
            faults.append("a side is empty")
            return faults
        if compliant == defective:
            faults.append("the two sides are the same")
        faults += self.length_faults(compliant, defective)
        return faults

    def length_faults(self, compliant: str, defective: str) -> list[str]:
        gap = abs(word_count(compliant) - word_count(defective))
        if gap > MAXIMUM_WORD_GAP:
            return [
                f"the sides differ by {gap} words, more than {MAXIMUM_WORD_GAP}, so length "
                "differs as well as the category"
            ]
        return []


CONTROLLED_AXES: Registry[ControlledAxis] = Registry("controlled axis", ControlledAxis)


@CONTROLLED_AXES.register("style")
class Style(ControlledAxis):
    """One side breaks the rules about tense, voice, person and meta-language."""

    category = "style"


@CONTROLLED_AXES.register("terminology")
class Terminology(ControlledAxis):
    """One side uses terms the slide does not use."""

    category = "terminology"


@CONTROLLED_AXES.register("length")
class Length(ControlledAxis):
    """One side runs past what the gap can deliver.

    Here the sides must differ in length: that is the manipulation. A difference too
    small to hear is not one.
    """

    category = "length"

    def length_faults(self, compliant: str, defective: str) -> list[str]:
        gap = abs(word_count(compliant) - word_count(defective))
        if gap <= MAXIMUM_WORD_GAP:
            return [
                f"the sides differ by {gap} words, which is the category of this pair, "
                "so the difference has to be audible"
            ]
        return []


@CONTROLLED_AXES.register("deixis")
class Deixis(ControlledAxis):
    """One side narrates the pointing instead of naming what is pointed at.

    Only moments where something is pointed at can carry this, and each supplies two
    pairs rather than one, because there are few such moments. Two pairs from one
    moment share a clip, a stretch of speech and a reference, so they are not two
    independent observations.
    """

    category = "deixis"
    requires_types = frozenset({"pointing"})
    pairs_per_moment = 2


@CONTROLLED_AXES.register("faithfulness")
class Faithfulness(ControlledAxis):
    """One side states something the slide does not support."""

    category = "faithfulness"


@CONTROLLED_AXES.register("non_redundancy")
class NonRedundancy(ControlledAxis):
    """One side repeats what the lecturer already said.

    The two checks pull against each other here: repeating the lecturer usually
    takes about as many words as not repeating them, but not always, so this
    category yields fewer usable pairs than the others. That is accepted rather than
    fixed by relaxing the length rule, which would let length differ too.
    """

    category = "non_redundancy"


def axes_for(moment_type: str | None, names: Sequence[str] | None = None) -> list[str]:
    """The categories that can be built on a moment of this type."""
    wanted = names or CONTROLLED_AXES.names()
    return [
        name for name in wanted if CONTROLLED_AXES.create(name).applies_to(moment_type)
    ]
