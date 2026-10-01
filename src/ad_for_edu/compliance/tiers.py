"""The modes of the metric, as data.

A mode is a record: its name, the categories it scores, the categories it leaves
unscored by design, and whether it costs money. The mechanical mode scores the
four categories that need no model; the local mode adds the two that need one;
the rubric mode is a hosted judge and is never computed by the scorer in this
package.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..core.errors import SettingsError
from ..data.schema import CATEGORIES


@dataclass(frozen=True)
class Tier:
    """One mode of the metric."""

    name: str
    #: Settings entries of the components, one per scored category.
    components: tuple[Any, ...] = ()
    #: Categories this mode leaves unscored by design.
    abstains: frozenset[str] = frozenset()
    paid: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        outside = sorted(set(self.abstains) - set(CATEGORIES))
        if outside:
            raise SettingsError(f"mode {self.name!r}: {outside} are not categories")

    @classmethod
    def from_settings(cls, name: str, entry: Mapping[str, Any]) -> Tier:
        known = {"components", "abstains", "paid", "description"}
        unknown = sorted(set(entry) - known)
        if unknown:
            raise SettingsError(f"mode {name!r}: unknown keys {unknown}; known: {sorted(known)}")
        return cls(
            name=name,
            components=tuple(entry.get("components") or ()),
            abstains=frozenset(entry.get("abstains") or ()),
            paid=bool(entry.get("paid", False)),
            description=str(entry.get("description", "")),
        )


def tier_named(name: str, tiers: Mapping[str, Tier]) -> Tier:
    """The mode called `name`; an unknown name fails and lists the known ones."""
    try:
        return tiers[name]
    except KeyError:
        raise SettingsError(f"unknown mode {name!r}; known: {sorted(tiers)}") from None
