"""Which rules apply to a moment.

The applicability table has one row per rule: the moment types the rule applies
to, whether it applies to a moment that is described, to one that is left
silent, or to both, and the published number of the rule. The published number
is an index in its own right and not the position of the rule in the rule book.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.errors import ContractError
from ..core.io import read_csv
from .rulebook import RESOURCES, RuleBook

APPLICABILITY_TABLE = RESOURCES / "rule_applicability.csv"
ALL_TYPES = "all"


@dataclass(frozen=True)
class Applicability:
    """The applicability of one rule."""

    rule_id: str
    number: int
    category: str
    types: frozenset[str]
    when_described: bool
    when_silent: bool
    route: str
    checks: tuple[str, ...]
    summary: str

    def applies(self, moment_type: str | None, described: bool) -> bool:
        if ALL_TYPES not in self.types and moment_type not in self.types:
            return False
        return self.when_described if described else self.when_silent

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> Applicability:
        return cls(
            rule_id=row["rule_id"],
            number=int(row["rule_number"]),
            category=row["category"],
            types=frozenset(part.strip() for part in row["applies_to_types"].split(",")),
            when_described=row["applies_when_emit"].strip().lower() == "true",
            when_silent=row["applies_when_suppress"].strip().lower() == "true",
            route=row["scoring_route"],
            checks=tuple(part for part in row["scoring_checks"].split(";") if part),
            summary=row["rule_summary"],
        )


class ApplicabilityTable:
    """The applicability of every rule, by rule id."""

    def __init__(self, entries: Mapping[str, Applicability]) -> None:
        self.entries = dict(entries)

    @classmethod
    def load(cls, path: str | Path | None = None) -> ApplicabilityTable:
        rows = read_csv(Path(path) if path else APPLICABILITY_TABLE)
        entries: dict[str, Applicability] = {}
        for row in rows:
            entry = Applicability.from_row(row)
            if entry.rule_id in entries:
                raise ContractError(f"rule {entry.rule_id!r} appears twice in the table")
            entries[entry.rule_id] = entry
        return cls(entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, rule_id: str) -> Applicability:
        try:
            return self.entries[rule_id]
        except KeyError:
            raise ContractError(f"no rule {rule_id!r} in the applicability table") from None

    def numbers(self) -> dict[str, int]:
        """The published number of every rule."""
        return {rule_id: entry.number for rule_id, entry in self.entries.items()}

    def applicable(self, moment_type: str | None, described: bool) -> list[str]:
        """Ids of the rules that apply, in the order of their published numbers."""
        matching = [
            entry for entry in self.entries.values() if entry.applies(moment_type, described)
        ]
        return [entry.rule_id for entry in sorted(matching, key=lambda entry: entry.number)]

    def applicable_to(self, row: Mapping[str, Any]) -> list[str]:
        return self.applicable(row.get("type"), row.get("emit") is True)

    def check_against(self, book: RuleBook) -> None:
        """Raise unless table and book describe the same rules, numbered 1 to n, with the
        same scoring route."""
        in_table, in_book = set(self.entries), set(book.ids)
        if in_table != in_book:
            raise ContractError(
                f"table and book differ: only in table {sorted(in_table - in_book)}, "
                f"only in book {sorted(in_book - in_table)}"
            )
        numbers = sorted(entry.number for entry in self.entries.values())
        if numbers != list(range(1, len(numbers) + 1)):
            raise ContractError("the published numbers are not 1 to n without gaps or repeats")
        differing = sorted(
            rule.id for rule in book if rule.route != self.entries[rule.id].route
        )
        if differing:
            raise ContractError(f"scoring route differs between table and book: {differing}")
