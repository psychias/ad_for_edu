"""The rule book: 45 rules for audio description of slide-based lecture recordings.

Fields of a rule:

    id, section     identifier and position in the book
    element         what the rule is about: `all`, or an element such as `charts`
    type            describe | dont_describe | how | length | cross_cutting | uncertainty
    tag             DOC, grounded in a cited guideline; PROV, provisional
    rule            the text of the rule
    citation        the source the rule rests on, with `citation_extra` for further ones
    basis           for a provisional rule, how it was derived
    open_question   what is still undecided about the rule
    scoring.route   mechanical | judge | penalty | unscored
    scoring.checks  the checks that score a mechanical rule
    blv_review      the verdict of a blind or low-vision reviewer, where one was given

The file is sent to the writing models verbatim, inside the shared prefix. `text`
is therefore part of the interface, not only the parsed `rules`.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from ..core.errors import ContractError, MissingSourceError

RESOURCES = Path(__file__).parent / "resources"
RULE_BOOK = RESOURCES / "rules_for_slides.yaml"

ROUTES: tuple[str, ...] = ("mechanical", "judge", "penalty", "unscored")
TAGS: tuple[str, ...] = ("DOC", "PROV")

#: Any token shaped like a rule id, whatever its family. The family is left open on
#: purpose: a citation of a family that does not exist is exactly what must be caught.
_RULE_ID = re.compile(r"\b([a-z][a-z_]*_\d{3})\b")
#: A range such as `deixis_001..006`, whose upper end is a bare number.
_RULE_RANGE = re.compile(
    r"\b([a-z][a-z_]*)_(\d{3})\s*(?:\.\.|---|--|-|–|—|to)\s*(\d{3})\b"
)


@dataclass(frozen=True)
class Rule:
    """One rule, with typed access to the fields the package reads."""

    fields: Mapping[str, Any]

    @property
    def id(self) -> str:
        return str(self.fields["id"])

    @property
    def section(self) -> str:
        return str(self.fields.get("section", ""))

    @property
    def element(self) -> str:
        return str(self.fields.get("element", ""))

    @property
    def type(self) -> str:
        return str(self.fields.get("type", ""))

    @property
    def tag(self) -> str:
        return str(self.fields.get("tag", ""))

    @property
    def text(self) -> str:
        return str(self.fields.get("rule", "")).strip()

    @property
    def documented(self) -> bool:
        return self.tag == "DOC"

    @property
    def route(self) -> str:
        return str((self.fields.get("scoring") or {}).get("route", ""))

    @property
    def checks(self) -> tuple[str, ...]:
        return tuple((self.fields.get("scoring") or {}).get("checks") or ())

    @property
    def reviewed(self) -> bool:
        return bool(self.fields.get("blv_review"))

    def grounding(self) -> str:
        """The sources and open points of the rule, one per line."""
        parts = []
        if self.fields.get("citation"):
            parts.append(f"- {self.fields['citation']}")
        for extra in self.fields.get("citation_extra") or []:
            parts.append(f"- {extra}")
        if self.fields.get("basis"):
            parts.append(f"- basis: {self.fields['basis']}")
        if self.fields.get("open_question"):
            parts.append(f"- open question: {self.fields['open_question']}")
        if not parts:
            return "(no external citation; rule is self-grounding)"
        return "\n".join(parts)


class RuleBook:
    """The parsed rule book together with the text it was parsed from."""

    def __init__(self, text: str) -> None:
        import yaml

        self.text = text
        parsed = yaml.safe_load(text)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("rules"), list):
            raise ContractError("the rule book has no list of rules")
        self.version = str(parsed.get("version", ""))
        self.scope = str(parsed.get("scope", ""))
        self.rules: tuple[Rule, ...] = tuple(Rule(entry) for entry in parsed["rules"])

    @classmethod
    def load(cls, path: str | Path | None = None) -> RuleBook:
        source = Path(path) if path else RULE_BOOK
        if not source.is_file():
            raise MissingSourceError(f"rule book not found: {source}")
        return cls(source.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self) -> Iterator[Rule]:
        return iter(self.rules)

    def __contains__(self, rule_id: object) -> bool:
        return rule_id in self._by_id

    @cached_property
    def _by_id(self) -> dict[str, Rule]:
        return {rule.id: rule for rule in self.rules}

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(rule.id for rule in self.rules)

    def get(self, rule_id: str) -> Rule:
        try:
            return self._by_id[rule_id]
        except KeyError:
            raise ContractError(f"no rule {rule_id!r} in the rule book") from None

    def tags(self) -> Counter[str]:
        return Counter(rule.tag for rule in self.rules)

    def routes(self) -> Counter[str]:
        return Counter(rule.route for rule in self.rules)

    def reviewed(self) -> tuple[Rule, ...]:
        return tuple(rule for rule in self.rules if rule.reviewed)

    def dangling_references(self) -> list[str]:
        """Rule ids cited in any text field of any rule that the book does not define."""
        known = set(self.ids)
        dangling: list[str] = []
        for rule in self.rules:
            for name, value in rule.fields.items():
                if not isinstance(value, str):
                    continue
                for cited in sorted(set(_RULE_ID.findall(value))):
                    if cited not in known:
                        dangling.append(f"{rule.id}.{name} -> {cited}")
                for family, _first, last in _RULE_RANGE.findall(value):
                    if f"{family}_{last}" not in known:
                        dangling.append(f"{rule.id}.{name} -> {family}_{last} (end of a range)")
        return dangling

    def check(self) -> None:
        """Raise when the book is not internally consistent."""
        repeated = sorted(rule_id for rule_id, count in Counter(self.ids).items() if count > 1)
        if repeated:
            raise ContractError(f"rule ids defined more than once: {repeated}")
        bad_tags = sorted({rule.id for rule in self.rules if rule.tag not in TAGS})
        if bad_tags:
            raise ContractError(f"rules with a tag outside {TAGS}: {bad_tags}")
        bad_routes = sorted({rule.id for rule in self.rules if rule.route not in ROUTES})
        if bad_routes:
            raise ContractError(f"rules with a scoring route outside {ROUTES}: {bad_routes}")
        empty = sorted(rule.id for rule in self.rules if not rule.text)
        if empty:
            raise ContractError(f"rules without text: {empty}")
        dangling = self.dangling_references()
        if dangling:
            raise ContractError(f"rule ids cited but not defined: {dangling}")

    def summary(self) -> dict[str, Any]:
        tags, routes = self.tags(), self.routes()
        return {
            "version": self.version,
            "rules": len(self),
            "documented": tags.get("DOC", 0),
            "provisional": tags.get("PROV", 0),
            "routes": {route: routes.get(route, 0) for route in ROUTES},
            "reviewed": len(self.reviewed()),
        }
