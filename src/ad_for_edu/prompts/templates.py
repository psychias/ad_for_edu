"""Prompt templates: loaded from text files, bound to values, never sent half-bound.

A template that reaches a model with an unfilled placeholder still produces a
reply, and the reply looks like a model that did not follow instructions. So
binding is strict in both directions: every placeholder of the template must be
given a value, and every value given must belong to a placeholder.

Two binding styles exist, because the templates use braces in two ways.

    format    `{name}` is a placeholder and `{{` `}}` are literal braces.
    replace   only the named tokens are replaced; every other brace is literal.
"""

from __future__ import annotations

import json
import string
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..core.errors import ContractError, MissingSourceError

RESOURCES = Path(__file__).parent / "resources"
STYLES = ("format", "replace")


@dataclass(frozen=True)
class PromptTemplate:
    """A template and the placeholders it declares."""

    name: str
    text: str
    style: str = "format"
    tokens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.style not in STYLES:
            raise ContractError(f"template {self.name!r}: style must be one of {STYLES}")
        if self.style == "replace":
            absent = [token for token in self.tokens if "{" + token + "}" not in self.text]
            if absent:
                raise ContractError(f"template {self.name!r} lacks the tokens {absent}")
        elif self.tokens:
            raise ContractError(
                f"template {self.name!r}: a format template reads its placeholders from "
                "its text; do not declare tokens for it"
            )

    @property
    def placeholders(self) -> tuple[str, ...]:
        """The names this template must be bound with, in order of first appearance."""
        if self.style == "replace":
            return self.tokens
        seen: list[str] = []
        for _literal, name, _spec, _conversion in string.Formatter().parse(self.text):
            if name is None:
                continue
            if not name or not name.isidentifier():
                raise ContractError(
                    f"template {self.name!r}: {name!r} is not a plain placeholder name"
                )
            if name not in seen:
                seen.append(name)
        return tuple(seen)

    def bind(self, values: Mapping[str, Any] | None = None, /, **named: Any) -> str:
        """The template with every placeholder filled."""
        given = {**(values or {}), **named}
        expected = set(self.placeholders)
        missing = sorted(expected - set(given))
        if missing:
            raise ContractError(f"template {self.name!r} was not given {missing}")
        unknown = sorted(set(given) - expected)
        if unknown:
            raise ContractError(
                f"template {self.name!r} has no placeholders {unknown}; "
                f"it has {sorted(expected)}"
            )
        if self.style == "format":
            return self.text.format(**given)
        bound = self.text
        for token in self.tokens:
            bound = bound.replace("{" + token + "}", str(given[token]))
        return bound


@lru_cache(maxsize=64)
def read_text(name: str) -> str:
    """The bytes of a text resource, decoded. Nothing is stripped or normalised."""
    path = RESOURCES / f"{name}.txt"
    if not path.is_file():
        raise MissingSourceError(f"prompt resource not found: {path.name}")
    return path.read_bytes().decode("utf-8")


@lru_cache(maxsize=16)
def read_data(name: str) -> Any:
    path = RESOURCES / f"{name}.json"
    if not path.is_file():
        raise MissingSourceError(f"prompt resource not found: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))
