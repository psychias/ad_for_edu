"""Named strategies, one registry per family.

A family is an abstract base plus the concrete classes that implement it. The
registry maps a short name to each concrete class so that a settings file can
select a strategy by name, and so that the set of valid names is closed: an
unknown name fails and lists the names that exist.

    CHANNELS: Registry[FrameChannel] = Registry("frame channel", FrameChannel)

    @CHANNELS.register("slide_turnover")
    class SlideTurnover(FrameChannel): ...

    channel = CHANNELS.create("slide_turnover", jaccard_below=0.55)
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any, Generic, TypeVar

from .errors import DuplicateStrategyError, StrategyConfigError, UnknownStrategyError
from .strategy import StrategySpec

T = TypeVar("T")

_CATALOGUE: dict[str, Registry[Any]] = {}


class Registry(Generic[T]):
    """The strategies of one family, by name."""

    def __init__(self, family: str, base: type[T], *, listed: bool = True) -> None:
        if listed and family in _CATALOGUE:
            raise DuplicateStrategyError(f"family {family!r} already has a registry")
        self.family = family
        self.base = base
        self._classes: dict[str, type[T]] = {}
        if listed:
            _CATALOGUE[family] = self

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Class decorator that adds a concrete strategy under `name`."""

        def decorate(cls: type[T]) -> type[T]:
            if name in self._classes:
                raise DuplicateStrategyError(
                    f"{self.family} {name!r} is already registered "
                    f"({self._classes[name].__name__})"
                )
            if not (inspect.isclass(cls) and issubclass(cls, self.base)):
                raise StrategyConfigError(
                    f"{self.family} {name!r}: {cls!r} does not derive from {self.base.__name__}"
                )
            self._classes[name] = cls
            cls.strategy_name = name  # type: ignore[attr-defined]
            return cls

        return decorate

    def get(self, name: str) -> type[T]:
        try:
            return self._classes[name]
        except KeyError:
            known = ", ".join(self.names()) or "(none registered)"
            raise UnknownStrategyError(f"unknown {self.family} {name!r}; known: {known}") from None

    def create(self, name: str, /, **params: Any) -> T:
        """Build the strategy `name` with `params`, checking the parameter names first."""
        cls = self.get(name)
        signature = inspect.signature(cls)
        accepts_any = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
        )
        if not accepts_any:
            accepted = set(signature.parameters)
            unknown = sorted(set(params) - accepted)
            if unknown:
                raise StrategyConfigError(
                    f"{self.family} {name!r} does not accept {unknown}; "
                    f"accepted: {sorted(accepted) or '(no parameters)'}"
                )
        try:
            return cls(**params)
        except TypeError as exc:
            raise StrategyConfigError(f"{self.family} {name!r}: {exc}") from exc

    def create_from(self, spec: StrategySpec | str | Mapping[str, Any]) -> T:
        """Build a strategy from a settings entry: a bare name or `{name, params}`."""
        parsed = StrategySpec.parse(spec)
        return self.create(parsed.name, **dict(parsed.params))

    def create_all(self, specs: Any) -> list[T]:
        """Build an ordered list of strategies from a list of settings entries."""
        if isinstance(specs, (str, Mapping, StrategySpec)):
            specs = [specs]
        return [self.create_from(spec) for spec in specs]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._classes))

    def __contains__(self, name: object) -> bool:
        return name in self._classes

    def __len__(self) -> int:
        return len(self._classes)


def catalogue() -> dict[str, Registry[Any]]:
    """Every listed registry created so far, by family name."""
    return dict(sorted(_CATALOGUE.items()))
