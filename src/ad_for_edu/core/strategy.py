"""How a settings file names a strategy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .errors import StrategyConfigError

_SPEC_KEYS = frozenset({"name", "params"})


@dataclass(frozen=True)
class StrategySpec:
    """A strategy name and the parameters to build it with."""

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: StrategySpec | str | Mapping[str, Any]) -> StrategySpec:
        """Accept a spec, a bare name, or a mapping with `name` and optional `params`."""
        if isinstance(raw, StrategySpec):
            return raw
        if isinstance(raw, str):
            if not raw.strip():
                raise StrategyConfigError("a strategy name must not be empty")
            return cls(raw.strip())
        if isinstance(raw, Mapping):
            unknown = sorted(set(raw) - _SPEC_KEYS)
            if unknown:
                raise StrategyConfigError(
                    f"a strategy entry accepts {sorted(_SPEC_KEYS)}; got extra keys {unknown}"
                )
            if "name" not in raw:
                raise StrategyConfigError(f"a strategy entry needs a name; got {dict(raw)!r}")
            params = raw.get("params") or {}
            if not isinstance(params, Mapping):
                raise StrategyConfigError(
                    f"strategy {raw['name']!r}: params must be a mapping, "
                    f"got {type(params).__name__}"
                )
            return cls(str(raw["name"]), dict(params))
        raise StrategyConfigError(f"cannot read a strategy from {type(raw).__name__}: {raw!r}")
