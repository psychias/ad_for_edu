"""The model catalogue: every model the pipeline calls, and what it may be used for.

    providers:
      <name>:
        strategy: <name of a registered provider>
        <parameter>: <value>
    models:
      <key>:
        provider: <name of a provider configuration above>
        model_id: <the name of the model at that provider>
        modalities: [text, image, video]
        roles: [writer, pair_judge, ...]
        price: {input: <per million tokens>, output: <per million tokens>}

A provider configuration is one way of reaching a service: the same strategy can
appear under two names with different parameters.

A model holds only the roles listed for it. The separation matters: a model that
scores descriptions as a metric must not also order the pairs a system is trained
on, or the metric would be measuring agreement with itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import ContractError, SettingsError
from ..core.settings import check_keys, config_dir, load_yaml
from .base import MODALITIES, PROVIDERS

SETTINGS_FILE = "models.yaml"

#: What a model can be asked to do in the pipeline.
ROLES: tuple[str, ...] = (
    "classifier",
    "writer",
    "pair_judge",
    "metric_judge",
    "annotator",
)

#: Roles that one model must not hold together.
EXCLUSIVE_ROLES: tuple[tuple[str, str], ...] = (("pair_judge", "metric_judge"),)

_ENTRY_KEYS = ("provider", "model_id", "modalities", "roles", "price", "options", "revision")
_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class ModelEntry:
    """One model of the catalogue."""

    key: str
    provider: str
    model_id: str
    modalities: frozenset[str]
    roles: frozenset[str]
    #: Price per token, as (input, output).
    price: tuple[float, float] = (0.0, 0.0)
    revision: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def free(self) -> bool:
        return self.price == (0.0, 0.0)

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return input_tokens * self.price[0] + output_tokens * self.price[1]

    def require_role(self, role: str) -> None:
        if role not in ROLES:
            raise ContractError(f"unknown role {role!r}; known: {ROLES}")
        if role not in self.roles:
            raise ContractError(
                f"model {self.key!r} may not act as {role}; its roles are {sorted(self.roles)}"
            )

    def require_modalities(self, needed: frozenset[str]) -> None:
        lacking = sorted(needed - self.modalities)
        if lacking:
            raise ContractError(
                f"model {self.key!r} cannot take {lacking}; it takes {sorted(self.modalities)}"
            )


class ModelCatalog:
    """The models of the pipeline, by key."""

    def __init__(
        self,
        entries: Mapping[str, ModelEntry],
        providers: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.entries = dict(entries)
        #: Construction parameters per provider name.
        self.providers = {name: dict(params) for name, params in (providers or {}).items()}

    @classmethod
    def load(cls, path: str | Path | None = None) -> ModelCatalog:
        source = Path(path) if path else config_dir() / SETTINGS_FILE
        raw = load_yaml(source)
        check_keys(raw, ("models", "providers"), required=("models",), where=source.name)
        entries = {
            key: _entry(key, value or {}, where=f"{source.name}:models.{key}")
            for key, value in (raw["models"] or {}).items()
        }
        if not entries:
            raise SettingsError(f"{source.name}: no models defined")
        return cls(entries, raw.get("providers") or {})

    def __contains__(self, key: object) -> bool:
        return key in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def keys(self) -> tuple[str, ...]:
        return tuple(self.entries)

    def get(self, key: str) -> ModelEntry:
        try:
            return self.entries[key]
        except KeyError:
            raise ContractError(f"unknown model {key!r}; known: {sorted(self.entries)}") from None

    def with_role(self, role: str) -> tuple[str, ...]:
        if role not in ROLES:
            raise ContractError(f"unknown role {role!r}; known: {ROLES}")
        return tuple(key for key, entry in self.entries.items() if role in entry.roles)

    def provider_spec(self, name: str) -> tuple[str, dict[str, Any]]:
        """The strategy and the parameters of the provider configuration `name`.

        A name without a configuration is taken to be a strategy used with its defaults.
        """
        params = dict(self.providers.get(name, {}))
        strategy = str(params.pop("strategy", name))
        return strategy, params


def _entry(key: str, raw: Mapping[str, Any], *, where: str) -> ModelEntry:
    check_keys(raw, _ENTRY_KEYS, required=("provider", "model_id", "roles"), where=where)
    modalities = frozenset(raw.get("modalities") or ("text",))
    outside = sorted(modalities - set(MODALITIES))
    if outside:
        raise SettingsError(f"{where}: modalities {outside} are not among {MODALITIES}")
    roles = frozenset(raw["roles"] or ())
    unknown = sorted(roles - set(ROLES))
    if unknown:
        raise SettingsError(f"{where}: roles {unknown} are not among {ROLES}")
    if not roles:
        raise SettingsError(f"{where}: a model needs at least one role")
    price = raw.get("price") or {}
    check_keys(price, ("input", "output"), where=f"{where}.price")
    return ModelEntry(
        key=key,
        provider=str(raw["provider"]),
        model_id=str(raw["model_id"]),
        modalities=modalities,
        roles=roles,
        price=(
            float(price.get("input", 0.0)) / _PER_MILLION,
            float(price.get("output", 0.0)) / _PER_MILLION,
        ),
        revision=raw.get("revision"),
        options=dict(raw.get("options") or {}),
    )


def check_providers(catalog: ModelCatalog) -> None:
    """Raise unless every model names a registered provider that carries its modalities,
    and no model holds two roles that exclude each other."""
    for entry in catalog.entries.values():
        for first, second in EXCLUSIVE_ROLES:
            if first in entry.roles and second in entry.roles:
                raise SettingsError(
                    f"model {entry.key!r} is listed as both {first} and {second}; "
                    "one model must not hold both"
                )
        strategy, _params = catalog.provider_spec(entry.provider)
        provider = PROVIDERS.get(strategy)
        lacking = sorted(entry.modalities - provider.modalities)
        if lacking:
            raise SettingsError(
                f"model {entry.key!r} lists {lacking}, which provider "
                f"{entry.provider!r} cannot carry"
            )
        if entry.free and provider.paid:
            raise SettingsError(
                f"model {entry.key!r} has no price but its provider {entry.provider!r} is paid"
            )
