"""The describers that can be trained, and how a cell is named.

A backbone names a released model and the exact revision of it. A revision is
required: a run against a name whose contents moved is a different experiment
wearing the same name.

A cell is one backbone, one input arm, one method and one seed. Its settings are
layered, each layer overriding the one before:

    defaults  <  method  <  backbone  <  the backbone's settings for that arm

An unknown key at any layer stops the run, with the known keys named.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import SettingsError
from ..core.runs import RunKey
from ..core.settings import check_keys, config_dir, load_yaml, merge

BACKBONE_KEYS = (
    "name",
    "model_id",
    "revision",
    "license",
    "trust_remote_code",
    "settings",
    "arms",
)
SETTINGS_KEYS = (
    "epochs",
    "learning_rate",
    "batch_size",
    "gradient_accumulation",
    "warmup_steps",
    "max_sequence_length",
    "precision",
    "save_every",
    "evaluate_every",
    "log_every",
    "keep_checkpoints",
    "maximum_gradient_norm",
    "weight_decay",
    "seed",
    "method",
    "lora",
    "processor_image_kwargs",
)


@dataclass(frozen=True)
class Backbone:
    """One describer that can be trained."""

    name: str
    model_id: str
    revision: str
    license: str = ""
    trust_remote_code: bool = False
    settings: Mapping[str, Any] = field(default_factory=dict)
    arms: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.revision:
            raise SettingsError(
                f"backbone {self.name!r} pins no revision. A run against a name whose contents "
                "moved is a different experiment under the same name."
            )

    def for_arm(self, arm: str) -> dict[str, Any]:
        """This backbone's settings for one input arm."""
        return merge(dict(self.settings), dict(self.arms.get(arm, {})))

    @classmethod
    def load(cls, path: str | Path) -> Backbone:
        source = Path(path)
        raw = load_yaml(source)
        check_keys(
            raw, BACKBONE_KEYS, required=("model_id", "revision"), where=source.name
        )
        settings = raw.get("settings") or {}
        check_keys(settings, SETTINGS_KEYS, where=f"{source.name}:settings")
        for arm, values in (raw.get("arms") or {}).items():
            check_keys(values or {}, SETTINGS_KEYS, where=f"{source.name}:arms.{arm}")
        return cls(
            name=str(raw.get("name") or source.stem),
            model_id=str(raw["model_id"]),
            revision=str(raw["revision"]),
            license=str(raw.get("license", "")),
            trust_remote_code=bool(raw.get("trust_remote_code", False)),
            settings=settings,
            arms={arm: dict(values or {}) for arm, values in (raw.get("arms") or {}).items()},
        )


def load_backbones(directory: str | Path | None = None) -> dict[str, Backbone]:
    """Every backbone of the settings directory, by name."""
    place = Path(directory) if directory else config_dir() / "training" / "backbones"
    if not place.is_dir():
        raise SettingsError(f"no backbones at {place}")
    found = {}
    for path in sorted(place.glob("*.yaml")):
        backbone = Backbone.load(path)
        if backbone.name in found:
            raise SettingsError(f"two backbones are called {backbone.name!r}")
        found[backbone.name] = backbone
    if not found:
        raise SettingsError(f"no backbones at {place}")
    return found


def settings_for(
    key: RunKey,
    backbone: Backbone,
    *,
    defaults: Mapping[str, Any],
    method: Mapping[str, Any],
) -> dict[str, Any]:
    """The settings of one cell, layered and checked."""
    for layer, where in ((defaults, "defaults"), (method, f"method {key.method}")):
        check_keys(layer, SETTINGS_KEYS, where=where)
    layered = merge(dict(defaults), dict(method), backbone.for_arm(key.arm))
    layered["seed"] = key.seed
    return layered
