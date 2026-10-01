"""What a stage costs, per model and per call.

A cost is always the cost of something: a given model, at a given stage, with
the token shape that stage has. The shapes differ by an order of magnitude
between stages, because one stage sends thirty images per call and another sends
a line of text, and the prices differ by more than that between models. So an
estimate is built from a table that carries those conditions, and it takes the
number of calls per model, never one number for the stage.

A shape is either measured on real calls of that model at that stage, or assumed
for the stage as a whole. An estimate says which, because an assumed shape has to
be replaced by a measured one before a large run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..core.errors import ContractError, SettingsError
from ..core.settings import check_keys, config_dir, load_yaml
from ..core.spend import SpendEstimate
from .catalog import ModelCatalog

SETTINGS_FILE = "pricing.yaml"


@dataclass(frozen=True)
class Shape:
    """Input and output tokens of one call."""

    input_tokens: int
    output_tokens: int
    note: str = ""


@dataclass(frozen=True)
class StageShapes:
    """The assumed shape of a stage and the shapes measured per model."""

    stage: str
    assumed: Shape
    measured: Mapping[str, Shape] = field(default_factory=dict)

    def shape(self, model: str) -> tuple[Shape, bool]:
        """The shape to price `model` with, and whether it was measured."""
        if model in self.measured:
            return self.measured[model], True
        return self.assumed, False


class PriceTable:
    """Token shapes per stage, priced with the model catalogue."""

    def __init__(self, stages: Mapping[str, StageShapes], catalog: ModelCatalog) -> None:
        self.stages = dict(stages)
        self.catalog = catalog

    @classmethod
    def load(cls, catalog: ModelCatalog, path: str | Path | None = None) -> PriceTable:
        source = Path(path) if path else config_dir() / SETTINGS_FILE
        raw = load_yaml(source)
        check_keys(raw, ("stages",), required=("stages",), where=source.name)
        stages = {}
        for stage, entry in (raw["stages"] or {}).items():
            where = f"{source.name}:stages.{stage}"
            check_keys(
                entry, ("input_tokens", "output_tokens", "note", "measured"),
                required=("input_tokens", "output_tokens"), where=where,
            )
            measured = {}
            for model, shape in (entry.get("measured") or {}).items():
                if model not in catalog:
                    raise SettingsError(
                        f"{where}: a shape is measured for {model!r}, "
                        "which the model catalogue does not list"
                    )
                check_keys(
                    shape, ("input_tokens", "output_tokens", "note"),
                    required=("input_tokens", "output_tokens"), where=f"{where}.measured.{model}",
                )
                measured[model] = Shape(
                    int(shape["input_tokens"]), int(shape["output_tokens"]), shape.get("note", "")
                )
            stages[stage] = StageShapes(
                stage,
                Shape(
                    int(entry["input_tokens"]),
                    int(entry["output_tokens"]),
                    entry.get("note", ""),
                ),
                measured,
            )
        return cls(stages, catalog)

    def stage(self, name: str) -> StageShapes:
        try:
            return self.stages[name]
        except KeyError:
            raise ContractError(
                f"no token shape for stage {name!r}; known: {sorted(self.stages)}"
            ) from None

    def per_call(self, model: str, stage: str) -> float:
        shape, _measured = self.stage(stage).shape(model)
        return self.catalog.get(model).cost(shape.input_tokens, shape.output_tokens)

    def is_measured(self, model: str, stage: str) -> bool:
        return self.stage(stage).shape(model)[1]

    def estimate(
        self,
        stage: str,
        calls: Mapping[str, Mapping[str, int]],
        *,
        free_columns: tuple[str, ...] = (),
    ) -> SpendEstimate:
        """The estimate for `{row: {model: calls}}`.

        The number of calls is given per model. A single number for a stage would
        have to be priced at some model's rate, and choosing one is the error this
        table exists to prevent.
        """
        if not isinstance(calls, Mapping):
            raise ContractError(
                f"an estimate takes {{row: {{model: calls}}}}, got {type(calls).__name__}"
            )
        estimate = SpendEstimate(stage, free_columns=free_columns)
        for row, per_model in calls.items():
            if not isinstance(per_model, Mapping):
                raise ContractError(
                    f"{row}: expected {{model: calls}}, got {type(per_model).__name__}. "
                    "Give the number of calls for each model."
                )
            for model, count in per_model.items():
                estimate.add(
                    row,
                    model,
                    int(count),
                    self.per_call(model, stage),
                    measured=self.is_measured(model, stage),
                    note=self.stage(stage).shape(model)[0].note,
                )
        return estimate
