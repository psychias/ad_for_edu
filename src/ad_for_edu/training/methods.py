"""The two ways a describer is trained.

**Supervised.** The describer is shown the moment and the reference answer, and
learns to produce the answer.

**Preference.** Starting from a supervised describer, it is shown two answers for
one moment, one preferred, and learns to prefer it. The describer it starts from
is also the reference it is kept close to.

The library that implements these renames and removes its settings between
releases, and passing one it does not have is not always an error: some releases
accept and ignore it, so the run trains under the default while the record says
otherwise. Every optional setting therefore goes through a check against the
fields the installed release actually has, and an unknown one stops the run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..core.errors import ContractError
from ..core.registry import Registry


@dataclass(frozen=True)
class TrainingSettings:
    """What a training cell is run with."""

    epochs: float
    learning_rate: float
    batch_size: int = 1
    gradient_accumulation: int = 8
    warmup_steps: int = 50
    max_sequence_length: int = 4096
    precision: str = "bf16"
    save_every: int = 100
    evaluate_every: int = 100
    log_every: int = 10
    keep_checkpoints: int = 20
    maximum_gradient_norm: float = 1.0
    weight_decay: float = 0.0
    seed: int = 0
    #: Settings of the method itself, checked against the library before use.
    method: Mapping[str, Any] = field(default_factory=dict)

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.gradient_accumulation


@dataclass(frozen=True)
class LoraSettings:
    """The adapter that is trained instead of the whole describer."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


def checked_settings(
    requested: Mapping[str, Any], available: set[str] | None, *, what: str
) -> dict[str, Any]:
    """The requested settings, having shown that the library has every one of them.

    A setting left unset is not a request and is dropped. `available` is passed in so
    that the check itself can be run without the library installed.
    """
    asked = {name: value for name, value in requested.items() if value is not None}
    if available is None:
        return asked
    unknown = sorted(name for name in asked if name not in available)
    if unknown:
        raise ContractError(
            f"{what} has no setting {unknown}. Passing it would be ignored and the cell would "
            "train under the default while the record said otherwise."
        )
    return asked


class TrainingMethod(ABC):
    """One way of training a describer."""

    name: ClassVar[str]
    #: Whether this method starts from a describer trained by another.
    starts_from_trained: ClassVar[bool] = False

    @abstractmethod
    def settings_class(self) -> Any:
        """The settings class of the library this method uses."""

    @abstractmethod
    def library_settings(
        self, settings: TrainingSettings, output: Any, available: set[str] | None = None
    ) -> dict[str, Any]: ...

    @abstractmethod
    def build_trainer(self, model: Any, data: Any, settings: Any, **parts: Any) -> Any: ...


TRAINING_METHODS: Registry[TrainingMethod] = Registry("training method", TrainingMethod)


def shared_settings(settings: TrainingSettings, output: Any) -> dict[str, Any]:
    """The settings both methods pass on, whatever the library calls them."""
    return {
        "output_dir": str(output),
        "num_train_epochs": settings.epochs,
        "learning_rate": settings.learning_rate,
        "per_device_train_batch_size": settings.batch_size,
        "gradient_accumulation_steps": settings.gradient_accumulation,
        "warmup_steps": settings.warmup_steps,
        "max_grad_norm": settings.maximum_gradient_norm,
        "weight_decay": settings.weight_decay,
        "logging_steps": settings.log_every,
        "save_strategy": "steps",
        "save_steps": settings.save_every,
        "save_total_limit": settings.keep_checkpoints,
        "eval_strategy": "steps",
        "eval_steps": settings.evaluate_every,
        "bf16": settings.precision == "bf16",
        "fp16": settings.precision == "fp16",
        "seed": settings.seed,
    }


@TRAINING_METHODS.register("sft")
class Supervised(TrainingMethod):
    """Learns to produce the reference answer for each moment."""

    name = "sft"

    def settings_class(self) -> Any:
        from trl import SFTConfig

        return SFTConfig

    def library_settings(
        self, settings: TrainingSettings, output: Any, available: set[str] | None = None
    ) -> dict[str, Any]:
        asked = {
            **shared_settings(settings, output),
            "max_length": settings.max_sequence_length,
            **dict(settings.method),
        }
        return checked_settings(asked, available, what="the supervised trainer")

    def build_trainer(self, model: Any, data: Any, settings: Any, **parts: Any) -> Any:
        from trl import SFTTrainer

        return SFTTrainer(
            model=model,
            args=settings,
            train_dataset=data.train,
            eval_dataset=data.development,
            **parts,
        )


@TRAINING_METHODS.register("dpo")
class Preference(TrainingMethod):
    """Learns to prefer one answer of a pair, starting from a supervised describer."""

    name = "dpo"
    starts_from_trained = True

    def settings_class(self) -> Any:
        from trl import DPOConfig

        return DPOConfig

    def library_settings(
        self, settings: TrainingSettings, output: Any, available: set[str] | None = None
    ) -> dict[str, Any]:
        asked = {
            **shared_settings(settings, output),
            "max_length": settings.max_sequence_length,
            **dict(settings.method),
        }
        return checked_settings(asked, available, what="the preference trainer")

    def build_trainer(self, model: Any, data: Any, settings: Any, **parts: Any) -> Any:
        from trl import DPOTrainer

        return DPOTrainer(
            model=model,
            args=settings,
            train_dataset=data.train,
            eval_dataset=data.development,
            **parts,
        )


def library_field_names(method: TrainingMethod) -> set[str]:
    """The settings the installed library actually has for this method."""
    import dataclasses

    return {field.name for field in dataclasses.fields(method.settings_class())}
