"""What a system produced, as it is written out and read back.

One row per moment, never one per reference. A moment has several reference rows,
one per writer, and the describer is given the same thing for each of them: asking
once per reference would produce the same description several times and count it as
several moments.

Reading a file of predictions checks two things. Every row names the moment it is
for, and no moment appears twice: a moment answered twice would be scored twice,
and which of the two a table used would depend on the order of the rows.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..core.errors import ContractError
from ..core.io import iter_jsonl
from ..data.schema import is_unfilled

FIELDS = ("output_id", "moment_id", "emit", "ad_text", "rung", "rationale", "forced", "how")


@dataclass
class Prediction:
    """One system's answer for one moment."""

    output_id: str
    moment_id: str
    emit: bool | None
    ad_text: str | None
    rung: int | None = None
    rationale: str | None = None
    #: Whether the describer was made to answer rather than left free to stay silent.
    forced: bool = False
    #: How the reply was read.
    how: str = "strict"

    @property
    def filled(self) -> bool:
        return not is_unfilled(self.ad_text)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_predictions(path: str | Path) -> list[Prediction]:
    """The predictions of a file, checked."""
    rows: list[Prediction] = []
    seen: set[str] = set()
    for row in iter_jsonl(path):
        missing = [name for name in ("moment_id", "ad_text") if name not in row]
        if missing:
            raise ContractError(f"{path}: a prediction lacks {missing}")
        moment = str(row["moment_id"])
        if moment in seen:
            raise ContractError(f"{path}: moment {moment!r} is predicted twice; one row per moment")
        seen.add(moment)
        rows.append(Prediction(**{name: row.get(name) for name in FIELDS}))
    return rows


def write_predictions(path: str | Path, predictions: Iterable[Prediction]) -> int:
    from ..core.io import write_jsonl

    return write_jsonl(path, (prediction.as_dict() for prediction in predictions))


def assert_covers(predictions: Sequence[Prediction], wanted: Iterable[str]) -> None:
    """Raise unless the predictions are for exactly the moments asked for.

    Both directions: a moment not answered is missing evidence, and a moment
    answered that was not asked for means the set moved under the run.
    """
    asked = set(wanted)
    answered = {prediction.moment_id for prediction in predictions}
    missing = sorted(asked - answered)
    extra = sorted(answered - asked)
    if missing or extra:
        raise ContractError(
            f"the predictions do not match the moments asked for: "
            f"{len(missing)} missing (for example {missing[:3]}), "
            f"{len(extra)} unasked (for example {extra[:3]})"
        )


def merge_repairs(
    original: Sequence[Prediction], repaired: Mapping[str, Prediction]
) -> tuple[list[Prediction], dict[str, int]]:
    """Fold a second pass into a first, changing nothing else.

    A repair may only touch a moment the first pass left without a description. A
    repair of a moment that already had one is a mistake, not a merge, and stops here.

    The decision the describer made in the first pass is kept: a repair asks again
    for the description, so it knows nothing new about what the describer would have
    chosen.
    """
    by_moment = {prediction.moment_id: prediction for prediction in original}
    unknown = sorted(set(repaired) - set(by_moment))
    if unknown:
        raise ContractError(f"repairs for moments that were not predicted: {unknown[:3]}")
    already = sorted(
        moment for moment in repaired if by_moment[moment].filled
    )
    if already:
        raise ContractError(f"repairs for moments that already had a description: {already[:3]}")
    merged: list[Prediction] = []
    counts = {"kept": 0, "repaired": 0, "still_empty": 0}
    for prediction in original:
        repair = repaired.get(prediction.moment_id)
        if repair is None:
            merged.append(prediction)
            counts["kept"] += 1
        elif not repair.filled:
            merged.append(prediction)
            counts["still_empty"] += 1
        else:
            merged.append(
                Prediction(
                    output_id=prediction.output_id,
                    moment_id=prediction.moment_id,
                    emit=prediction.emit,
                    ad_text=repair.ad_text,
                    rung=repair.rung,
                    rationale=repair.rationale,
                    forced=True,
                    how=repair.how,
                )
            )
            counts["repaired"] += 1
    if {prediction.moment_id for prediction in merged} != set(by_moment):
        raise ContractError("the merge changed which moments are predicted")
    return merged, counts


def summarise(predictions: Sequence[Prediction]) -> dict[str, Any]:
    """How many moments were answered, and how the answers were read."""
    how: dict[str, int] = {}
    for prediction in predictions:
        how[prediction.how] = how.get(prediction.how, 0) + 1
    filled = [prediction for prediction in predictions if prediction.filled]
    return {
        "moments": len(predictions),
        "described": len(filled),
        "empty": len(predictions) - len(filled),
        "chose_to_describe": sum(1 for prediction in predictions if prediction.emit),
        "read_by": dict(sorted(how.items())),
    }
