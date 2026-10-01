"""The material a training cell is given, and the checks on it.

Two checks run before anything is trained, and both stop the run.

**Nothing from a test lecture is trained on.** The split is by lecture, and a
moment of a test lecture inside the training material means a system is evaluated
on what it was trained on. Nothing downstream would show it: the two arms stay
aligned, training and development stay apart by their own definition, and the
result simply comes out too high.

**The development slice shares no moment with the training material.** For the
supervised arm the slice is by moment, since the examples of one moment are
several writers' answers to it. For the preference arm it is by lecture, because
pairs of one lecture share slides and phrasing.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..core.errors import ContractError
from ..core.ids import lecture_of, moment_of
from ..data.splits import SplitPolicy, assert_no_test_lecture


@dataclass(frozen=True)
class TrainingData:
    """What one cell trains and evaluates on."""

    train: Sequence[Mapping[str, Any]]
    development: Sequence[Mapping[str, Any]]

    def __post_init__(self) -> None:
        if not self.train:
            raise ContractError("there is nothing to train on")

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "development": len(self.development)}


def split_by_moment(
    rows: Sequence[Mapping[str, Any]], share: float, seed: int
) -> TrainingData:
    """Hold out a share of the moments, with every row of a held-out moment.

    By moment, because the rows of one moment are several writers' answers to the
    same moment: holding out a row would leave the others in, and the answer to the
    held-out one is nearly in the training material.
    """
    if not 0 < share < 1:
        raise ContractError(f"the share held out must be between 0 and 1, got {share}")
    moments = sorted({moment_of(row) for row in rows})
    if len(moments) < 2:
        raise ContractError("a split by moment needs at least two moments")
    generator = random.Random(seed)
    generator.shuffle(moments)
    held = set(moments[: max(1, round(share * len(moments)))])
    train = [row for row in rows if moment_of(row) not in held]
    development = [row for row in rows if moment_of(row) in held]
    if not train or not development:
        raise ContractError("the split left one side empty; adjust the share")
    return TrainingData(train, development)


def assert_no_shared_moment(data: TrainingData) -> None:
    """Raise when a moment is on both sides."""
    shared = {moment_of(row) for row in data.train} & {
        moment_of(row) for row in data.development
    }
    if shared:
        raise ContractError(
            f"{len(shared)} moments are in both the training and the development material, "
            f"for example {sorted(shared)[:3]}"
        )


def assert_no_shared_lecture(data: TrainingData) -> None:
    """Raise when a lecture is on both sides."""
    shared = {lecture_of(moment_of(row)) for row in data.train} & {
        lecture_of(moment_of(row)) for row in data.development
    }
    if shared:
        raise ContractError(f"lectures on both sides: {sorted(shared)[:3]}")


def prepare(
    rows: Sequence[Mapping[str, Any]],
    policy: SplitPolicy,
    *,
    share: float,
    seed: int,
    by: str = "moment",
) -> TrainingData:
    """The training material, checked.

    `by` says what the development slice is taken by: moments for the supervised
    arm, lectures for the preference arm.
    """
    assert_no_test_lecture(rows, policy)
    if by == "moment":
        data = split_by_moment(rows, share, seed)
        assert_no_shared_moment(data)
        return data
    if by == "lecture":
        from ..pairs.pool import hold_out_by_lecture

        train, development = hold_out_by_lecture(
            rows, share, seed, lecture_of=lambda row: lecture_of(moment_of(row))
        )
        data = TrainingData(train, development)
        assert_no_shared_lecture(data)
        return data
    raise ContractError(f"a development slice is taken by moment or by lecture, not by {by!r}")


def as_preference_rows(
    pairs: Sequence[Mapping[str, Any]],
    prompt_of: Callable[[Mapping[str, Any]], Any],
    chosen_of: Callable[[Mapping[str, Any]], str],
) -> list[dict[str, Any]]:
    """Pairs as the preference trainer reads them: a prompt, a preferred answer, the other.

    A pair with no direction is left out rather than given one: there is no side to
    prefer, and choosing one would invent a preference.
    """
    rows = []
    for pair in pairs:
        side = chosen_of(pair)
        if side not in ("a", "b"):
            continue
        other = "b" if side == "a" else "a"
        rows.append(
            {
                "pair_id": pair.get("pair_id"),
                "moment_id": pair.get("moment_id"),
                "prompt": prompt_of(pair),
                "chosen": pair[side],
                "rejected": pair[other],
            }
        )
    return rows
