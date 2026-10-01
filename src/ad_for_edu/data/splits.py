"""The train/test split, and the checks that keep the two sides apart.

The split is by lecture: every moment of a lecture is on the side of its
lecture. A lecture whose side is unknown is an error. It is never taken to be
training material, because that is how test material reaches a training set.

The checks raise. A check that returned a flag could be logged and passed over,
and these are boundaries that must stop a run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any

from ..core.errors import ContractError
from ..core.ids import lecture_of, moment_of
from ..core.registry import Registry
from .manifest import LectureManifest

TRAIN, TEST = "train", "test"
SIDES: tuple[str, str] = (TRAIN, TEST)


class SplitPolicy(ABC):
    """Says which side of the split a lecture is on."""

    @abstractmethod
    def side(self, lecture: str) -> str:
        """`train` or `test`. Raises for a lecture the policy does not know."""

    def is_test(self, lecture: str) -> bool:
        return self.side(lecture) == TEST

    def side_of_row(self, row: Mapping[str, Any]) -> str:
        return self.side(lecture_of_row(row))

    def select(self, rows: Iterable[Mapping[str, Any]], side: str) -> list[Mapping[str, Any]]:
        if side not in SIDES:
            raise ContractError(f"side must be one of {SIDES}, got {side!r}")
        return [row for row in rows if self.side_of_row(row) == side]


SPLIT_POLICIES: Registry[SplitPolicy] = Registry("split policy", SplitPolicy)


def lecture_of_row(row: Mapping[str, Any]) -> str:
    """The lecture of a row: its `lecture` field, or the lecture of its moment."""
    lecture = row.get("lecture") or lecture_of(moment_of(row))
    if not lecture:
        raise ContractError(f"cannot tell the lecture of a row with keys {sorted(row)[:8]}")
    return str(lecture)


@SPLIT_POLICIES.register("manifest")
class ManifestSplit(SplitPolicy):
    """The side stated for each lecture in the lecture manifest."""

    def __init__(self, manifest: LectureManifest) -> None:
        self.manifest = manifest

    def side(self, lecture: str) -> str:
        return self.manifest.get(lecture).split


@SPLIT_POLICIES.register("dataset_split")
class DatasetSplit(SplitPolicy):
    """The side each lecture has in the published training and test files."""

    def __init__(
        self,
        train_rows: Iterable[Mapping[str, Any]],
        test_rows: Iterable[Mapping[str, Any]],
    ) -> None:
        train = {lecture_of_row(row) for row in train_rows}
        test = {lecture_of_row(row) for row in test_rows}
        shared = sorted(train & test)
        if shared:
            raise ContractError(f"lectures on both sides of the published split: {shared}")
        if not train or not test:
            raise ContractError("a side of the published split is empty")
        self.sides = {**dict.fromkeys(train, TRAIN), **dict.fromkeys(test, TEST)}

    def side(self, lecture: str) -> str:
        try:
            return self.sides[lecture]
        except KeyError:
            raise ContractError(
                f"lecture {lecture!r} is in neither side of the published split"
            ) from None


def assert_no_test_lecture(rows: Iterable[Mapping[str, Any]], policy: SplitPolicy) -> None:
    """Raise when any of `rows`, which are meant for training, comes from a test lecture."""
    offending = sorted({lecture_of_row(row) for row in rows if policy.side_of_row(row) == TEST})
    if offending:
        raise ContractError(
            f"training material drawn from test lectures: {offending}. "
            "The training set would contain the evaluation data."
        )


def assert_disjoint_moments(
    first: Iterable[Mapping[str, Any]],
    second: Iterable[Mapping[str, Any]],
    *,
    first_name: str,
    second_name: str,
    same_lectures: bool = True,
) -> None:
    """Raise when the two sets of rows share a moment.

    An empty overlap proves something only when the two sets could have overlapped.
    Two sets that use different identifiers never share a moment, and that would
    read as "disjoint" without having tested anything. With `same_lectures`, the
    sets are therefore required to share at least one lecture; pass False for sets
    that are meant to come from different lectures.
    """
    first, second = list(first), list(second)
    left = {moment_of(row) for row in first} - {""}
    right = {moment_of(row) for row in second} - {""}
    if not left or not right:
        raise ContractError(
            f"cannot compare {first_name} and {second_name}: one of them has no moments"
        )
    shared = sorted(left & right)
    if shared:
        raise ContractError(
            f"{first_name} and {second_name} share {len(shared)} moments, "
            f"for example {shared[:3]}"
        )
    if same_lectures:
        lectures = {lecture_of(moment) for moment in left} & {lecture_of(m) for m in right}
        if not lectures:
            raise ContractError(
                f"{first_name} and {second_name} share no lecture, so their moments could "
                "not have overlapped. Check that both use the same identifiers."
            )


def assert_disjoint_lectures(
    first: Iterable[Mapping[str, Any]],
    second: Iterable[Mapping[str, Any]],
    *,
    first_name: str,
    second_name: str,
) -> None:
    """Raise when the two sets of rows share a lecture."""
    left = {lecture_of_row(row) for row in first}
    right = {lecture_of_row(row) for row in second}
    shared = sorted(left & right)
    if shared:
        raise ContractError(f"{first_name} and {second_name} share the lectures {shared}")
