"""The lecture manifest: which recordings exist, and what is known about each.

The manifest is a table with one row per lecture:

    lecture_id      the identifier used in every moment id
    video           the recording, relative to the video directory
    course          the course the lecture belongs to
    split           train or test
    renders_cursor  whether the recording shows a pointer; empty when not probed

It is data, supplied with the recordings. Nothing about a particular lecture is
written in code: not its identifier, not its course, not its side of the split.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

from ..core.errors import ContractError
from ..core.io import read_csv
from .schema import LectureEntry

COLUMNS: tuple[str, ...] = ("lecture_id", "video", "course", "split", "renders_cursor")
_TRUE, _FALSE = {"true", "1", "yes"}, {"false", "0", "no"}


def _flag(value: object, *, where: str) -> bool | None:
    """A yes/no cell, or None when the cell is empty."""
    if isinstance(value, bool) or value is None:
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ContractError(f"{where}: renders_cursor must be true, false or empty, got {value!r}")


class LectureManifest:
    """The lectures, by identifier."""

    def __init__(self, entries: Iterable[LectureEntry]) -> None:
        self.entries: dict[str, LectureEntry] = {}
        for entry in entries:
            if entry.lecture_id in self.entries:
                raise ContractError(f"lecture {entry.lecture_id!r} is listed twice")
            self.entries[entry.lecture_id] = entry
        if not self.entries:
            raise ContractError("the lecture manifest is empty")

    @classmethod
    def load(cls, path: str | Path) -> LectureManifest:
        rows = read_csv(path)
        if rows:
            missing = [name for name in COLUMNS[:4] if name not in rows[0]]
            if missing:
                raise ContractError(f"{path}: the manifest lacks the columns {missing}")
        return cls(cls._entry(row, where=f"{path}") for row in rows)

    @staticmethod
    def _entry(row: Mapping[str, object], *, where: str) -> LectureEntry:
        values = dict(row)
        lecture = str(values.get("lecture_id", "")).strip()
        values["lecture_id"] = lecture
        values["split"] = str(values.get("split", "")).strip().lower()
        values["renders_cursor"] = _flag(
            values.get("renders_cursor"), where=f"{where}: lecture {lecture!r}"
        )
        return LectureEntry.from_row(values)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[LectureEntry]:
        return iter(self.entries.values())

    def __contains__(self, lecture: object) -> bool:
        return lecture in self.entries

    def get(self, lecture: str) -> LectureEntry:
        try:
            return self.entries[lecture]
        except KeyError:
            raise ContractError(
                f"lecture {lecture!r} is not in the manifest. A lecture that is not listed "
                "has no side of the split and is never assumed to be training material."
            ) from None

    def on_side(self, side: str) -> tuple[str, ...]:
        if side not in ("train", "test"):
            raise ContractError(f"side must be train or test, got {side!r}")
        return tuple(entry.lecture_id for entry in self if entry.split == side)

    def courses(self) -> dict[str, tuple[str, ...]]:
        """The lectures of each course."""
        by_course: dict[str, list[str]] = {}
        for entry in self:
            by_course.setdefault(entry.course, []).append(entry.lecture_id)
        return {course: tuple(lectures) for course, lectures in sorted(by_course.items())}

    def cursor_facts(self) -> dict[str, bool]:
        """Whether each probed lecture shows a pointer. Lectures not probed are left out."""
        return {
            entry.lecture_id: entry.renders_cursor
            for entry in self
            if entry.renders_cursor is not None
        }

    def check_courses_are_not_split(self) -> list[str]:
        """Courses with lectures on both sides. Holding out whole courses leaves none."""
        mixed = []
        for course, lectures in self.courses().items():
            sides = {self.entries[lecture].split for lecture in lectures}
            if len(sides) > 1:
                mixed.append(course)
        return mixed
