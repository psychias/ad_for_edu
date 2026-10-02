"""Reading someone else's rows: a map from this package's field names to theirs.

The evaluation does not need the corpus this package was built around. It needs, per
moment, an identifier, the lecture the moment belongs to, a time, a type, and the text
that was on screen and spoken nearby; and per description, an identifier and the text.
Any dataset that can supply those can be scored, whatever it calls its columns.

`FieldMap` is that translation, and it has one job beyond renaming: it makes the
**lecture explicit**.

That matters more than it sounds. The novelty term groups a system's descriptions by
lecture, and it takes the lecture from the moment id by splitting on `#`. An id with no
`#` splits to itself, so every moment becomes a lecture of its own, nothing can be seen
to repeat, and the factor is 1 everywhere. The score stays plausible and the term has
stopped working. That is not a hypothetical: the published reference rows carry a second
identifier in the older namespace, and grouping by it turns 65 lectures into 1,574 and
the mean factor from 0.9933 to exactly 1.

So a dataset says which column holds the lecture. If it does not, the lecture is parsed
from the id and `assert_lectures_group` checks that the parse actually grouped anything,
refusing rather than scoring with the term inert.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import ContractError, SettingsError
from ..core.ids import MOMENT_SEPARATOR, moment_of
from ..core.settings import check_keys, load_yaml

#: What this package calls each thing, and the column it reads by default.
DEFAULTS: Mapping[str, str] = {
    "moment_id": "moment_id",
    "lecture": "lecture",
    "time": "t",
    "type": "type",
    "slide_text": "slide_ocr",
    "on_screen": "what_on_screen",
    "transcript_window": "transcript_window",
    "pause_after": "pause_after",
    "text": "ad_text",
    "emit": "emit",
}
#: The fields a moment cannot be scored without.
REQUIRED = ("moment_id",)


@dataclass(frozen=True)
class FieldMap:
    """Which column of a dataset holds each thing this package reads."""

    columns: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULTS))

    def __post_init__(self) -> None:
        unknown = sorted(set(self.columns) - set(DEFAULTS))
        if unknown:
            raise SettingsError(
                f"a field map names {unknown}, which this package does not read. "
                f"Known: {sorted(DEFAULTS)}"
            )

    @classmethod
    def load(cls, path: str | Path) -> FieldMap:
        raw = load_yaml(Path(path))
        check_keys(raw, tuple(DEFAULTS), where=Path(path).name)
        return cls({**DEFAULTS, **{k: str(v) for k, v in raw.items() if v}})

    @classmethod
    def parse(cls, pairs: Sequence[str] | None) -> FieldMap:
        """A map from `name=column` arguments. No argument means the defaults."""
        columns = dict(DEFAULTS)
        for pair in pairs or ():
            name, _, column = str(pair).partition("=")
            name, column = name.strip(), column.strip()
            if not column:
                raise SettingsError(
                    f"{pair!r} is not name=column; for example moment_id=id"
                )
            if name not in DEFAULTS:
                raise SettingsError(
                    f"this package does not read a field called {name!r}; "
                    f"known: {sorted(DEFAULTS)}"
                )
            columns[name] = column
        return cls(columns)

    @property
    def renamed(self) -> dict[str, str]:
        """Only the fields read from a column other than the default one."""
        return {
            name: column
            for name, column in self.columns.items()
            if column != DEFAULTS[name]
        }

    def column(self, name: str) -> str:
        try:
            return self.columns[name]
        except KeyError:
            raise ContractError(f"no field called {name!r}") from None

    def get(self, row: Mapping[str, Any], name: str, default: Any = None) -> Any:
        return row.get(self.column(name), default)

    # ------------------------------------------------------------ whole rows
    def moment(self, row: Mapping[str, Any]) -> str:
        """The moment a row is about, by the mapped column or by this package's own ids."""
        found = self.get(row, "moment_id")
        if found:
            return str(found)
        # Fall back to the package's own conventions, which also read `output_id`.
        found = moment_of(row)
        if found:
            return found
        raise ContractError(
            f"a row carries no {self.column('moment_id')!r}, so it cannot be joined to "
            "anything. Name the column with the moment identifier."
        )

    def lecture(self, row: Mapping[str, Any]) -> str:
        """The lecture a row belongs to: the named column, else parsed from the id."""
        found = self.get(row, "lecture")
        if found:
            return str(found)
        return self.moment(row).split(MOMENT_SEPARATOR)[0]

    def canonical(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """One row under this package's own field names, with the rest carried along."""
        moment = self.moment(row)
        out: dict[str, Any] = {
            "moment_id": moment,
            "lecture": self.lecture(row),
            "t": _number(self.get(row, "time")),
            "time": _number(self.get(row, "time")),
            "type": str(self.get(row, "type") or ""),
            "slide_ocr": str(self.get(row, "slide_text") or ""),
            "what_on_screen": str(self.get(row, "on_screen") or ""),
            "transcript_window": str(self.get(row, "transcript_window") or ""),
            "pause_after": _number(self.get(row, "pause_after")),
        }
        text = self.get(row, "text")
        if text is not None:
            out["ad_text"] = text
        emit = self.get(row, "emit")
        if emit is not None:
            out["emit"] = emit
        # Anything the map does not name is kept, so a dataset's own columns survive
        # into the report that reads them.
        mapped = set(self.columns.values())
        for key, value in row.items():
            if key not in mapped and key not in out:
                out[key] = value
        return out

    def rows(self, rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [self.canonical(row) for row in rows]

    def as_dict(self) -> dict[str, Any]:
        return {"columns": dict(self.columns), "renamed": self.renamed}


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def assert_lectures_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    fields: FieldMap | None = None,
    what: str = "these moments",
) -> dict[str, Any]:
    """Check that the lecture actually groups the moments, and say how.

    Takes rows already put through `FieldMap.canonical`, so it reads this package's own
    `lecture` and `moment_id`. `fields` is used only to name the column a reader should
    set, in the message.

    The novelty term is only meaningful if several descriptions share a lecture. When
    every moment is its own lecture the term can never fire, and a score computed that
    way is not this metric even though nothing errored. So this refuses.
    """
    if not rows:
        raise ContractError(f"{what}: there are no rows to group")
    lectures = {str(row.get("lecture") or "") for row in rows}
    moments = {str(row.get("moment_id") or "") for row in rows}
    if len(moments) > 1 and len(lectures) == len(moments):
        column = (fields or FieldMap()).column("lecture")
        raise ContractError(
            f"{what}: every one of {len(moments)} moments is its own lecture, so the "
            "novelty term could never fire and the score would not be this metric. "
            f"Set the field map's 'lecture' to the column that groups them; it is "
            f"reading {column!r}."
        )
    return {
        "moments": len(moments),
        "lectures": len(lectures),
        "moments_per_lecture": round(len(moments) / len(lectures), 2),
    }
