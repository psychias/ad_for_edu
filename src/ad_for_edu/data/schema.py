"""The rows the stages exchange.

Each class names the fields a stage relies on. Rows read from a file may carry
further fields; `from_row` keeps them in `extra` so that a later stage can write
them back unchanged. Field names match the published dataset files.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar

from ..core.errors import ContractError
from ..core.ids import description_id, lecture_of, moment_of, writer_of
from ..core.text import slide_premise

R = TypeVar("R", bound="Row")

#: The closed set of moment types. A classifier reply outside this set is rejected.
MOMENT_TYPES: tuple[str, ...] = (
    "slide",
    "transition",
    "reveal",
    "pointing",
    "ink",
    "figure",
    "chart",
    "table",
    "math",
    "code",
    "animation",
    "other",
)

#: What a classifier answers when a candidate event is not a moment to describe.
REJECT = "reject"

#: The six rule categories, in reporting order.
CATEGORIES: tuple[str, ...] = (
    "style",
    "terminology",
    "length",
    "deixis",
    "faithfulness",
    "non_redundancy",
)

#: How a preference pair was obtained.
PAIR_SOURCES: tuple[str, ...] = ("natural", "controlled")

#: The strata of the rated pool.
STRATA: tuple[str, ...] = ("controlled", "natural_decided", "natural_flip")

SIDES: tuple[str, str] = ("a", "b")
TIE = "tie"


@dataclass
class Row:
    """Base of every row class: construction from, and conversion to, a plain mapping."""

    required: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def from_row(cls: type[R], row: Mapping[str, Any]) -> R:
        missing = [name for name in cls.required if name not in row]
        if missing:
            raise ContractError(f"{cls.__name__} row lacks {missing}; has {sorted(row)[:12]}")
        names = {f.name for f in dataclasses.fields(cls)} - {"extra"}
        known = {name: row[name] for name in names if name in row}
        extra = {name: value for name, value in row.items() if name not in names}
        return cls(**known, extra=extra)  # type: ignore[call-arg]

    def to_row(self) -> dict[str, Any]:
        out = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name != "extra"
        }
        out.update(getattr(self, "extra", {}) or {})
        return out


@dataclass(frozen=True)
class MomentContext:
    """What is known about a moment when a description of it is written or scored.

    `renders_cursor` says whether the recording of this lecture shows a pointer at
    all. None means the fact was not supplied, which is different from False.
    """

    moment_id: str = ""
    type: str | None = None
    pause_after: float | None = None
    slide_ocr: str | None = None
    what_on_screen: str | None = None
    transcript_window: str | None = None
    renders_cursor: bool | None = None

    #: The row fields a context is read from. Closed, so a renamed field is noticed.
    FIELDS: ClassVar[tuple[str, ...]] = (
        "type",
        "pause_after",
        "slide_ocr",
        "what_on_screen",
        "transcript_window",
    )

    @classmethod
    def from_row(
        cls,
        row: Mapping[str, Any],
        *,
        moment_id: str = "",
        renders_cursor: bool | None = None,
    ) -> MomentContext:
        return cls(
            moment_id=moment_id or moment_of(row),
            renders_cursor=renders_cursor,
            **{name: row.get(name) for name in cls.FIELDS},
        )

    @property
    def premise(self) -> str:
        """Slide text plus on-screen summary, with no-content markers removed."""
        return slide_premise({"slide_ocr": self.slide_ocr, "what_on_screen": self.what_on_screen})

    @property
    def pause(self) -> float:
        return float(self.pause_after or 0.0)


@dataclass
class LectureEntry(Row):
    """One line of the lecture manifest."""

    required: ClassVar[tuple[str, ...]] = ("lecture_id", "video", "course", "split")

    lecture_id: str = ""
    video: str = ""
    course: str = ""
    split: str = ""
    renders_cursor: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.split not in ("train", "test"):
            raise ContractError(
                f"lecture {self.lecture_id!r}: split must be train or test, got {self.split!r}"
            )


@dataclass
class CandidateEvent(Row):
    """A visual event proposed by the detector, before classification."""

    required: ClassVar[tuple[str, ...]] = ("id", "lecture", "t", "channel")

    id: str = ""
    lecture: str = ""
    t: float = 0.0
    channel: str = ""
    visual: bool = True
    gap_after: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Moment(Row):
    """A classified event: a moment to describe, with its type."""

    required: ClassVar[tuple[str, ...]] = ("id", "lecture", "t", "type")

    id: str = ""
    lecture: str = ""
    t: float = 0.0
    type: str = ""
    channel: str = ""
    revisit: bool = False
    what_on_screen: str = ""
    model: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in MOMENT_TYPES:
            raise ContractError(
                f"moment {self.id!r}: type {self.type!r} is outside the closed set {MOMENT_TYPES}"
            )


@dataclass
class ReferenceRow(Row):
    """One writer's decision for one moment: a description, or silence, with its context."""

    required: ClassVar[tuple[str, ...]] = ("output_id", "type", "emit")

    output_id: str = ""
    time: Any = None
    type: str = ""
    emit: bool = False
    ad_text: str | None = None
    rung: int | None = None
    rationale: str | None = None
    family: str = ""
    transcript_window: str = ""
    slide_ocr: str = ""
    what_on_screen: str = ""
    pause_after: float = 0.0
    emit_window: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def moment_id(self) -> str:
        return moment_of(self.output_id)

    @property
    def lecture(self) -> str:
        return lecture_of(self.moment_id)

    @property
    def writer(self) -> str:
        return self.family or writer_of(self.output_id) or ""

    @property
    def describes(self) -> bool:
        """True when the writer chose to describe and wrote something."""
        return bool(self.emit) and bool((self.ad_text or "").strip())

    @classmethod
    def build(cls, moment_id: str, writer: str, **fields: Any) -> ReferenceRow:
        return cls(output_id=description_id(moment_id, writer), family=writer, **fields)


@dataclass
class PreferencePair(Row):
    """Two candidate descriptions of one moment."""

    required: ClassVar[tuple[str, ...]] = ("pair_id", "moment_id", "a", "b", "source")

    pair_id: str = ""
    moment_id: str = ""
    lecture: str = ""
    type: str = ""
    t: float = 0.0
    source: str = ""
    a: str = ""
    b: str = ""
    a_family: str = ""
    b_family: str = ""
    axis: str | None = None
    directive: str | None = None
    band: str | None = None
    known_direction: bool = False
    label_source: str = ""
    judge_winner: str | None = None
    stratum: str | None = None
    train_eligible: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.axis is not None and self.axis not in CATEGORIES:
            raise ContractError(
                f"pair {self.pair_id!r}: axis {self.axis!r} is outside {CATEGORIES}"
            )

    @property
    def controlled(self) -> bool:
        return self.axis is not None


@dataclass
class Verdict(Row):
    """A judge's answer for one pair in one presentation order."""

    required: ClassVar[tuple[str, ...]] = ("pair_id", "order")

    pair_id: str = ""
    moment_id: str = ""
    order: int = 0
    winner: str | None = None
    reason: str = ""
    model: str = ""
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return bool(self.error)


@dataclass
class RaterLabel(Row):
    """One rater's choice on one pair of the rated pool."""

    required: ClassVar[tuple[str, ...]] = ("pair_id", "rater_id", "side")

    pair_id: str = ""
    moment_id: str = ""
    rater_id: str = ""
    side: str = ""
    stratum: str = ""
    dimension: str | None = None
    compliant: bool | None = None
    reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.side not in (*SIDES, TIE):
            raise ContractError(
                f"label on {self.pair_id!r}: side must be a, b or tie, got {self.side!r}"
            )

    @property
    def decided(self) -> bool:
        return self.side in SIDES


@dataclass
class Prediction(Row):
    """A system's output for one moment. One row per moment, never per reference."""

    required: ClassVar[tuple[str, ...]] = ("output_id", "moment_id", "emit", "ad_text")

    output_id: str = ""
    moment_id: str = ""
    emit: bool | None = None
    ad_text: str | None = None
    rung: int | None = None
    rationale: str | None = None
    forced: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def filled(self) -> bool:
        return not is_unfilled(self.ad_text)


def is_unfilled(text: Any) -> bool:
    """True when a stored description carries nothing: None, blank, or the word `None`."""
    return text is None or not str(text).strip() or str(text).strip() == "None"
