"""Identifiers and how they nest.

    lecture            lec-a
    moment             lec-a#0017          a frame event, numbered within the lecture
                       lec-a#g007          a speech-gap event
    description        lec-a#0017::writer  one writer's row for that moment

A moment can carry several description rows, one per writer. Anything that counts
or joins must say which of the three units it works on; `moment_of` is the one
place a description id is reduced to its moment.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import ContractError

MOMENT_SEPARATOR = "#"
WRITER_SEPARATOR = "::"
GAP_PREFIX = "g"


def moment_id(lecture: str, index: int, *, from_gap: bool = False) -> str:
    """The id of the `index`-th event of `lecture`."""
    if not lecture or MOMENT_SEPARATOR in lecture or WRITER_SEPARATOR in lecture:
        raise ContractError(f"not a lecture id: {lecture!r}")
    if index < 0:
        raise ContractError(f"event index must not be negative, got {index}")
    number = f"{GAP_PREFIX}{index:03d}" if from_gap else f"{index:04d}"
    return f"{lecture}{MOMENT_SEPARATOR}{number}"


def description_id(moment: str, writer: str) -> str:
    return f"{moment}{WRITER_SEPARATOR}{writer}"


def moment_of(row: Mapping[str, Any] | str) -> str:
    """The moment a row belongs to.

    A description row is keyed by `output_id`; a prediction or a pair is keyed by
    `moment_id`. Both resolve to the same moment id.
    """
    if isinstance(row, str):
        return row.split(WRITER_SEPARATOR)[0]
    described = str(row.get("output_id") or "")
    if described:
        return described.split(WRITER_SEPARATOR)[0]
    return str(row.get("moment_id") or "")


def writer_of(row: Mapping[str, Any] | str) -> str | None:
    """The writer named in a description id, or None when the id names no writer."""
    described = row if isinstance(row, str) else str(row.get("output_id") or "")
    if WRITER_SEPARATOR not in described:
        return None
    return described.split(WRITER_SEPARATOR, 1)[1] or None


def lecture_of(moment: Mapping[str, Any] | str) -> str:
    """The lecture a moment belongs to: the part of its id before the separator."""
    identifier = moment if isinstance(moment, str) else moment_of(moment)
    return identifier.split(MOMENT_SEPARATOR)[0]


def is_gap_moment(moment: str) -> bool:
    _, _, number = moment.partition(MOMENT_SEPARATOR)
    return number.startswith(GAP_PREFIX)
