"""Does this recording draw a pointer at all?

The question matters because a description cannot name what a lecturer points at
if the recording shows no pointing. A model asked about such a moment may still
answer in detail, so the fact is measured from the images and the label is
withheld where the recording cannot carry it.

The probe is deliberately simple: greyscale, a grid of cells, the absolute
difference between one frame and the next, and a count of the cells that changed
by more than a threshold. It has no learned part, so it cannot itself invent a
pointer, and it can be described in a sentence.

The count, not its size, gives the verdict.

    still     no cell changed. Nothing happened here.
    pointer   a few cells. A pointer is small and drags a short trail.
    content   many cells. A new slide, a build, a large highlight.

Two things must be left out of the count or every verdict is `content`. The area
where the time is burned in changes in every frame by design. And a recording
that composites a camera image, a clock or a progress bar beside the slide has a
part that changes always; a cell that is hot in most frame pairs of a whole
lecture is such a part, since an event is by definition occasional.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CELL = 16
DELTA_THRESHOLD = 25.0
#: A cell hot in at least this share of a lecture's frame pairs is part of the layout.
LAYOUT_SHARE = 0.60
#: At most this many hot cells is a pointer rather than a change of content.
POINTER_MAX_CELLS = 6
#: Below this share of sampled points showing a pointer, the recording draws none.
POINTER_MIN_RATE = 0.05
#: Fewer sampled points than this cannot settle the question, and the label is kept.
POINTER_MIN_SAMPLES = 20

STILL, POINTER, CONTENT = "still", "pointer", "content"


@dataclass(frozen=True)
class MotionProbe:
    """What changed between the frames of one candidate."""

    hot_cells: int
    peak_delta: float
    frames: int
    cells: int
    at: tuple[int, int] | None = None
    per_pair: tuple[int, ...] = field(default_factory=tuple)

    def verdict(self, pointer_max: int = POINTER_MAX_CELLS) -> str:
        if self.hot_cells == 0:
            return STILL
        return POINTER if self.hot_cells <= pointer_max else CONTENT


def to_grey(image: Any):
    """An image as a grid of grey values, in a type wide enough to subtract."""
    import numpy

    return numpy.asarray(image.convert("L"), dtype=numpy.int16)


def load_grey(path: Path):
    from PIL import Image

    with Image.open(path) as image:
        return to_grey(image)


def cell_deltas(first, second, cell: int = CELL):
    """The mean change per square of `cell` by `cell` pixels.

    A partial square at an edge is dropped rather than padded: padding would
    average real pixels against nothing and read as a change.
    """
    import numpy

    rows, columns = first.shape[0] // cell, first.shape[1] // cell
    difference = numpy.abs(
        first[: rows * cell, : columns * cell] - second[: rows * cell, : columns * cell]
    )
    return difference.reshape(rows, cell, columns, cell).mean(axis=(1, 3))


def layout_mask(
    deltas: Sequence[Any], threshold: float = DELTA_THRESHOLD, share: float = LAYOUT_SHARE
):
    """The cells that change in most frame pairs of one lecture.

    Learned per lecture, because what is always moving is a property of how that
    recording is composed. A mask from one layout applied to another would erase
    real events.
    """
    import numpy

    stacked = numpy.stack(list(deltas))
    return (stacked > threshold).mean(axis=0) >= share


def timestamp_mask(shape: tuple[int, int], box: tuple[int, int, int, int], cell: int = CELL):
    """The cells covering the burned-in time, which changes in every frame."""
    import numpy

    rows, columns = shape[0] // cell, shape[1] // cell
    mask = numpy.zeros((rows, columns), dtype=bool)
    left, top, right, bottom = box
    mask[top // cell : -(-bottom // cell), left // cell : -(-right // cell)] = True
    return mask


def probe_frames(
    frames: Sequence[Any],
    mask: Any = None,
    threshold: float = DELTA_THRESHOLD,
    cell: int = CELL,
) -> MotionProbe:
    """Probe the frames of one candidate, already read as grids of grey values.

    The frames must have been extracted without the burned-in time, or the area it
    covers must be in `mask`.
    """
    import numpy

    if len(frames) < 2:
        return MotionProbe(hot_cells=0, peak_delta=0.0, frames=len(frames), cells=0)
    best_hot, best_peak, best_at, per_pair, cells = 0, 0.0, None, [], 0
    for first, second in zip(frames, frames[1:], strict=False):
        delta = cell_deltas(first, second, cell)
        if mask is not None and mask.shape == delta.shape:
            delta = numpy.where(mask, 0.0, delta)
        cells = delta.size
        hot = int((delta > threshold).sum())
        per_pair.append(hot)
        best_peak = max(best_peak, float(delta.max()))
        if hot > best_hot:
            best_hot = hot
            row, column = numpy.unravel_index(delta.argmax(), delta.shape)
            best_at = (int(column * cell + cell // 2), int(row * cell + cell // 2))
    return MotionProbe(
        hot_cells=best_hot,
        peak_delta=best_peak,
        frames=len(frames),
        cells=cells,
        at=best_at,
        per_pair=tuple(per_pair),
    )


def renders_cursor(verdicts: Sequence[str], minimum_rate: float = POINTER_MIN_RATE) -> bool:
    """Whether the recording draws a pointer, from verdicts at sampled quiet points.

    The points must be sampled away from the candidate events. A candidate is
    chosen because something changed there, so candidates over-represent changes of
    slide and under-represent exactly the idle time in which an unmoving pointer
    shows up.

    With too few points the question is left open and the label is kept: withholding
    it on thin evidence would lose real material.
    """
    seen = list(verdicts)
    if len(seen) < POINTER_MIN_SAMPLES:
        return True
    return sum(1 for verdict in seen if verdict == POINTER) / len(seen) >= minimum_rate


def allowed_types(types: Sequence[str], draws_pointer: bool, pointing_type: str = "pointing"):
    """The labels a recording can support. The pointing label is offered only where
    a pointer is drawn: not offering it removes the opportunity to invent one."""
    return [name for name in types if draws_pointer or name != pointing_type]
