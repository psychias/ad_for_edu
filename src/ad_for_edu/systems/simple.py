"""The systems that need no describer: the readout, and a writer of the references.

**The readout** answers every moment with the title of its slide. It writes nothing
of its own: it copies what is already on screen, which is the thing the standard
forbids most plainly. It is in the table as a floor, and what it scores says what a
metric rewards when nothing has been described.

The title is taken as the first line of the slide text, or the first clause of it,
so that the readout is a plausible-looking description rather than a page of text.

**A reference writer** is scored from the descriptions it already wrote. It answered
only the moments it chose to answer, so its number is over its own moments; a table
that prints it beside the systems must say so, since the systems answered all of them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from ..core.text import strip_sentinels
from ..inference.predictions import Prediction
from .base import DESCRIPTION_SYSTEMS, DescriptionSystem, EvalMoment

_CLAUSE = re.compile(r"[.;:]|\s-\s")


def slide_title(slide_text: str | None, *, most_words: int = 12) -> str | None:
    """The title of a slide: its first line, or the first clause of it.

    None when the slide carried no text, since there is then nothing to read out.
    """
    cleaned = strip_sentinels(slide_text)
    if not cleaned:
        return None
    first = cleaned.splitlines()[0].strip()
    first = _CLAUSE.split(first)[0].strip()
    words = first.split()
    if not words:
        return None
    return " ".join(words[:most_words])


@DESCRIPTION_SYSTEMS.register("slide_title")
class SlideTitleReadout(DescriptionSystem):
    """Answers every moment with the title of its slide."""

    forced = True

    def __init__(self, most_words: int = 12) -> None:
        self.most_words = most_words

    def describe(self, moments: Sequence[EvalMoment]) -> list[Prediction]:
        answers = []
        for moment in moments:
            title = slide_title(moment.slide_ocr, most_words=self.most_words) or slide_title(
                moment.what_on_screen, most_words=self.most_words
            )
            answers.append(
                Prediction(
                    output_id=moment.moment_id,
                    moment_id=moment.moment_id,
                    emit=True,
                    ad_text=title,
                    forced=True,
                    how="readout",
                )
            )
        return answers


@DESCRIPTION_SYSTEMS.register("reference_writer")
class ReferenceWriter(DescriptionSystem):
    """A writer of the references, scored from what it already wrote.

    It answered the moments it chose to answer, so its number is over those and not
    over the shared set.
    """

    forced = False
    on_shared_moments = False

    def __init__(self, writer: str, descriptions: Mapping[str, str]) -> None:
        self.writer = writer
        self.descriptions = dict(descriptions)

    def describe(self, moments: Sequence[EvalMoment]) -> list[Prediction]:
        answers = []
        for moment in moments:
            written = self.descriptions.get(moment.moment_id)
            answers.append(
                Prediction(
                    output_id=f"{moment.moment_id}::{self.writer}",
                    moment_id=moment.moment_id,
                    emit=written is not None,
                    ad_text=written,
                    forced=False,
                    how="written",
                )
            )
        return answers

    def own_moments(self, moments: Sequence[EvalMoment]) -> list[EvalMoment]:
        """The moments this writer chose to describe."""
        return [moment for moment in moments if moment.moment_id in self.descriptions]
