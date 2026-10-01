"""A trained describer as a system, and the same describer before training.

Both are asked for a description of every moment, in the same way, so that the
difference between them is training and nothing else. The prompt, the pictures and
what is written for the describer are the same in both.

A moment the first way of asking does not answer is asked again, sampled, up to a
stated number of times. The ways are tried in order and the first that yields a
description wins; which one that was is recorded on the prediction, so a table can
say how its descriptions were obtained.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..inference.decoding import (
    PREFILL,
    DecodingAttempt,
    SalvageStep,
    read_completion,
)
from ..inference.predictions import Prediction
from ..training.examples import InputArm, system_prompt
from .base import DESCRIPTION_SYSTEMS, DescriptionSystem, EvalMoment

#: What the describer is asked, as a conversation ready for a chat template.
Generate = Callable[[list[dict[str, Any]], Sequence[Any], dict[str, Any]], str]


def conversation(moment: EvalMoment, arm: InputArm) -> list[dict[str, Any]]:
    """What the describer is shown: the same messages it was trained on, without the answer."""
    from ..data.schema import ReferenceRow

    row = ReferenceRow(
        output_id=moment.moment_id,
        type=moment.type,
        emit=False,
        transcript_window=moment.transcript_window,
        slide_ocr=moment.slide_ocr,
        what_on_screen=moment.what_on_screen,
        pause_after=moment.pause_after,
    )
    references = [str(path) for path in moment.keyframes] if arm.carries_images else []
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": arm.user_content(row, references)},
    ]


@DESCRIPTION_SYSTEMS.register("adapter")
class TrainedDescriber(DescriptionSystem):
    """A describer that was trained, asked for a description of every moment."""

    forced = True

    def __init__(
        self,
        generate: Generate,
        arm: InputArm,
        attempts: Sequence[DecodingAttempt],
        salvage: Sequence[SalvageStep],
        *,
        seed: int = 0,
        maximum_tokens: int = 0,
        prefill: str = PREFILL,
        name: str = "",
    ) -> None:
        if not attempts:
            raise ValueError("a describer needs at least one way of being asked")
        self.generate = generate
        self.arm = arm
        self.attempts = tuple(attempts)
        self.salvage = tuple(salvage)
        self.seed = seed
        self.maximum_tokens = maximum_tokens
        self.prefill = prefill
        self.name = name

    def describe(self, moments: Sequence[EvalMoment]) -> list[Prediction]:
        answers = []
        for moment in moments:
            messages = conversation(moment, self.arm)
            decoded = None
            for attempt in self.attempts:
                settings = attempt.generation_settings(self.maximum_tokens, self.seed)
                completion = self.generate(messages, moment.keyframes, settings)
                decoded = read_completion(completion, self.salvage, prefill=self.prefill)
                if decoded.filled:
                    decoded = type(decoded)(
                        ad_text=decoded.ad_text,
                        emit=decoded.emit,
                        rung=decoded.rung,
                        rationale=decoded.rationale,
                        how=f"{attempt.name}/{decoded.how}",
                    )
                    break
            answers.append(
                Prediction(
                    output_id=f"{moment.moment_id}::{self.name}" if self.name else moment.moment_id,
                    moment_id=moment.moment_id,
                    emit=decoded.emit if decoded else None,
                    ad_text=decoded.ad_text if decoded else None,
                    rung=decoded.rung if decoded else None,
                    rationale=decoded.rationale if decoded else None,
                    forced=True,
                    how=decoded.how if decoded else "unrecovered",
                )
            )
        return answers


@DESCRIPTION_SYSTEMS.register("zero_shot")
class UntrainedDescriber(TrainedDescriber):
    """The same describer before it was trained, asked in exactly the same way."""
