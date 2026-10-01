"""Getting a description out of a trained describer.

Every system is asked for a description of every moment of the evaluation set,
whether or not it would have chosen to describe that moment. A system that prefers
silence must still produce something there, or the systems are not comparable: each
would be scored on the moments it chose, and a system that describes twenty easy
moments would score above one that describes all of them.

The answer is asked for as an object. The start of that object is written for the
describer, up to the opening of the description itself, so that what it produces is
the description. Its own decision whether to describe is recorded separately, from
a pass in which nothing is written for it.

A reply that does not parse is recovered in three ways before it is given up on,
and each is recorded so that a table can say how its descriptions were obtained.
Every way demands a word character: a reply of punctuation is not a description,
and the strictest reader will produce one as readily as the most forgiving.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from ..core.registry import Registry
from ..data.schema import is_unfilled
from ..llm.replies import extract_json

#: What is written for the describer, so that what follows is the description.
PREFILL = '{"emit": true, "ad_text": "'
_BACKSLASH = chr(92)
# A quotation mark that no backslash escapes: the pattern needs two characters to
# mean one literal backslash.
_UNESCAPED_QUOTE = re.compile(r"(?<!" + _BACKSLASH * 2 + r')"')
_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


@dataclass(frozen=True)
class Decoded:
    """What a describer produced for one moment."""

    ad_text: str | None
    emit: bool | None = None
    rung: int | None = None
    rationale: str | None = None
    #: How the reply was read: which way recovered it.
    how: str = "strict"

    @property
    def filled(self) -> bool:
        return not is_unfilled(self.ad_text)


class DecodingAttempt(ABC):
    """One way of asking a describer for a description."""

    name: ClassVar[str]

    @abstractmethod
    def generation_settings(self, maximum_tokens: int, seed: int) -> dict[str, Any]: ...


DECODING_ATTEMPTS: Registry[DecodingAttempt] = Registry("decoding attempt", DecodingAttempt)


@DECODING_ATTEMPTS.register("greedy")
class Greedy(DecodingAttempt):
    """The most likely continuation. The same answer every time."""

    name = "greedy"

    def __init__(self, maximum_new_tokens: int = 160) -> None:
        self.maximum_new_tokens = maximum_new_tokens

    def generation_settings(self, maximum_tokens: int, seed: int) -> dict[str, Any]:
        return {
            "do_sample": False,
            "max_new_tokens": maximum_tokens or self.maximum_new_tokens,
        }


@DECODING_ATTEMPTS.register("sampled")
class Sampled(DecodingAttempt):
    """A sampled continuation, for a moment the greedy one did not answer.

    Each attempt draws from its own seed, so a second attempt is a different
    continuation rather than the same one again, and the run stays reproducible.
    """

    name = "sampled"

    def __init__(
        self,
        temperature: float = 0.7,
        top_p: float = 0.9,
        maximum_new_tokens: int = 256,
        attempt: int = 1,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.maximum_new_tokens = maximum_new_tokens
        self.attempt = attempt

    def generation_settings(self, maximum_tokens: int, seed: int) -> dict[str, Any]:
        return {
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": maximum_tokens or self.maximum_new_tokens,
            "seed": seed * 1000 + self.attempt,
        }


class SalvageStep(ABC):
    """One way of reading a reply the strict reader could not."""

    name: ClassVar[str]

    @abstractmethod
    def recover(self, completion: str, prefill: str) -> Decoded | None: ...


SALVAGE_STEPS: Registry[SalvageStep] = Registry("salvage step", SalvageStep)


def says_something(text: Any) -> bool:
    """Whether a recovered description carries a word.

    Not merely whether it is non-empty: a reply of quotes and braces parses to a
    string of quotes and braces, which is not a description.
    """
    return not is_unfilled(text) and bool(re.search(r"\w", str(text)))


def _from_object(parsed: Any, how: str) -> Decoded | None:
    if not isinstance(parsed, dict) or not says_something(parsed.get("ad_text")):
        return None
    rung = parsed.get("rung")
    rationale = parsed.get("rationale")
    return Decoded(
        ad_text=str(parsed["ad_text"]).strip(),
        emit=bool(parsed.get("emit", True)),
        rung=rung if isinstance(rung, int) else None,
        rationale=rationale if isinstance(rationale, str) else None,
        how=how,
    )


@SALVAGE_STEPS.register("strict_object")
class StrictObject(SalvageStep):
    """The reply is the object it was asked for."""

    name = "strict_object"

    def recover(self, completion: str, prefill: str) -> Decoded | None:
        body = _FENCE.sub("", (prefill + completion).strip())
        match = re.search(r"\{.*\}", body, re.S)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return _from_object(parsed, self.name)


@SALVAGE_STEPS.register("balanced_object")
class BalancedObject(SalvageStep):
    """An object wrapped in a preamble, a fence, or trailing remarks."""

    name = "balanced_object"

    def recover(self, completion: str, prefill: str) -> Decoded | None:
        return _from_object(extract_json(prefill + completion), self.name)


@SALVAGE_STEPS.register("open_string")
class OpenString(SalvageStep):
    """What was written for the describer opened the description, so what follows is it.

    Up to the first quotation mark that is not escaped. With no such mark the
    describer reached its token limit in the middle of the description, and the whole
    of what it wrote is the description, cut short.

    Only the description is recovered here. The rung and the reason are gone, and
    they stay gone: this recovers what was written, it does not invent what was not.
    """

    name = "open_string"

    def recover(self, completion: str, prefill: str) -> Decoded | None:
        if not prefill.endswith('"'):
            return None
        # Everything up to the quotation mark that closes the description. With no
        # such mark the whole completion is the description, cut short. A completion
        # that opens with one closed an empty description, which is not a description:
        # taking the rest of the reply there would hand back the broken object as one.
        candidate = _UNESCAPED_QUOTE.split(completion, maxsplit=1)[0]
        if not candidate.strip():
            return None
        try:
            candidate = json.loads(
                chr(34)
                + candidate.replace(chr(10), _BACKSLASH + "n").replace(
                    chr(9), _BACKSLASH + "t"
                )
                + chr(34)
            )
        except (ValueError, TypeError):
            candidate = candidate.replace(_BACKSLASH + '"', '"')
        candidate = candidate.strip()
        if not says_something(candidate):
            return None
        return Decoded(ad_text=candidate, emit=True, how=self.name)


def read_completion(
    completion: str,
    steps: Sequence[SalvageStep],
    *,
    prefill: str = PREFILL,
) -> Decoded:
    """The description a completion carries, by the first way that recovers one."""
    for step in steps:
        recovered = step.recover(completion, prefill)
        if recovered is not None:
            return recovered
    return Decoded(ad_text=None, how="unrecovered")
