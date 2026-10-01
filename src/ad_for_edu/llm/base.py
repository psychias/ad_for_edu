"""The provider family: one way to call a model, whatever serves it.

A request names a model by its key in the model catalogue and carries the full
prompt text, with still images or clips where the stage needs them. A provider
turns the request into one call of its service and returns the reply together
with what the service reported about it: why generation stopped, how many
tokens were used, which model answered.

Those reported facts are part of the reply on purpose. A reply cut off by the
token limit looks like a formatting problem until `finish_reason` says `length`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from ..core.errors import AdForEduError
from ..core.registry import Registry

TEXT, IMAGE, VIDEO = "text", "image", "video"
MODALITIES: tuple[str, ...] = (TEXT, IMAGE, VIDEO)

#: The token limit was reached before the reply was complete.
FINISHED_BY_LENGTH = "length"


class ProviderError(AdForEduError, RuntimeError):
    """A call failed in a way that another attempt may clear."""


class TerminalProviderError(ProviderError):
    """A call failed in a way that no retry will clear: credit, quota or authentication.

    Retrying such a failure spends the retry budget of every remaining unit and
    lets a run end as if it had merely met transient errors. A stage stops on it.
    """

    def __init__(self, condition: str, detail: str) -> None:
        super().__init__(f"{condition}: {detail[:200]}")
        self.condition = condition


@dataclass(frozen=True)
class LLMRequest:
    """One call."""

    model: str
    prompt: str
    images: tuple[Path, ...] = ()
    videos: tuple[Path, ...] = ()
    temperature: float = 0.0
    max_tokens: int = 1200
    #: Bounds the reasoning of a reasoning model. It can be bounded, not switched off.
    reasoning_effort: str | None = None
    #: Handle of a server-side prompt cache, for providers that have one.
    cached_prefix: str | None = None
    #: Carried through to the reply, so a caller can match replies to units.
    tag: str = ""

    @property
    def modalities(self) -> frozenset[str]:
        used = {TEXT}
        if self.images:
            used.add(IMAGE)
        if self.videos:
            used.add(VIDEO)
        return frozenset(used)


@dataclass(frozen=True)
class LLMReply:
    """What came back, and what the service said about it."""

    text: str
    model: str
    tag: str = ""
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    #: The model the service reports as having answered.
    response_model: str | None = None
    request_id: str | None = None
    latency_ms: int | None = None
    details: dict[str, object] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return self.finish_reason == FINISHED_BY_LENGTH


class LLMProvider(ABC):
    """Calls the models of one service."""

    #: What this provider can carry in a request.
    modalities: ClassVar[frozenset[str]] = frozenset({TEXT})
    #: Whether a call costs money.
    paid: ClassVar[bool] = True

    @abstractmethod
    def generate(self, request: LLMRequest, model_id: str) -> LLMReply:
        """Make one call. `model_id` is the name of the model at this provider."""

    def generate_many(
        self, requests: Sequence[LLMRequest], model_id: str
    ) -> list[LLMReply | Exception]:
        """Make several calls, in order. A failed call yields its exception in place of a
        reply, so the caller can record an error for exactly the unit that failed."""
        replies: list[LLMReply | Exception] = []
        for request in requests:
            try:
                replies.append(self.generate(request, model_id))
            except TerminalProviderError:
                raise
            except Exception as error:  # noqa: BLE001 - handed back per unit
                replies.append(error.with_traceback(None))
        return replies


PROVIDERS: Registry[LLMProvider] = Registry("model provider", LLMProvider)

_TERMINAL = (
    ("OUT OF CREDIT", ("Error code: 402", "Insufficient credits", "insufficient_quota")),
    ("AUTHENTICATION REJECTED", ("Error code: 401", "invalid_api_key", "Error code: 403")),
    (
        "QUOTA EXHAUSTED",
        ("Error code: 429", "RESOURCE_EXHAUSTED", "exceeded your current quota"),
    ),
)


def terminal_condition(message: str) -> str | None:
    """The name of the condition when `message` reports a failure no retry will clear.

    Matching is on the error forms the client libraries emit, never on a bare status
    number: identifiers of moments contain digits and appear inside error messages.
    """
    for condition, markers in _TERMINAL:
        if any(marker in message for marker in markers):
            return condition
    return None
