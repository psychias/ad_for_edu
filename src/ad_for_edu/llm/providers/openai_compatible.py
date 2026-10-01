"""Models behind an OpenAI-compatible chat endpoint.

Images and clips travel inside the request as base64 data, never as links: the
material is lecture video, and a link would have to be reachable from outside.

Three kinds of failure are told apart. A terminal one (credit, quota,
authentication) stops the stage. A parameter the model does not accept is
dropped and the call repeated. A transient one is retried after a wait.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ...core.secrets import credential
from ..base import (
    IMAGE,
    PROVIDERS,
    TEXT,
    VIDEO,
    LLMProvider,
    LLMReply,
    LLMRequest,
    ProviderError,
    TerminalProviderError,
    terminal_condition,
)

TRANSIENT_MARKERS: tuple[str, ...] = (
    "rate limit",
    "overloaded",
    "timeout",
    "503",
    "500",
    "connection",
    "ssl",
    "certificate",
)
LIMIT_FIELDS: tuple[str, ...] = ("max_tokens", "max_completion_tokens")


def data_url(path: Path, mime_type: str) -> str:
    encoded = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:{mime_type};base64,{encoded}"


def message_content(request: LLMRequest) -> list[dict[str, Any]]:
    """The user message: the prompt, then the images, then the clips."""
    content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]
    for image in request.images:
        content.append({"type": "image_url", "image_url": {"url": data_url(image, "image/jpeg")}})
    for video in request.videos:
        content.append({"type": "video_url", "video_url": {"url": data_url(video, "video/mp4")}})
    return content


@PROVIDERS.register("openai_compatible")
class OpenAICompatible(LLMProvider):
    """Chat completions over an OpenAI-compatible endpoint."""

    modalities = frozenset({TEXT, IMAGE, VIDEO})
    paid = True

    def __init__(
        self,
        base_url: str,
        credential_variable: str,
        timeout_seconds: float = 120.0,
        retry_waits: Sequence[float] = (2, 8, 20),
        limit_field: str = "max_tokens",
        workers: int = 1,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        if limit_field not in LIMIT_FIELDS:
            raise ValueError(f"limit_field must be one of {LIMIT_FIELDS}, got {limit_field!r}")
        self.base_url = base_url
        self.credential_variable = credential_variable
        self.timeout_seconds = timeout_seconds
        self.retry_waits = tuple(retry_waits)
        self.limit_field = limit_field
        self.workers = max(1, int(workers))
        self._client_factory = client_factory
        self._client: Any = None
        self._sleep = time.sleep

    def _connect(self) -> Any:
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                from openai import OpenAI

                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key=credential(self.credential_variable),
                    timeout=self.timeout_seconds,
                )
        return self._client

    def _arguments(self, request: LLMRequest, model_id: str) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": message_content(request)}],
            "temperature": request.temperature,
            self.limit_field: request.max_tokens,
        }
        if request.reasoning_effort:
            arguments["extra_body"] = {"reasoning": {"effort": request.reasoning_effort}}
        return arguments

    def generate(self, request: LLMRequest, model_id: str) -> LLMReply:
        if request.cached_prefix:
            raise ProviderError(
                "this provider has no server-side prompt cache; the prefix is sent with every call"
            )
        client = self._connect()
        arguments = self._arguments(request, model_id)
        attempts = len(self.retry_waits) + 1
        last: Exception | None = None
        for attempt in range(attempts):
            started = time.time()
            try:
                response = client.chat.completions.create(**arguments)
            except Exception as error:  # noqa: BLE001 - classified below
                last = error
                message = str(error)
                condition = terminal_condition(message)
                if condition:
                    raise TerminalProviderError(condition, message) from None
                lowered = message.lower()
                if "temperature" in lowered and "temperature" in arguments:
                    del arguments["temperature"]
                    continue
                unsupported = "reasoning" in lowered or "unsupported" in lowered
                if unsupported and "extra_body" in arguments:
                    del arguments["extra_body"]
                    continue
                if any(marker in lowered for marker in TRANSIENT_MARKERS):
                    if attempt < attempts - 1:
                        self._sleep(self.retry_waits[min(attempt, len(self.retry_waits) - 1)])
                    continue
                raise
            return self._reply(request, response, int((time.time() - started) * 1000))
        raise ProviderError(f"no reply after {attempts} attempts: {str(last)[:200]}")

    def generate_many(
        self, requests: Sequence[LLMRequest], model_id: str
    ) -> list[LLMReply | Exception]:
        if self.workers <= 1 or len(requests) <= 1:
            return super().generate_many(requests, model_id)

        def one(request: LLMRequest) -> LLMReply | Exception:
            try:
                return self.generate(request, model_id)
            except Exception as error:  # noqa: BLE001 - handed back per unit
                return error.with_traceback(None)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(one, requests))

    @staticmethod
    def _reply(request: LLMRequest, response: Any, latency_ms: int) -> LLMReply:
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        return LLMReply(
            text=(choice.message.content or "").strip(),
            model=request.model,
            tag=request.tag,
            finish_reason=getattr(choice, "finish_reason", None),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            reasoning_tokens=int(getattr(details, "reasoning_tokens", 0) or 0),
            response_model=getattr(response, "model", None),
            request_id=getattr(response, "id", None),
            latency_ms=latency_ms,
        )
