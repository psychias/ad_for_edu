"""Gemini, called through its own interface.

Two things set this provider apart. It accepts clips as well as images, which a
judge that must watch a moment needs. And it offers a prompt cache with a handle:
the shared prefix is stored once, each call names the handle, and the reply
reports how many tokens were served from the cache. That number is reported
with every reply, because a handle that is silently not applied looks exactly
like one that is.

Every call has a time limit. A long run of requests with images or clips can be
answered slower and slower until one never returns, and without a limit that
looks like a hang with nothing to catch.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ...core.secrets import GEMINI, credential
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

PartFactory = Callable[[bytes, str], Any]

TRANSIENT_MARKERS: tuple[str, ...] = (
    "503",
    "UNAVAILABLE",
    "429",
    "RESOURCE_EXHAUSTED",
    "500",
    "INTERNAL",
)


@PROVIDERS.register("gemini_native")
class GeminiNative(LLMProvider):
    """Gemini over its native interface, with bounded retry and an explicit prompt cache."""

    modalities = frozenset({TEXT, IMAGE, VIDEO})
    paid = True

    def __init__(
        self,
        credential_variable: str = GEMINI,
        timeout_seconds: float | None = 120.0,
        backoff_seconds: Sequence[float] = (30, 60, 120, 240, 480, 600, 600, 600),
        client_factory: Callable[[], Any] | None = None,
        part_factory: PartFactory | None = None,
    ) -> None:
        self.credential_variable = credential_variable
        self.timeout_seconds = timeout_seconds
        self.backoff_seconds = tuple(backoff_seconds)
        self._client_factory = client_factory
        self._part_factory = part_factory
        self._client: Any = None
        self._sleep = time.sleep

    def _connect(self) -> Any:
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                from google import genai
                from google.genai import types

                options = None
                if self.timeout_seconds:
                    options = types.HttpOptions(timeout=int(self.timeout_seconds * 1000))
                self._client = genai.Client(
                    api_key=credential(self.credential_variable), http_options=options
                )
        return self._client

    def _part(self, data: bytes, mime_type: str) -> Any:
        if self._part_factory is not None:
            return self._part_factory(data, mime_type)
        from google.genai import types

        return types.Part.from_bytes(data=data, mime_type=mime_type)

    def _parts(self, request: LLMRequest) -> list[Any]:
        """The images, then the clips, then the prompt."""
        parts: list[Any] = [
            self._part(Path(image).read_bytes(), "image/jpeg") for image in request.images
        ]
        parts += [self._part(Path(video).read_bytes(), "video/mp4") for video in request.videos]
        parts.append(request.prompt)
        return parts

    def generate(self, request: LLMRequest, model_id: str) -> LLMReply:
        client = self._connect()
        config: dict[str, Any] = {"temperature": request.temperature}
        if request.cached_prefix:
            config["cached_content"] = request.cached_prefix
        contents = self._parts(request)
        last: Exception | None = None
        attempts = len(self.backoff_seconds)
        for attempt in range(attempts):
            started = time.time()
            try:
                response = client.models.generate_content(
                    model=model_id, contents=contents, config=config
                )
            except Exception as error:  # noqa: BLE001 - classified below
                last = error
                message = str(error)
                if any(marker in message for marker in TRANSIENT_MARKERS):
                    if attempt < attempts - 1:
                        self._sleep(self.backoff_seconds[attempt])
                    continue
                condition = terminal_condition(message)
                if condition:
                    raise TerminalProviderError(condition, message) from None
                raise
            return self._reply(request, response, int((time.time() - started) * 1000))
        message = str(last)
        condition = terminal_condition(message)
        if condition:
            raise TerminalProviderError(condition, message)
        raise ProviderError(f"transient errors through all {attempts} attempts: {message[:200]}")

    @staticmethod
    def _reply(request: LLMRequest, response: Any, latency_ms: int) -> LLMReply:
        usage = getattr(response, "usage_metadata", None)
        return LLMReply(
            text=(getattr(response, "text", None) or "").strip(),
            model=request.model,
            tag=request.tag,
            prompt_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            completion_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            reasoning_tokens=int(getattr(usage, "thoughts_token_count", 0) or 0),
            cached_tokens=int(getattr(usage, "cached_content_token_count", 0) or 0),
            response_model=getattr(response, "model_version", None),
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------ prompt cache
    def create_cache(
        self, model_id: str, prefix: str, ttl_seconds: int = 3600, name: str = "shared-prefix"
    ) -> str:
        """Store `prefix` on the server and return the handle to name in later calls.

        A handle belongs to the model it was created for. A stored prefix is billed
        for as long as it lives, so `delete_cache` should follow the run.
        """
        from google.genai import types

        cache = self._connect().caches.create(
            model=model_id,
            config=types.CreateCachedContentConfig(
                contents=[prefix], ttl=f"{int(ttl_seconds)}s", display_name=name
            ),
        )
        return cache.name

    def delete_cache(self, handle: str) -> None:
        """Remove a stored prefix. A failure here never fails a run that has finished."""
        try:
            self._connect().caches.delete(name=handle)
        except Exception:  # noqa: BLE001, S110 - teardown is best effort
            pass
