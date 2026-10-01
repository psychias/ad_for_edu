"""A chat model run on this machine, for judging that needs no service.

Two properties of the rendered prompt are checked when the provider is built,
because a judge run without its chat format answers in a different register and
its ratings would still parse: the chat template must contain the marker the
model was trained with, and a rendered prompt must start with exactly one
beginning-of-sequence token.

Decoding is greedy with a small token limit, since the prompts ask for a single
token. Prompts are batched by length. When a batch does not fit the memory of
the device it is split in two and tried again, down to single prompts, and the
number of splits is recorded.
"""

from __future__ import annotations

import gc
import hashlib
import time
from collections.abc import Callable, Sequence
from typing import Any

from ...core.errors import ContractError
from ..base import PROVIDERS, TEXT, LLMProvider, LLMReply, LLMRequest

Loader = Callable[[str, "str | None", str, str], "tuple[Any, Any]"]


def out_of_memory(error: BaseException) -> bool:
    return type(error).__name__ == "OutOfMemoryError" or "out of memory" in str(error).lower()


def release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001, S110 - best effort before a retry
        pass


def load_model(model_id: str, revision: str | None, dtype: str, device: str) -> tuple[Any, Any]:
    """The tokenizer and the model, ready for batched greedy decoding."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    # New tokens of every row must start in the same column, so rows are padded on the left.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    precision = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=precision,
        device_map={"": device} if torch.cuda.is_available() else None,
    )
    model.eval()
    return tokenizer, model


@PROVIDERS.register("huggingface_local")
class HuggingFaceLocal(LLMProvider):
    """Greedy, batched decoding with a local chat model."""

    modalities = frozenset({TEXT})
    paid = False

    def __init__(
        self,
        template_marker: str = "[INST]",
        dtype: str = "float16",
        max_new_tokens: int = 8,
        batch_size: int = 32,
        device: str = "cuda:0",
        revision: str | None = None,
        assistant_prefill: str = "",
        loader: Loader | None = None,
    ) -> None:
        self.template_marker = template_marker
        self.dtype = dtype
        self.max_new_tokens = int(max_new_tokens)
        self.batch_size = max(1, int(batch_size))
        self.device = device
        self.revision = revision
        self.assistant_prefill = assistant_prefill or ""
        self._loader = loader or load_model
        self._loaded: dict[str, tuple[Any, Any]] = {}
        self.memory_splits = 0

    # ------------------------------------------------------------ loading
    def _model(self, model_id: str) -> tuple[Any, Any]:
        if model_id not in self._loaded:
            tokenizer, model = self._loader(model_id, self.revision, self.dtype, self.device)
            self._check_rendering(model_id, tokenizer)
            self._loaded[model_id] = (tokenizer, model)
        return self._loaded[model_id]

    def _check_rendering(self, model_id: str, tokenizer: Any) -> None:
        template = getattr(tokenizer, "chat_template", None) or ""
        if self.template_marker and self.template_marker not in template:
            raise ContractError(
                f"{model_id}: the chat template does not contain {self.template_marker!r}. "
                "A judge run without its chat format answers in a different register."
            )
        probe = self._encode(tokenizer, ["probe"])[0]
        beginning = getattr(tokenizer, "bos_token_id", None)
        count = sum(1 for token in probe[:2] if token == beginning)
        if beginning is None or count != 1:
            raise ContractError(
                f"{model_id}: a rendered prompt starts with {count} beginning-of-sequence "
                "tokens; exactly one is expected"
            )

    def describe(self, model_id: str) -> dict[str, Any]:
        """What judged, for the record of a run."""
        tokenizer, _model = self._model(model_id)
        template = getattr(tokenizer, "chat_template", None) or ""
        return {
            "provider": "huggingface_local",
            "model_id": model_id,
            "revision": self.revision,
            "dtype": self.dtype,
            "decoding": {"greedy": True, "max_new_tokens": self.max_new_tokens},
            "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
            "assistant_prefill": self.assistant_prefill,
            "batch_size": self.batch_size,
            "memory_splits": self.memory_splits,
        }

    # ------------------------------------------------------------ rendering
    def render(self, tokenizer: Any, prompt: str) -> str:
        """The exact string the model reads: chat template, generation prompt, prefill."""
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        return text + self.assistant_prefill

    def _encode(self, tokenizer: Any, prompts: Sequence[str]) -> list[list[int]]:
        # The template writes the beginning-of-sequence token itself, so the tokenizer
        # must not add a second one.
        return [
            tokenizer(self.render(tokenizer, prompt), add_special_tokens=False)["input_ids"]
            for prompt in prompts
        ]

    # ------------------------------------------------------------ calls
    def generate(self, request: LLMRequest, model_id: str) -> LLMReply:
        reply = self.generate_many([request], model_id)[0]
        if isinstance(reply, Exception):
            raise reply
        return reply

    def generate_many(
        self, requests: Sequence[LLMRequest], model_id: str
    ) -> list[LLMReply | Exception]:
        if not requests:
            return []
        tokenizer, model = self._model(model_id)
        encoded = self._encode(tokenizer, [request.prompt for request in requests])
        by_length = sorted(range(len(requests)), key=lambda index: len(encoded[index]))
        replies: list[LLMReply | Exception | None] = [None] * len(requests)
        for start in range(0, len(by_length), self.batch_size):
            batch = by_length[start : start + self.batch_size]
            self._run(batch, requests, encoded, replies, tokenizer, model, model_id)
        return [reply for reply in replies if reply is not None]

    def _run(
        self,
        batch: list[int],
        requests: Sequence[LLMRequest],
        encoded: list[list[int]],
        replies: list[LLMReply | Exception | None],
        tokenizer: Any,
        model: Any,
        model_id: str,
    ) -> None:
        started = time.time()
        try:
            decoded = self._decode([encoded[index] for index in batch], tokenizer, model)
        except Exception as error:  # noqa: BLE001 - handed back per unit
            full = out_of_memory(error)
            # A stored error keeps no traceback: its frames would hold the tensors of
            # the failed batch, and the memory they occupy would never be released.
            stored = error.with_traceback(None)
            del error
            if full:
                release_memory()
            if full and len(batch) > 1:
                self.memory_splits += 1
                middle = len(batch) // 2
                self._run(batch[:middle], requests, encoded, replies, tokenizer, model, model_id)
                self._run(batch[middle:], requests, encoded, replies, tokenizer, model, model_id)
                return
            for index in batch:
                replies[index] = stored
            return
        latency = int((time.time() - started) * 1000)
        for index, (text, produced) in zip(batch, decoded, strict=True):
            replies[index] = LLMReply(
                text=text,
                model=requests[index].model,
                tag=requests[index].tag,
                prompt_tokens=len(encoded[index]),
                completion_tokens=produced,
                response_model=model_id,
                latency_ms=latency,
            )

    def _decode(
        self, batch: list[list[int]], tokenizer: Any, model: Any
    ) -> list[tuple[str, int]]:
        import torch

        padding = tokenizer.pad_token_id
        width = max(len(row) for row in batch)
        input_ids = torch.full((len(batch), width), padding, dtype=torch.long)
        attention = torch.zeros((len(batch), width), dtype=torch.long)
        for row, tokens in enumerate(batch):
            input_ids[row, width - len(tokens) :] = torch.tensor(tokens, dtype=torch.long)
            attention[row, width - len(tokens) :] = 1
        device = next(model.parameters()).device
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention.to(device),
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=padding,
            )
        end = tokenizer.eos_token_id
        decoded = []
        for row in output[:, width:].tolist():
            kept = []
            for token in row:
                if token in (end, padding):
                    break
                kept.append(token)
            decoded.append((tokenizer.decode(kept, skip_special_tokens=True), len(kept)))
        return decoded
