"""The model catalogue, the spend-gated client, pricing and the providers."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from ad_for_edu.core import spend
from ad_for_edu.core.errors import ContractError, SettingsError, SpendNotApprovedError
from ad_for_edu.llm import (
    PROVIDERS,
    LLMClient,
    LLMProvider,
    LLMReply,
    LLMRequest,
    ModelCatalog,
    PriceTable,
    ProviderError,
    TerminalProviderError,
    check_providers,
    terminal_condition,
)
from ad_for_edu.llm.catalog import ModelEntry
from ad_for_edu.llm.providers.gemini import GeminiNative
from ad_for_edu.llm.providers.huggingface_local import HuggingFaceLocal
from ad_for_edu.llm.providers.openai_compatible import OpenAICompatible, message_content


class Recording(LLMProvider):
    """A provider that answers from a script and records what it was asked."""

    modalities = frozenset({"text", "image", "video"})
    paid = True

    def __init__(self, answers=None):
        self.answers = list(answers or [])
        self.requests = []

    def generate(self, request, model_id):
        self.requests.append((request, model_id))
        answer = self.answers.pop(0) if self.answers else "ok"
        if isinstance(answer, Exception):
            raise answer
        return LLMReply(text=answer, model=request.model, tag=request.tag, prompt_tokens=10,
                        completion_tokens=2)


@pytest.fixture
def catalog(shipped_configs) -> ModelCatalog:
    return ModelCatalog.load(shipped_configs / "models.yaml")


def approved(total: float = 1.0) -> spend.SpendApproval:
    estimate = spend.SpendEstimate("test stage")
    estimate.add("lec-a", "gpt-5.5", 1, total)
    return spend.require_approval(estimate, approved=True, out=io.StringIO())


def client(catalog, provider=None, total: float = 1.0) -> tuple[LLMClient, Recording]:
    recording = provider or Recording()
    names = {entry.provider for entry in catalog.entries.values()}
    return LLMClient(approved(total), catalog, {name: recording for name in names}), recording


# ---------------------------------------------------------------- catalogue
def test_the_shipped_catalogue_is_consistent(catalog):
    check_providers(catalog)
    assert len(catalog) == 9


def test_roles_of_the_shipped_models(catalog):
    assert set(catalog.with_role("classifier")) == {"gemini-3.1-pro", "gemini-3.1-pro-or"}
    assert set(catalog.with_role("pair_judge")) == {"gemini-3.1-pro", "gemini-3.1-pro-or"}
    assert set(catalog.with_role("annotator")) == {"gpt-5.6-terra-pro", "claude-opus-5"}
    assert set(catalog.with_role("metric_judge")) == {
        "gpt-5.5",
        "gpt-3.5-turbo",
        "llama-2-7b-chat",
    }
    assert "qwen3-vl-235b" in catalog.with_role("writer")


def test_no_shipped_model_both_orders_pairs_and_scores(catalog):
    for entry in catalog.entries.values():
        assert not {"pair_judge", "metric_judge"} <= entry.roles, entry.key


def test_a_model_with_both_roles_is_refused(catalog):
    entries = dict(catalog.entries)
    entries["both"] = ModelEntry(
        "both", "openrouter", "vendor/both", frozenset({"text"}),
        frozenset({"pair_judge", "metric_judge"}), (1e-6, 1e-6),
    )
    with pytest.raises(SettingsError) as caught:
        check_providers(ModelCatalog(entries, catalog.providers))
    assert "both" in str(caught.value)


def test_prices_are_stored_per_token(catalog):
    entry = catalog.get("gpt-5.5")
    assert entry.price == pytest.approx((5.0e-6, 30.0e-6))
    assert entry.cost(1_000_000, 0) == pytest.approx(5.0)
    assert entry.cost(0, 1_000_000) == pytest.approx(30.0)


def test_the_local_model_is_free_and_its_provider_too(catalog):
    entry = catalog.get("llama-2-7b-chat")
    assert entry.free
    strategy, _ = catalog.provider_spec(entry.provider)
    assert not PROVIDERS.get(strategy).paid


def test_a_paid_provider_needs_a_price(catalog):
    entries = dict(catalog.entries)
    entries["unpriced"] = ModelEntry(
        "unpriced", "openrouter", "vendor/unpriced", frozenset({"text"}), frozenset({"writer"})
    )
    with pytest.raises(SettingsError):
        check_providers(ModelCatalog(entries, catalog.providers))


def test_one_strategy_can_be_configured_twice(catalog):
    first, first_params = catalog.provider_spec("openrouter")
    second, second_params = catalog.provider_spec("openrouter_rating")
    assert first == second == "openai_compatible"
    assert first_params["limit_field"] == "max_tokens"
    assert second_params["limit_field"] == "max_completion_tokens"


def test_an_unknown_model_lists_the_known_ones(catalog):
    with pytest.raises(ContractError) as caught:
        catalog.get("gpt-9")
    assert "gpt-5.5" in str(caught.value)


@pytest.mark.parametrize(
    "entry",
    [
        "provider: openrouter\n    model_id: v/m\n    roles: [author]",
        "provider: openrouter\n    model_id: v/m\n    roles: []",
        "provider: openrouter\n    model_id: v/m\n    roles: [writer]\n    modalities: [audio]",
        "provider: openrouter\n    model_id: v/m\n    roles: [writer]\n    cost: 3",
        "provider: openrouter\n    roles: [writer]",
    ],
)
def test_a_malformed_entry_is_refused(tmp_path, entry):
    path = tmp_path / "models.yaml"
    path.write_text(f"models:\n  m:\n    {entry}\n", encoding="utf-8")
    with pytest.raises(SettingsError):
        ModelCatalog.load(path)


# ---------------------------------------------------------------- the spend gate
def test_a_client_cannot_be_built_without_an_approval(catalog):
    for stand_in in (None, True, "approved", object()):
        with pytest.raises(SpendNotApprovedError):
            LLMClient(stand_in, catalog)


def test_a_free_approval_does_not_cover_a_paid_model(catalog):
    free = LLMClient(spend.free_approval("local rating"), catalog, {"openrouter": Recording()})
    with pytest.raises(SpendNotApprovedError):
        free.generate(LLMRequest("gpt-5.5", "hello"), role="writer")


def test_a_free_approval_covers_a_local_model(catalog):
    class Local(Recording):
        paid = False

    local = Local(["4"])
    free = LLMClient(spend.free_approval("local rating"), catalog, {"local": local})
    reply = free.generate(LLMRequest("llama-2-7b-chat", "rate"), role="metric_judge")
    assert reply.text == "4"


# ---------------------------------------------------------------- roles and modalities
def test_a_model_is_called_under_a_role_it_holds(catalog):
    llm, recording = client(catalog)
    reply = llm.generate(LLMRequest("gpt-5.5", "describe", tag="lec-a#0001"), role="writer")
    assert reply.tag == "lec-a#0001"
    assert recording.requests[0][1] == "openai/gpt-5.5"
    assert llm.calls == 1 and llm.prompt_tokens == 10


def test_the_metric_judge_may_not_order_pairs(catalog):
    llm, recording = client(catalog)
    with pytest.raises(ContractError) as caught:
        llm.generate(LLMRequest("gpt-5.5", "which is better"), role="pair_judge")
    assert "pair_judge" in str(caught.value)
    assert recording.requests == []


def test_an_unknown_role_is_refused(catalog):
    llm, _ = client(catalog)
    with pytest.raises(ContractError):
        llm.generate(LLMRequest("gpt-5.5", "x"), role="selector")


def test_a_clip_is_not_sent_to_a_model_that_cannot_watch(catalog, tmp_path):
    clip = tmp_path / "clip.bin"
    clip.write_bytes(b"0")
    llm, recording = client(catalog)
    with pytest.raises(ContractError) as caught:
        llm.generate(LLMRequest("gpt-5.5", "watch", videos=(clip,)), role="writer")
    assert "video" in str(caught.value)
    assert recording.requests == []


def test_a_batch_goes_to_one_model_and_keeps_its_order(catalog):
    llm, recording = client(catalog, Recording(["a", ProviderError("timeout"), "c"]))
    requests = [LLMRequest("gpt-5.5", f"p{i}", tag=str(i)) for i in range(3)]
    replies = llm.generate_many(requests, role="writer")
    assert [getattr(r, "text", None) for r in replies] == ["a", None, "c"]
    assert isinstance(replies[1], ProviderError)
    assert llm.calls == 2
    mixed = [LLMRequest("gpt-5.5", "x"), LLMRequest("claude-sonnet-4.6", "y")]
    with pytest.raises(ContractError):
        llm.generate_many(mixed, role="writer")


def test_a_terminal_failure_stops_the_batch(catalog):
    failing = Recording(["a", TerminalProviderError("OUT OF CREDIT", "Error code: 402")])
    llm, _ = client(catalog, failing)
    requests = [LLMRequest("gpt-5.5", f"p{i}") for i in range(3)]
    with pytest.raises(TerminalProviderError):
        llm.generate_many(requests, role="writer")
    assert len(failing.requests) == 2


# ---------------------------------------------------------------- terminal conditions
@pytest.mark.parametrize(
    ("message", "condition"),
    [
        ("Error code: 402 - Insufficient credits", "OUT OF CREDIT"),
        ("insufficient_quota", "OUT OF CREDIT"),
        ("Error code: 401 - invalid_api_key", "AUTHENTICATION REJECTED"),
        ("Error code: 403", "AUTHENTICATION REJECTED"),
        ("Error code: 429", "QUOTA EXHAUSTED"),
        ("RESOURCE_EXHAUSTED: you exceeded your current quota", "QUOTA EXHAUSTED"),
    ],
)
def test_failures_no_retry_will_clear(message, condition):
    assert terminal_condition(message) == condition


@pytest.mark.parametrize(
    "message",
    [
        "timed out while processing lec-a#0402",
        "moment lec-b#0401 could not be read",
        "connection reset",
        "",
    ],
)
def test_a_digit_run_inside_an_identifier_is_not_a_status(message):
    assert terminal_condition(message) is None


# ---------------------------------------------------------------- the chat provider
class FakeCompletions:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **arguments):
        self.calls.append(arguments)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def chat_response(text="ok", finish="stop"):
    return SimpleNamespace(
        id="req-1",
        model="vendor/model",
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=12),
        ),
    )


def chat_provider(script, **params):
    completions = FakeCompletions(script)
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = OpenAICompatible(
        base_url="https://example.invalid/v1",
        credential_variable="UNUSED",
        client_factory=lambda: fake,
        **params,
    )
    provider._sleep = lambda seconds: None
    return provider, completions


def test_media_travel_inside_the_request(tmp_path):
    image, clip = tmp_path / "000010.jpg", tmp_path / "clip.mp4"
    image.write_bytes(b"image-bytes")
    clip.write_bytes(b"clip-bytes")
    content = message_content(LLMRequest("m", "look", images=(image,), videos=(clip,)))
    assert [part["type"] for part in content] == ["text", "image_url", "video_url"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[2]["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert "http" not in content[1]["image_url"]["url"]


def test_the_reply_carries_what_the_service_reported():
    provider, completions = chat_provider([chat_response("  hello  ", finish="length")])
    reply = provider.generate(
        LLMRequest("gpt-5.5", "p", max_tokens=900, reasoning_effort="low", tag="t"),
        "openai/gpt-5.5",
    )
    assert reply.text == "hello" and reply.truncated and reply.tag == "t"
    assert reply.prompt_tokens == 100 and reply.reasoning_tokens == 12
    assert reply.response_model == "vendor/model" and reply.request_id == "req-1"
    sent = completions.calls[0]
    assert sent["max_tokens"] == 900 and sent["temperature"] == 0.0
    assert sent["extra_body"] == {"reasoning": {"effort": "low"}}


def test_the_limit_field_is_a_setting():
    provider, completions = chat_provider([chat_response()], limit_field="max_completion_tokens")
    provider.generate(LLMRequest("gpt-3.5-turbo", "p", max_tokens=16), "openai/gpt-3.5-turbo")
    assert completions.calls[0]["max_completion_tokens"] == 16
    assert "max_tokens" not in completions.calls[0]
    with pytest.raises(ValueError):
        OpenAICompatible("u", "K", limit_field="tokens")


def test_a_parameter_the_model_rejects_is_dropped_and_the_call_repeated():
    script = [RuntimeError("temperature is not supported for this model"), chat_response("fine")]
    provider, completions = chat_provider(script)
    assert provider.generate(LLMRequest("m", "p"), "vendor/m").text == "fine"
    assert "temperature" in completions.calls[0] and "temperature" not in completions.calls[1]


def test_a_transient_failure_is_retried():
    script = [RuntimeError("Connection reset"), RuntimeError("overloaded"), chat_response("fine")]
    provider, completions = chat_provider(script)
    assert provider.generate(LLMRequest("m", "p"), "vendor/m").text == "fine"
    assert len(completions.calls) == 3


def test_a_terminal_failure_is_not_retried():
    provider, completions = chat_provider([RuntimeError("Error code: 402 - Insufficient credits")])
    with pytest.raises(TerminalProviderError) as caught:
        provider.generate(LLMRequest("m", "p"), "vendor/m")
    assert caught.value.condition == "OUT OF CREDIT"
    assert len(completions.calls) == 1


def test_retries_are_bounded():
    provider, completions = chat_provider([RuntimeError("timeout")] * 10, retry_waits=(1, 1))
    with pytest.raises(ProviderError):
        provider.generate(LLMRequest("m", "p"), "vendor/m")
    assert len(completions.calls) == 3


def test_an_unknown_failure_is_raised_as_it_is():
    provider, _ = chat_provider([KeyError("unexpected")])
    with pytest.raises(KeyError):
        provider.generate(LLMRequest("m", "p"), "vendor/m")


def test_this_provider_has_no_prompt_cache():
    provider, completions = chat_provider([chat_response()])
    with pytest.raises(ProviderError):
        provider.generate(LLMRequest("m", "p", cached_prefix="cache/1"), "vendor/m")
    assert completions.calls == []


# ---------------------------------------------------------------- the native provider
def native_provider(script):
    calls = []

    def generate_content(**arguments):
        calls.append(arguments)
        step = script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    fake = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    provider = GeminiNative(
        client_factory=lambda: fake,
        part_factory=lambda data, mime: (mime, len(data)),
        backoff_seconds=(1, 1, 1),
    )
    provider._sleep = lambda seconds: None
    return provider, calls


def native_response(text="ok", cached=0):
    usage = SimpleNamespace(
        prompt_token_count=200,
        candidates_token_count=30,
        thoughts_token_count=5,
        cached_content_token_count=cached,
    )
    return SimpleNamespace(text=text, usage_metadata=usage, model_version="model-v")


def test_images_and_clips_precede_the_prompt(tmp_path):
    image, clip = tmp_path / "a.jpg", tmp_path / "c.mp4"
    image.write_bytes(b"12345")
    clip.write_bytes(b"123")
    provider, calls = native_provider([native_response()])
    provider.generate(LLMRequest("g", "which one", images=(image,), videos=(clip,)), "model")
    assert calls[0]["contents"] == [("image/jpeg", 5), ("video/mp4", 3), "which one"]


def test_the_cache_handle_is_named_and_the_saving_reported():
    provider, calls = native_provider([native_response(cached=8700)])
    reply = provider.generate(LLMRequest("g", "p", cached_prefix="cachedContents/1"), "model")
    assert calls[0]["config"]["cached_content"] == "cachedContents/1"
    assert reply.cached_tokens == 8700


def test_a_transient_failure_of_the_native_service_is_retried():
    provider, calls = native_provider([RuntimeError("503 UNAVAILABLE"), native_response("fine")])
    assert provider.generate(LLMRequest("g", "p"), "model").text == "fine"
    assert len(calls) == 2


def test_quota_that_never_clears_ends_as_a_terminal_failure():
    provider, calls = native_provider([RuntimeError("429 RESOURCE_EXHAUSTED")] * 5)
    with pytest.raises(TerminalProviderError) as caught:
        provider.generate(LLMRequest("g", "p"), "model")
    assert caught.value.condition == "QUOTA EXHAUSTED"
    assert len(calls) == 3


def test_a_rejected_key_is_terminal_at_once():
    provider, calls = native_provider([RuntimeError("Error code: 401 invalid_api_key")])
    with pytest.raises(TerminalProviderError):
        provider.generate(LLMRequest("g", "p"), "model")
    assert len(calls) == 1


# ---------------------------------------------------------------- the local provider
class FakeTokenizer:
    chat_template = "{% for m in messages %}[INST] {{ m.content }} [/INST]{% endfor %}"
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "<s>[INST] " + messages[0]["content"] + " [/INST]"

    def __call__(self, text, add_special_tokens):
        body = [10 + (ord(char) % 50) for char in text.replace("<s>", "")]
        begin = [self.bos_token_id] if text.startswith("<s>") else []
        extra = [self.bos_token_id] if add_special_tokens else []
        return {"input_ids": extra + begin + body}


def local_provider(tokenizer=None, **params):
    loaded = []

    def loader(model_id, revision, dtype, device):
        loaded.append(model_id)
        return tokenizer or FakeTokenizer(), object()

    return HuggingFaceLocal(loader=loader, **params), loaded


def test_a_prompt_is_rendered_with_the_chat_template_and_one_beginning_token():
    provider, loaded = local_provider()
    tokenizer, _ = provider._model("vendor/chat")
    assert provider.render(tokenizer, "rate this") == "<s>[INST] rate this [/INST]"
    assert provider._encode(tokenizer, ["rate this"])[0][:2].count(1) == 1
    assert loaded == ["vendor/chat"]


def test_a_model_without_its_chat_format_is_refused():
    class Plain(FakeTokenizer):
        chat_template = "{{ messages[0].content }}"

    provider, _ = local_provider(Plain())
    with pytest.raises(ContractError) as caught:
        provider._model("vendor/base")
    assert "[INST]" in str(caught.value)


def test_a_doubled_beginning_token_is_refused():
    class Doubling(FakeTokenizer):
        def __call__(self, text, add_special_tokens):
            ids = super().__call__(text, add_special_tokens)["input_ids"]
            return {"input_ids": [self.bos_token_id] + ids}

    provider, _ = local_provider(Doubling())
    with pytest.raises(ContractError) as caught:
        provider._model("vendor/chat")
    assert "2 beginning-of-sequence" in str(caught.value)


def test_a_batch_that_does_not_fit_is_split_until_it_does():
    provider, _ = local_provider(batch_size=4)
    sizes = []

    def decode(batch, tokenizer, model):
        sizes.append(len(batch))
        if len(batch) > 1:
            raise RuntimeError("CUDA out of memory")
        return [("3", 1)]

    provider._decode = decode
    requests = [LLMRequest("llama", f"prompt {i}", tag=str(i)) for i in range(4)]
    replies = provider.generate_many(requests, "vendor/chat")
    assert [reply.text for reply in replies] == ["3"] * 4
    assert sorted(reply.tag for reply in replies) == ["0", "1", "2", "3"]
    assert sizes == [4, 2, 1, 1, 2, 1, 1]
    assert provider.memory_splits == 3


def test_another_failure_is_handed_back_per_prompt_without_a_traceback():
    provider, _ = local_provider(batch_size=2)

    def decode(batch, tokenizer, model):
        raise ValueError("bad batch")

    provider._decode = decode
    replies = provider.generate_many([LLMRequest("llama", "a"), LLMRequest("llama", "b")], "v/c")
    assert all(isinstance(reply, ValueError) for reply in replies)
    assert all(reply.__traceback__ is None for reply in replies)


def test_replies_keep_the_order_of_the_requests_whatever_their_length():
    provider, _ = local_provider(batch_size=2)
    provider._decode = lambda batch, tokenizer, model: [(str(len(row)), 1) for row in batch]
    prompts = ["a much longer prompt than the others", "b", "medium prompt"]
    replies = provider.generate_many([LLMRequest("llama", p, tag=p) for p in prompts], "v/c")
    assert [reply.tag for reply in replies] == prompts
    lengths = [int(reply.text) for reply in replies]
    assert lengths[0] > lengths[2] > lengths[1]


# ---------------------------------------------------------------- pricing
@pytest.fixture
def prices(catalog, shipped_configs) -> PriceTable:
    return PriceTable.load(catalog, shipped_configs / "pricing.yaml")


def test_a_measured_shape_is_preferred_to_the_assumed_one(prices):
    assert prices.is_measured("gpt-5.5", "generate_references")
    assert not prices.is_measured("gpt-5.5", "generate_candidates")
    measured = prices.per_call("gpt-5.5", "generate_references")
    assert measured == pytest.approx(11844 * 5e-6 + 373 * 30e-6)


def test_the_same_stage_costs_differently_per_model(prices):
    cheap = prices.per_call("qwen3-vl-235b", "generate_references")
    dear = prices.per_call("gpt-5.5", "generate_references")
    assert dear > 10 * cheap


def test_an_estimate_is_built_per_row_and_per_model(prices):
    estimate = prices.estimate(
        "generate_references",
        {"lec-a": {"gpt-5.5": 10, "claude-sonnet-4.6": 10}, "lec-b": {"gpt-5.5": 5}},
        free_columns=("chrF",),
    )
    assert estimate.calls == 25 and len(estimate.lines) == 3
    assert estimate.all_measured
    expected = 15 * prices.per_call("gpt-5.5", "generate_references") + 10 * prices.per_call(
        "claude-sonnet-4.6", "generate_references"
    )
    assert estimate.total == pytest.approx(expected)


def test_an_estimate_on_an_assumed_shape_says_so(prices):
    estimate = prices.estimate("score_rubric", {"draw": {"gpt-5.5": 150}})
    assert not estimate.all_measured
    assert "assumed" in estimate.render()


@pytest.mark.parametrize("calls", [25, {"lec-a": 25}, {"lec-a": ["gpt-5.5"]}])
def test_a_single_number_of_calls_is_refused(prices, calls):
    with pytest.raises(ContractError):
        prices.estimate("generate_references", calls)


def test_an_unknown_stage_or_model_is_refused(prices):
    with pytest.raises(ContractError):
        prices.per_call("gpt-5.5", "generate_poems")
    with pytest.raises(ContractError):
        prices.estimate("generate_references", {"lec-a": {"gpt-9": 1}})


def test_a_shape_measured_for_an_unknown_model_is_refused(catalog, tmp_path):
    path = tmp_path / "pricing.yaml"
    path.write_text(
        "stages:\n  s:\n    input_tokens: 1\n    output_tokens: 1\n"
        "    measured:\n      gpt-9: {input_tokens: 1, output_tokens: 1}\n",
        encoding="utf-8",
    )
    with pytest.raises(SettingsError):
        PriceTable.load(catalog, path)


def test_every_stage_with_a_paid_model_has_a_shape(prices):
    assert set(prices.stages) == {
        "classify_moments",
        "generate_references",
        "generate_candidates",
        "generate_controlled_pairs",
        "judge_pairs",
        "rate_against_references",
        "score_rubric",
        "judge_rules",
        "judge_head_to_head",
        "annotate_rated_pairs",
    }
