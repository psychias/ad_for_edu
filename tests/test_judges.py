"""The judges: what each asks, how each reads a reply, and how a run is driven."""

from __future__ import annotations

import io

import pytest

from ad_for_edu.core import spend
from ad_for_edu.core.errors import StrategyConfigError
from ad_for_edu.core.io import read_jsonl
from ad_for_edu.data.schema import CATEGORIES
from ad_for_edu.judges import (
    JUDGES,
    DescriptionToGrade,
    Judge,
    JudgeRow,
    JudgeRunner,
    PairToJudge,
    RuleToCheck,
    compliance_per_rule,
    estimate_for,
)
from ad_for_edu.llm import LLMClient, LLMProvider, LLMReply, ModelCatalog, PriceTable
from ad_for_edu.llm.base import TerminalProviderError

PAIR = {
    "pair_id": "p1",
    "moment_id": "lec-a#0001",
    "a": "The nucleus, at the centre.",
    "b": "The cursor moves to it.",
}
CONTEXT = {
    "type": "pointing",
    "transcript_window": "and this is where it is stored",
    "slide_ocr": "Nucleus, nucleolus",
    "what_on_screen": "A cell with its nucleus",
    "pause_after": 1.5,
    "subject": "biology",
}


class Scripted(LLMProvider):
    """A provider that answers from a script and records what it was asked."""

    modalities = frozenset({"text", "image", "video"})
    paid = True

    def __init__(self, answers):
        self.answers = list(answers)
        self.requests = []

    def generate(self, request, model_id):
        self.requests.append(request)
        answer = self.answers.pop(0) if self.answers else ""
        if isinstance(answer, Exception):
            raise answer
        return LLMReply(text=answer, model=request.model, tag=request.tag)


@pytest.fixture
def catalog(shipped_configs) -> ModelCatalog:
    return ModelCatalog.load(shipped_configs / "models.yaml")


def client_with(catalog, provider) -> LLMClient:
    estimate = spend.SpendEstimate("judging")
    estimate.add("all", "gemini-3.1-pro-or", 1, 0.01)
    approval = spend.require_approval(estimate, approved=True, out=io.StringIO())
    names = {entry.provider for entry in catalog.entries.values()}
    return LLMClient(approval, catalog, {name: provider for name in names})


# ---------------------------------------------------------------- the family
def test_the_judges_of_the_pipeline():
    assert JUDGES.names() == (
        "reference_rating",
        "reference_rating_pooled",
        "rubric",
        "rule_compliance",
        "sheet_pairwise",
        "text_pairwise",
        "video_pairwise",
    )


def test_each_judge_names_the_stage_it_belongs_to(shipped_configs, catalog):
    prices = PriceTable.load(catalog, shipped_configs / "pricing.yaml")
    for name in JUDGES.names():
        assert JUDGES.get(name).stage in prices.stages


def test_only_the_judge_that_must_watch_does():
    assert JUDGES.get("video_pairwise").watches
    assert not any(JUDGES.get(name).watches for name in JUDGES.names() if name != "video_pairwise")


# ---------------------------------------------------------------- pairs
def test_the_second_order_swaps_the_sides():
    first = PairToJudge.from_pair(PAIR, 0)
    second = PairToJudge.from_pair(PAIR, 1)
    assert (first.first, first.second) == (PAIR["a"], PAIR["b"])
    assert (second.first, second.second) == (PAIR["b"], PAIR["a"])
    assert first.item_id == "p1#order0" and second.item_id == "p1#order1"


def test_the_judge_that_watches_sends_the_clip(tmp_path):
    clip = tmp_path / "lec-a#0001.mp4"
    clip.write_bytes(b"clip")
    judge = JUDGES.create("video_pairwise", model="gemini-3.1-pro-or")
    request = judge.request(PairToJudge.from_pair(PAIR, 0, clip=clip))
    assert request.videos == (clip,) and not request.images
    assert PAIR["a"] in request.prompt and PAIR["b"] in request.prompt
    assert request.tag == "p1#order0"


def test_the_judge_that_reads_text_sends_the_context():
    judge = JUDGES.create("text_pairwise", model="gpt-5.6-terra-pro")
    request = judge.request(PairToJudge.from_pair(PAIR, 0, context=CONTEXT))
    assert "pointing" in request.prompt
    assert CONTEXT["transcript_window"] in request.prompt
    assert not request.videos and not request.images


def test_the_judge_that_stands_in_for_a_rater_is_shown_what_a_rater_sees(tmp_path):
    still = tmp_path / "still.jpg"
    still.write_bytes(b"still")
    judge = JUDGES.create("sheet_pairwise", model="claude-opus-5")
    request = judge.request(PairToJudge.from_pair(PAIR, 1, still=still, context=CONTEXT))
    assert request.images == (still,)
    assert "biology" in request.prompt
    # Shown the other way round, so the description recorded second comes first.
    assert request.prompt.index(PAIR["b"]) < request.prompt.index(PAIR["a"])


@pytest.mark.parametrize("name", ["video_pairwise", "text_pairwise", "sheet_pairwise"])
def test_a_pairwise_judge_hands_back_the_option_it_was_told(name):
    judge = JUDGES.create(name, model="gemini-3.1-pro-or")
    assert judge.parse("1") == "1" and judge.parse("2") == "2"
    assert judge.parse("TIE") == "tie"
    assert judge.parse("Description 1 is better than Description 2.") is None


# ---------------------------------------------------------------- grading
def graded(**fields) -> DescriptionToGrade:
    values = {
        "moment_id": "lec-a#0001",
        "text": "The nucleus, at the centre.",
        "references": ("A dark nucleolus sits inside the nucleus.",),
        "standard": "the compressed standard",
        "context": CONTEXT,
        "system": "system-x",
    }
    values.update(fields)
    return DescriptionToGrade(**values)


def rubric_reply(overall="7", **grades) -> str:
    lines = [f"{category}: {grades.get(category, 4)}" for category in CATEGORIES]
    return "\n".join(lines + [f"overall: {overall}"])


def test_the_rubric_is_given_the_standard_the_context_and_the_references():
    judge = JUDGES.create("rubric", model="gpt-5.5")
    request = judge.request(graded())
    assert "the compressed standard" in request.prompt
    assert "A dark nucleolus" in request.prompt
    assert CONTEXT["slide_ocr"] in request.prompt
    assert request.tag == "lec-a#0001::system-x"


def test_the_rubric_reads_seven_grades():
    judge = JUDGES.create("rubric", model="gpt-5.5")
    grades = judge.parse(rubric_reply(overall="7", deixis=5, length=2))
    assert grades.overall == 7.0
    assert grades.categories["deixis"] == 5.0 and grades.categories["length"] == 2.0
    assert set(grades.categories) == set(CATEGORIES)
    assert grades.as_dict()["overall"] == 7.0


def test_a_rubric_reply_missing_a_category_is_not_read_as_a_low_grade():
    judge = JUDGES.create("rubric", model="gpt-5.5")
    without = "\n".join(
        line for line in rubric_reply().splitlines() if not line.startswith("deixis")
    )
    assert judge.parse(without) is None


def test_a_rubric_grade_outside_its_scale_is_not_read():
    judge = JUDGES.create("rubric", model="gpt-5.5")
    assert judge.parse(rubric_reply(style=9)) is None
    assert judge.parse(rubric_reply(overall="12")) is None
    assert judge.parse("") is None and judge.parse(None) is None


def test_the_two_scales_of_the_rubric():
    from ad_for_edu.judges import CATEGORY_SCALE, OVERALL_SCALE

    assert CATEGORY_SCALE == (1, 5) and OVERALL_SCALE == (1, 10)


def test_rating_against_one_reference_and_against_all_of_them():
    one = JUDGES.create("reference_rating", model="gpt-5.5")
    pooled = JUDGES.create("reference_rating_pooled", model="gpt-5.5")
    item = graded(references=("first reference", "second reference"))
    assert "first reference" in one.request(item).prompt
    assert "second reference" not in one.request(item).prompt
    assert "second reference" in pooled.request(item).prompt


def test_a_rating_is_read_on_its_scale():
    judge = JUDGES.create("reference_rating", model="gpt-5.5")
    assert judge.parse("4") == 4
    assert judge.parse("4 out of 5") == 4
    assert judge.parse("9") is None
    assert judge.parse("good") is None


def test_the_rule_judge_asks_about_one_named_rule():
    from ad_for_edu.standard import RuleBook, render_rule_prompt

    book = RuleBook.load()
    prompt = render_rule_prompt(book.get("deixis_004"))
    item = RuleToCheck(
        "lec-a#0001",
        "deixis_004",
        prompt,
        {
            "type": "pointing",
            "what_on_screen": "A cell",
            "transcript_window": "this one here",
            "emit": True,
            "ad_text": "The cursor moves to it.",
        },
    )
    judge = JUDGES.create("rule_compliance", model="gpt-5.5")
    request = judge.request(item)
    assert "deixis_004" in request.prompt and "The cursor moves to it." in request.prompt
    assert "{" not in request.prompt.replace('{"verdict"', "")
    assert request.tag == "lec-a#0001::deixis_004"


def test_the_rule_judge_reads_a_verdict():
    judge = JUDGES.create("rule_compliance", model="gpt-5.5")
    assert judge.parse('{"verdict": "FAIL", "reasoning": "narrates the pointing"}') == "FAIL"
    assert judge.parse('{"verdict": "PASS"}') == "PASS"
    assert judge.parse("it fails") is None


def test_the_share_kept_is_counted_per_rule():
    rows = [
        {"rule_id": "deixis_004", "answer": "PASS"},
        {"rule_id": "deixis_004", "answer": "FAIL"},
        {"rule_id": "length_001", "answer": "PASS"},
        {"rule_id": "length_001", "error": "timeout"},
        {"rule_id": "style_001", "error": "timeout"},
    ]
    assert compliance_per_rule(rows) == {"deixis_004": 0.5, "length_001": 1.0}


# ---------------------------------------------------------------- the runner
def pairs_to_judge(count: int) -> list[PairToJudge]:
    return [
        PairToJudge.from_pair({**PAIR, "pair_id": f"p{index}"}, order)
        for index in range(count)
        for order in (0, 1)
    ]


def test_every_answer_is_written_as_it_arrives(catalog, tmp_path):
    provider = Scripted(["1", "2", "1", "1"])
    runner = JudgeRunner(
        JUDGES.create("video_pairwise", model="gemini-3.1-pro-or"),
        client_with(catalog, provider),
        role="pair_judge",
        fields_of=lambda item: {"pair_id": item.pair_id, "order": item.order},
    )
    output = tmp_path / "verdicts.jsonl"
    summary = runner.run(pairs_to_judge(2), output)
    assert summary.asked == 4 and summary.answered == 4
    rows = read_jsonl(output)
    assert [row["item_id"] for row in rows] == [
        "p0#order0",
        "p0#order1",
        "p1#order0",
        "p1#order1",
    ]
    assert rows[0]["answer"] == "1" and rows[0]["order"] == 0
    assert rows[0]["judge"] == "video_pairwise"


def test_a_reply_that_cannot_be_read_is_recorded_with_the_reply(catalog, tmp_path):
    provider = Scripted(["1", "I cannot decide between these two descriptions at all"])
    runner = JudgeRunner(
        JUDGES.create("video_pairwise", model="gemini-3.1-pro-or"),
        client_with(catalog, provider),
        role="pair_judge",
    )
    output = tmp_path / "verdicts.jsonl"
    summary = runner.run(pairs_to_judge(1), output)
    assert summary.answered == 1 and summary.unread == 1
    rows = read_jsonl(output)
    assert rows[1]["error"] and rows[1]["raw"].startswith("I cannot decide")


def test_a_failed_call_is_recorded_and_the_run_goes_on(catalog, tmp_path):
    from ad_for_edu.llm.base import ProviderError

    provider = Scripted(["1", ProviderError("timeout"), "2", "2"])
    runner = JudgeRunner(
        JUDGES.create("video_pairwise", model="gemini-3.1-pro-or"),
        client_with(catalog, provider),
        role="pair_judge",
    )
    output = tmp_path / "verdicts.jsonl"
    summary = runner.run(pairs_to_judge(2), output)
    assert summary.failed == 1 and summary.answered == 3
    assert "timeout" in read_jsonl(output)[1]["error"]


def test_a_failure_no_retry_will_clear_stops_the_run(catalog, tmp_path):
    provider = Scripted(["1", TerminalProviderError("OUT OF CREDIT", "Error code: 402")])
    runner = JudgeRunner(
        JUDGES.create("video_pairwise", model="gemini-3.1-pro-or"),
        client_with(catalog, provider),
        role="pair_judge",
    )
    with pytest.raises(TerminalProviderError):
        runner.run(pairs_to_judge(2), tmp_path / "verdicts.jsonl")


def test_a_resumed_run_asks_only_about_what_has_no_answer(catalog, tmp_path):
    output = tmp_path / "verdicts.jsonl"
    first = Scripted(["1", "unreadable prose that names neither option at all"])
    runner = JudgeRunner(
        JUDGES.create("video_pairwise", model="gemini-3.1-pro-or"),
        client_with(catalog, first),
        role="pair_judge",
    )
    runner.run(pairs_to_judge(1), output)
    second = Scripted(["2"])
    resumed = JudgeRunner(
        JUDGES.create("video_pairwise", model="gemini-3.1-pro-or"),
        client_with(catalog, second),
        role="pair_judge",
    )
    summary = resumed.run(pairs_to_judge(1), output)
    assert summary.skipped == 1 and summary.asked == 1
    assert [request.tag for request in second.requests] == ["p0#order1"]


def test_a_run_can_be_told_not_to_resume(catalog, tmp_path):
    output = tmp_path / "verdicts.jsonl"
    runner = JudgeRunner(
        JUDGES.create("video_pairwise", model="gemini-3.1-pro-or"),
        client_with(catalog, Scripted(["1", "2"])),
        role="pair_judge",
    )
    runner.run(pairs_to_judge(1), output)
    again = JudgeRunner(
        JUDGES.create("video_pairwise", model="gemini-3.1-pro-or"),
        client_with(catalog, Scripted(["1", "2"])),
        role="pair_judge",
    )
    assert again.run(pairs_to_judge(1), output, resume=False).asked == 2


def test_a_judge_is_only_run_in_a_role_its_model_holds(catalog, tmp_path):
    from ad_for_edu.core.errors import ContractError

    runner = JudgeRunner(
        JUDGES.create("rubric", model="gpt-5.5"),
        client_with(catalog, Scripted([rubric_reply()])),
        role="pair_judge",
    )
    with pytest.raises(ContractError):
        runner.run([graded()], tmp_path / "grades.jsonl")


def test_what_a_judging_run_would_cost(catalog, shipped_configs):
    prices = PriceTable.load(catalog, shipped_configs / "pricing.yaml")
    judge = JUDGES.create("video_pairwise", model="gemini-3.1-pro-or")
    estimate = estimate_for(judge, pairs_to_judge(75), prices)
    assert estimate.calls == 150
    assert estimate.stage == "judge_pairs"
    assert estimate.total == pytest.approx(
        150 * prices.per_call("gemini-3.1-pro-or", "judge_pairs")
    )


def test_a_row_carries_what_was_asked_as_well_as_what_came_back():
    row = JudgeRow("p1#order0", "video_pairwise", "model-x", "1", fields={"pair_id": "p1"})
    written = row.as_dict()
    assert written["item_id"] == "p1#order0" and written["answer"] == "1"
    assert written["pair_id"] == "p1" and written["model"] == "model-x"
    assert row.answered


def test_a_judge_outside_the_family_cannot_be_registered():
    with pytest.raises(StrategyConfigError, match="does not derive from"):

        @JUDGES.register("stranger")
        class Stranger:
            pass


def test_every_registered_judge_implements_the_interface():
    for name in JUDGES.names():
        assert issubclass(JUDGES.get(name), Judge)
