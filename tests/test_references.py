"""Reference descriptions: the memory per lecture, the ways of asking, the writers."""

from __future__ import annotations

import pytest

from ad_for_edu.core.errors import ContractError, SettingsError
from ad_for_edu.core.text import NO_SPEECH
from ad_for_edu.references import (
    COVERAGE,
    REFERENCE_PROMPTS,
    DescribedSoFar,
    ReferenceDecision,
    ReferenceGenerator,
    WriterAnswer,
    by_lecture,
    complete_only,
    in_order,
    incomplete_moments,
    to_row,
    writers_per_moment,
)
from ad_for_edu.references.build import (
    build_generator,
    build_prompt,
    load_reference_settings,
)
from ad_for_edu.references.described import NOTHING_YET, STRUCTURAL

CONTEXT = {
    "type": "figure",
    "transcript_window": "so the energy of the cell is produced in this organelle",
    "slide_ocr": "Mitochondrion: cristae, matrix",
    "pause_after": 1.5,
    "reachable_rung": 3,
    "described_so_far": NOTHING_YET,
}


# ---------------------------------------------------------------- the memory
def test_the_memory_starts_by_saying_it_is_empty():
    memory = DescribedSoFar("lec-a")
    assert memory.prompt_value() == NOTHING_YET
    assert len(memory) == 0


def test_an_element_named_on_the_slide_is_recorded():
    memory = DescribedSoFar("lec-a")
    added = memory.add("Folded cristae fill the mitochondrion.", "Mitochondrion cristae matrix")
    assert sorted(added) == ["cristae", "mitochondrion"]
    assert "Cristae" in memory or "cristae" in memory
    assert memory.prompt_value() == "cristae; mitochondrion"


def test_a_word_not_on_the_slide_is_not_recorded():
    memory = DescribedSoFar("lec-a")
    memory.add("A beautiful folded cristae structure.", "Mitochondrion cristae")
    assert "beautiful" not in memory and "structure" not in memory
    assert "cristae" in memory


def test_without_a_slide_every_content_word_is_recorded():
    memory = DescribedSoFar("lec-a")
    added = memory.add("Folded cristae appear.", "")
    assert "cristae" in {term.lower() for term in added}


def test_a_word_that_names_a_kind_of_thing_is_not_an_element():
    memory = DescribedSoFar("lec-a")
    memory.add("The figure shows a table and a chart.", "figure table chart cristae")
    assert len(memory) == 0
    assert {"figure", "table", "chart"} <= STRUCTURAL


def test_a_short_word_is_not_an_element():
    memory = DescribedSoFar("lec-a")
    memory.add("The ion and the axon.", "ion axon")
    assert "ion" not in memory and "axon" in memory


def test_an_element_is_recorded_once():
    memory = DescribedSoFar("lec-a")
    first = memory.add("The cristae fold.", "cristae")
    again = memory.add("The cristae fold again.", "cristae")
    assert first and again == []
    assert len(memory) == 1


def test_the_prompt_carries_the_most_recent_elements_only():
    memory = DescribedSoFar("lec-a", window=2)
    slide = "alpha beta gamma"
    memory.add("alpha", slide)
    memory.add("beta", slide)
    memory.add("gamma", slide)
    assert memory.prompt_value() == "beta; gamma"
    assert len(memory) == 3


def test_a_description_that_says_nothing_records_nothing():
    memory = DescribedSoFar("lec-a")
    assert memory.add(None, "cristae") == [] and memory.add("  ", "cristae") == []


def test_the_memory_survives_being_written_out_and_read_back():
    memory = DescribedSoFar("lec-a", window=5)
    memory.add("Folded cristae.", "cristae")
    again = DescribedSoFar.from_dict(memory.as_dict())
    assert again.prompt_value() == memory.prompt_value()
    assert "cristae" in again and again.add("More cristae.", "cristae") == []


# ---------------------------------------------------------------- moments in order
def test_the_moments_of_a_lecture_are_written_in_order_and_carry_the_memory():
    seen = []

    def write(moment, described):
        seen.append((moment["id"], described))
        return {"emit": True, "ad_text": moment["text"]}

    moments = [
        {"id": "lec-a#0001", "text": "Folded cristae.", "slide": "cristae"},
        {"id": "lec-a#0002", "text": "The matrix fills it.", "slide": "matrix"},
    ]
    written, memory = in_order(
        moments, write, lecture="lec-a", slide_of=lambda moment: moment["slide"]
    )
    assert [name for name, _ in seen] == ["lec-a#0001", "lec-a#0002"]
    assert seen[0][1] == NOTHING_YET and "cristae" in seen[1][1]
    assert [row["new_elements"] for row in written] == [["cristae"], ["matrix"]]
    assert len(memory) == 2


def test_a_moment_passed_over_in_silence_adds_nothing():
    def write(moment, described):
        return {"emit": False, "ad_text": None}

    moments = [{"id": "lec-a#0001", "slide": "cristae"}]
    _written, memory = in_order(moments, write, lecture="lec-a")
    assert len(memory) == 0


def test_moments_are_grouped_by_lecture_keeping_their_order():
    moments = [
        {"id": "lec-a#0001"},
        {"id": "lec-b#0001"},
        {"id": "lec-a#0002"},
    ]
    grouped = by_lecture(moments, lambda moment: moment["id"].split("#")[0])
    assert list(grouped) == ["lec-a", "lec-b"]
    assert [moment["id"] for moment in grouped["lec-a"]] == ["lec-a#0001", "lec-a#0002"]


# ---------------------------------------------------------------- the ways of asking
def test_the_ways_of_asking_in_use():
    assert REFERENCE_PROMPTS.names() == (
        "fixed_window",
        "next_event_window",
        "visual_condition",
    )


def test_each_way_of_asking_names_its_window_and_its_template():
    windows = {name: REFERENCE_PROMPTS.get(name).window for name in REFERENCE_PROMPTS.names()}
    assert windows == {
        "fixed_window": "fixed",
        "next_event_window": "next_event",
        "visual_condition": "next_event",
    }


@pytest.mark.parametrize("name", REFERENCE_PROMPTS.names())
def test_every_way_of_asking_renders_from_the_same_context(name):
    rendered = REFERENCE_PROMPTS.create(name).render(CONTEXT)
    assert "figure" in rendered and "Mitochondrion" in rendered
    assert "{" not in rendered.split("Output ONLY")[0]


def test_a_context_with_an_empty_window_of_speech_is_refused():
    with pytest.raises(ContractError):
        REFERENCE_PROMPTS.create("fixed_window").render({**CONTEXT, "transcript_window": ""})


def test_a_silent_window_is_accepted_because_it_says_so():
    REFERENCE_PROMPTS.create("fixed_window").render({**CONTEXT, "transcript_window": NO_SPEECH})


def test_a_decision_is_read_from_the_reply():
    way = REFERENCE_PROMPTS.create("fixed_window")
    decision = way.parse(
        '{"coverage": "UNCOVERED", "covering_quote": null, "uncovered_content": "two curves",'
        ' "emit": true, "ad_text": "Two curves cross.", "rung": 3, "rationale": "deixis"}'
    )
    assert decision.coverage == "UNCOVERED" and decision.describes
    assert decision.rung == 3 and decision.rationale == "deixis"


def test_a_reply_with_no_coverage_or_no_decision_is_not_read():
    way = REFERENCE_PROMPTS.create("fixed_window")
    assert way.parse('{"emit": true, "ad_text": "x"}') is None
    assert way.parse('{"coverage": "MOSTLY", "emit": true}') is None
    assert way.parse('{"coverage": "UNCOVERED"}') is None
    assert way.parse("no object here") is None
    assert way.parse(None) is None


def test_the_three_answers_about_coverage():
    assert COVERAGE == ("FULLY_COVERED", "PARTLY_COVERED", "UNCOVERED")


def test_a_claim_that_the_lecturer_said_it_must_quote_the_window():
    way = REFERENCE_PROMPTS.create("fixed_window")
    grounded = way.parse(
        '{"coverage": "FULLY_COVERED", "covering_quote": "energy of the cell", "emit": false}'
    )
    assert way.violations(grounded, CONTEXT) == []
    invented = way.parse(
        '{"coverage": "FULLY_COVERED", "covering_quote": "the cell makes energy", "emit": false}'
    )
    faults = way.violations(invented, CONTEXT)
    assert len(faults) == 1 and "covering_quote" in faults[0]


def test_the_way_that_asks_about_form_requires_an_answer():
    way = REFERENCE_PROMPTS.create("visual_condition")
    decision = way.parse(
        '{"coverage": "UNCOVERED", "emit": true, "ad_text": "Two curves.", "rung": 3,'
        ' "content": "VISUAL", "unspoken_form": "shape_trend"}'
    )
    assert decision.content == "VISUAL" and decision.unspoken_form == "shape_trend"
    assert way.violations(decision, CONTEXT) == []
    with pytest.raises(ContractError):
        way.parse('{"coverage": "UNCOVERED", "emit": true, "ad_text": "x"}')


def test_a_kind_of_content_or_a_form_outside_its_set_is_a_violation():
    way = REFERENCE_PROMPTS.create("visual_condition")
    decision = way.parse(
        '{"coverage": "UNCOVERED", "emit": true, "content": "PICTURE", "unspoken_form": "size"}'
    )
    faults = way.violations(decision, CONTEXT)
    assert len(faults) == 2


# ---------------------------------------------------------------- rows
def test_an_answer_becomes_a_row_carrying_the_context_it_was_given():
    decision = ReferenceDecision("UNCOVERED", True, "Two curves cross.", 3, "deixis")
    row = to_row(
        WriterAnswer("lec-a#0001", "writer-x", decision),
        CONTEXT,
        window="next_event",
        time=12.0,
    )
    assert row["output_id"] == "lec-a#0001::writer-x" and row["family"] == "writer-x"
    assert row["ad_text"] == "Two curves cross." and row["emit"] is True
    assert row["slide_ocr"] == CONTEXT["slide_ocr"]
    assert row["transcript_window"] == CONTEXT["transcript_window"]
    assert row["emit_window"] == "next_event"
    assert row["time"] == 12.0


def test_a_row_says_which_window_of_speech_produced_it():
    decision = ReferenceDecision("UNCOVERED", True, "x", 3)
    fixed = to_row(WriterAnswer("lec-a#0001", "w", decision), CONTEXT, window="fixed")
    later = to_row(WriterAnswer("lec-a#0001", "w", decision), CONTEXT, window="next_event")
    assert fixed["emit_window"] != later["emit_window"]


def test_a_failed_call_becomes_a_row_that_carries_no_answer():
    row = to_row(
        WriterAnswer("lec-a#0001", "writer-x", None, error="timeout", raw="..."),
        CONTEXT,
        window="fixed",
    )
    assert row["error"] == "timeout" and "emit" not in row


def test_violations_are_recorded_on_the_row():
    decision = ReferenceDecision("FULLY_COVERED", False, covering_quote="invented")
    answer = WriterAnswer("lec-a#0001", "w", decision, violations=("not a quotation",))
    assert to_row(answer, CONTEXT, window="fixed")["violations"] == ["not a quotation"]


# ---------------------------------------------------------------- complete moments
def rows_for(moment: str, writers) -> list[dict]:
    return [
        {"output_id": f"{moment}::{writer}", "family": writer, "emit": True, "ad_text": "x"}
        for writer in writers
    ]


def test_which_writers_answered_each_moment():
    rows = rows_for("lec-a#0001", ["a", "b"]) + rows_for("lec-a#0002", ["a"])
    assert writers_per_moment(rows) == {"lec-a#0001": {"a", "b"}, "lec-a#0002": {"a"}}


def test_a_failed_call_does_not_count_as_an_answer():
    rows = rows_for("lec-a#0001", ["a"]) + [
        {"output_id": "lec-a#0001::b", "family": "b", "error": "timeout"}
    ]
    assert writers_per_moment(rows) == {"lec-a#0001": {"a"}}


def test_a_moment_a_writer_is_missing_from_is_named():
    rows = rows_for("lec-a#0001", ["a", "b"]) + rows_for("lec-a#0002", ["a"])
    assert incomplete_moments(rows, ["a", "b"]) == {"lec-a#0002": ["b"]}


def test_the_incomplete_moments_are_left_out_and_recorded():
    rows = rows_for("lec-a#0001", ["a", "b"]) + rows_for("lec-a#0002", ["a"])
    kept, left_out = complete_only(rows, ["a", "b"])
    assert {row["output_id"] for row in kept} == {"lec-a#0001::a", "lec-a#0001::b"}
    assert left_out == {"lec-a#0002": ["b"]}


def test_nothing_is_left_out_when_every_moment_is_complete():
    rows = rows_for("lec-a#0001", ["a", "b"]) + rows_for("lec-a#0002", ["a", "b"])
    kept, left_out = complete_only(rows, ["a", "b"])
    assert len(kept) == 4 and left_out == {}


# ---------------------------------------------------------------- the generator
def test_every_writer_answers_every_moment_and_each_keeps_its_own_memory():
    asked = []

    def ask(writer, text, stills):
        asked.append((writer, text))
        return WriterAnswer(
            "lec-a#0001",
            writer,
            ReferenceDecision("UNCOVERED", True, "Folded cristae.", 3),
        )

    generator = ReferenceGenerator(
        REFERENCE_PROMPTS.create("fixed_window"), ["writer-x", "writer-y"], ask
    )
    moments = [{"id": "lec-a#0001", "t": 10.0}, {"id": "lec-a#0002", "t": 20.0}]
    rows = generator.write_lecture(
        "lec-a",
        moments,
        context_of=lambda moment, described: {
            **CONTEXT, "described_so_far": described
        },
    )
    assert len(rows) == 4
    assert [writer for writer, _text in asked] == ["writer-x", "writer-y"] * 2
    # Each writer's second prompt names what that writer described, not the other's.
    second_x = asked[2][1]
    assert "cristae" in second_x


def test_a_generator_needs_a_writer():
    with pytest.raises(ContractError):
        ReferenceGenerator(REFERENCE_PROMPTS.create("fixed_window"), [], lambda *_: None)


def test_a_failed_call_does_not_stop_the_lecture():
    calls = {"n": 0}

    def ask(writer, text, stills):
        calls["n"] += 1
        if calls["n"] == 1:
            return WriterAnswer("lec-a#0001", writer, None, error="timeout")
        return WriterAnswer(
            "lec-a#0001", writer, ReferenceDecision("UNCOVERED", True, "Two curves.", 3)
        )

    generator = ReferenceGenerator(REFERENCE_PROMPTS.create("fixed_window"), ["w"], ask)
    rows = generator.write_lecture(
        "lec-a",
        [{"id": "lec-a#0001", "t": 1.0}, {"id": "lec-a#0002", "t": 2.0}],
        context_of=lambda moment, described: {**CONTEXT, "described_so_far": described},
    )
    assert rows[0]["error"] == "timeout" and rows[1]["emit"] is True


# ---------------------------------------------------------------- the settings
def test_the_shipped_reference_settings(shipped_configs):
    settings = load_reference_settings(shipped_configs / "references.yaml")
    assert settings.prompt.name == "next_event_window"
    assert settings.writers == (
        "gemini-3.1-pro-or",
        "gpt-5.5",
        "claude-sonnet-4.6",
        "qwen3-vl-235b",
    )
    assert settings.temperature == 0.2
    assert settings.rung_policy.name == "uniform_budget"
    assert settings.rung_policy.params == {"words": 25}
    assert build_prompt(settings).window == "next_event"


def test_the_writers_of_the_settings_are_models_that_may_write(shipped_configs):
    from ad_for_edu.llm import ModelCatalog

    settings = load_reference_settings(shipped_configs / "references.yaml")
    catalog = ModelCatalog.load(shipped_configs / "models.yaml")
    for writer in settings.writers:
        catalog.get(writer).require_role("writer")


def test_a_misspelled_reference_key_is_refused(tmp_path):
    path = tmp_path / "references.yaml"
    path.write_text("prompt: fixed_window\nwriters: [a]\ntemperatures: 0.2\n", encoding="utf-8")
    with pytest.raises(SettingsError):
        load_reference_settings(path)


def test_the_generator_is_built_from_the_settings(shipped_configs):
    settings = load_reference_settings(shipped_configs / "references.yaml")
    generator = build_generator(lambda *_: None, settings)
    assert generator.writers == settings.writers
