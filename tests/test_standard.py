"""The rule book, its applicability table and the per-rule judge prompts."""

from __future__ import annotations

import pytest

from ad_for_edu.core.errors import ContractError
from ad_for_edu.data.schema import MOMENT_TYPES
from ad_for_edu.standard import (
    WITHOUT_PROMPT,
    ApplicabilityTable,
    RuleBook,
    bind_moment,
    check_standard,
    render_all,
    render_rule_prompt,
)
from ad_for_edu.standard.templates import MOMENT_TOKENS, RULE_TOKENS, load_template

BOOK_WITH_A_DANGLING_ID = """
version: 9.9
rules:
  - id: deixis_001
    tag: DOC
    rule: "Name the element. See deixis_002 and inked_math_003."
    scoring: {route: judge, checks: []}
  - id: deixis_002
    tag: PROV
    rule: "Stay silent when the element cannot be named (deixis_001..007)."
    scoring: {route: unscored, checks: []}
"""


@pytest.fixture(scope="module")
def book() -> RuleBook:
    return RuleBook.load()


@pytest.fixture(scope="module")
def table() -> ApplicabilityTable:
    return ApplicabilityTable.load()


# ---------------------------------------------------------------- the shipped book
def test_the_book_has_forty_five_rules(book):
    assert len(book) == 45
    assert len(set(book.ids)) == 45


def test_documented_and_provisional_rules(book):
    assert book.tags() == {"DOC": 24, "PROV": 21}


def test_scoring_routes(book):
    assert dict(book.routes()) == {"mechanical": 14, "judge": 17, "unscored": 12, "penalty": 2}


def test_thirteen_rules_carry_a_review(book):
    assert len(book.reviewed()) == 13


def test_the_shipped_book_is_consistent(book):
    book.check()
    assert book.dangling_references() == []


def test_the_summary_reports_the_counts():
    summary = check_standard()
    assert summary["rules"] == 45
    assert summary["documented"] == 24 and summary["provisional"] == 21
    assert summary["routes"] == {"mechanical": 14, "judge": 17, "penalty": 2, "unscored": 12}
    assert summary["judge_prompts"] == 42


def test_the_text_sent_to_the_models_carries_no_comments(book):
    for line in book.text.splitlines():
        assert not line.lstrip().startswith("#"), line


def test_only_a_model_under_study_may_be_named_in_the_book(book):
    lowered = book.text.lower()
    for word in ("todo", "fixme"):
        assert word not in lowered


# ---------------------------------------------------------------- the integrity check fires
def test_an_id_of_a_family_that_does_not_exist_is_found():
    dangling = RuleBook(BOOK_WITH_A_DANGLING_ID).dangling_references()
    assert "deixis_001.rule -> inked_math_003" in dangling


def test_a_range_that_runs_past_the_last_rule_is_found():
    dangling = RuleBook(BOOK_WITH_A_DANGLING_ID).dangling_references()
    assert any("deixis_007" in entry and "range" in entry for entry in dangling)


def test_an_existing_id_is_not_reported():
    dangling = RuleBook(BOOK_WITH_A_DANGLING_ID).dangling_references()
    assert not any(entry.endswith("-> deixis_002") for entry in dangling)


def test_check_raises_on_a_dangling_id():
    with pytest.raises(ContractError) as caught:
        RuleBook(BOOK_WITH_A_DANGLING_ID).check()
    assert "inked_math_003" in str(caught.value)


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ("tag: DOC", "tag"),
        ("route: judge", "route"),
    ],
)
def test_check_raises_on_a_value_outside_a_closed_set(change, expected):
    text = BOOK_WITH_A_DANGLING_ID.replace(" See deixis_002 and inked_math_003.", "")
    text = text.replace(" (deixis_001..007)", "")
    RuleBook(text).check()
    broken = text.replace(change, change.split(":")[0] + ": SOMETHING", 1)
    with pytest.raises(ContractError) as caught:
        RuleBook(broken).check()
    assert expected in str(caught.value)


def test_check_raises_on_a_repeated_id():
    text = BOOK_WITH_A_DANGLING_ID.replace("id: deixis_002", "id: deixis_001")
    with pytest.raises(ContractError) as caught:
        RuleBook(text).check()
    assert "more than once" in str(caught.value)


def test_a_file_without_rules_is_not_a_rule_book():
    with pytest.raises(ContractError):
        RuleBook("version: 1\n")


def test_an_unknown_rule_is_an_error(book):
    with pytest.raises(ContractError):
        book.get("deixis_099")


# ---------------------------------------------------------------- applicability
def test_table_and_book_describe_the_same_rules(book, table):
    assert len(table) == 45
    table.check_against(book)


def test_published_numbers_run_from_one_to_forty_five(table):
    assert sorted(table.numbers().values()) == list(range(1, 46))


def test_the_published_number_is_not_the_position_in_the_book(book, table):
    numbers = table.numbers()
    positions = {rule_id: index for index, rule_id in enumerate(book.ids, start=1)}
    moved = [rule_id for rule_id in book.ids if numbers[rule_id] != positions[rule_id]]
    assert moved, "numbers and positions agree everywhere; the distinction would be untested"
    assert numbers["visual_content_001"] > 42


def test_a_rule_for_charts_applies_to_charts_only(table):
    entry = table["chart_001"]
    assert entry.applies("chart", described=True)
    assert not entry.applies("table", described=True)
    assert not entry.applies(None, described=True)


def test_a_general_rule_applies_to_every_type(table):
    entry = table["cross_cutting_002"]
    assert all(entry.applies(kind, described=True) for kind in MOMENT_TYPES)


def test_a_rule_about_wording_does_not_apply_to_silence(table):
    assert not table["cross_cutting_002"].applies("slide", described=False)


def test_the_redundancy_rule_applies_to_silence_too(table):
    entry = table["cross_cutting_001"]
    assert entry.applies("slide", described=True) and entry.applies("slide", described=False)


def test_a_rule_about_building_pairs_applies_to_no_moment(table):
    entry = table["pair_perceptibility_001"]
    assert not any(
        entry.applies(kind, described) for kind in MOMENT_TYPES for described in (True, False)
    )


def test_applicable_rules_come_in_published_order(table):
    rule_ids = table.applicable("pointing", described=True)
    numbers = [table[rule_id].number for rule_id in rule_ids]
    assert numbers == sorted(numbers)
    assert "deixis_004" in rule_ids and "chart_001" not in rule_ids


def test_applicable_rules_are_read_off_a_row(table):
    described = table.applicable_to({"type": "chart", "emit": True})
    silent = table.applicable_to({"type": "chart", "emit": False})
    assert "chart_001" in described
    assert set(silent) < set(described) | set(silent) and described != silent


def test_a_table_that_disagrees_with_the_book_is_refused(book, table):
    entries = dict(table.entries)
    removed = entries.pop("chart_001")
    with pytest.raises(ContractError) as caught:
        ApplicabilityTable(entries).check_against(book)
    assert "chart_001" in str(caught.value)
    assert removed.rule_id == "chart_001"


def test_an_unknown_rule_is_not_in_the_table(table):
    with pytest.raises(ContractError):
        table["deixis_099"]


# ---------------------------------------------------------------- judge prompts
def test_forty_two_rules_have_a_prompt_and_three_do_not(book):
    rendered = render_all(book)
    assert len(rendered) == 42
    assert set(book.ids) - set(rendered) == WITHOUT_PROMPT
    assert WITHOUT_PROMPT == {
        "visual_content_001",
        "slide_text_003",
        "pair_perceptibility_001",
    }


def test_a_rule_without_a_prompt_cannot_be_rendered(book):
    with pytest.raises(ContractError):
        render_rule_prompt(book.get("visual_content_001"))


def test_rendering_fills_the_rule_and_leaves_the_moment_open(book):
    prompt = render_rule_prompt(book.get("deixis_004"))
    assert "rule_id: deixis_004" in prompt
    assert book.get("deixis_004").text in prompt
    for token in RULE_TOKENS:
        assert "{" + token + "}" not in prompt
    for token in MOMENT_TOKENS:
        assert "{" + token + "}" in prompt


def test_every_prompt_binds_fully(book):
    values = {
        "type": "pointing",
        "what_on_screen": "A cell with its nucleus",
        "transcript_window": "and this is where it is stored",
        "emit": True,
        "ad_text": "The nucleus, at the centre of the cell.",
    }
    for rule_id, prompt in render_all(book).items():
        bound = bind_moment(prompt, values)
        for token in (*RULE_TOKENS, *MOMENT_TOKENS):
            assert "{" + token + "}" not in bound, (rule_id, token)
        assert values["ad_text"] in bound
        assert '{"verdict": "PASS" | "FAIL"' in bound


def test_binding_without_a_sourced_value_is_refused(book):
    prompt = render_rule_prompt(book.get("deixis_004"))
    with pytest.raises(ContractError) as caught:
        bind_moment(prompt, {"type": "pointing", "emit": True, "ad_text": "x"})
    assert "transcript_window" in str(caught.value) and "what_on_screen" in str(caught.value)


def test_binding_an_unknown_value_is_refused(book):
    prompt = render_rule_prompt(book.get("deixis_004"))
    values = {
        "type": "pointing",
        "what_on_screen": "x",
        "transcript_window": "y",
        "emit": True,
        "ad_text": "z",
        "slide_ocr": "not a token of this prompt",
    }
    with pytest.raises(ContractError):
        bind_moment(prompt, values)


def test_a_rule_without_a_citation_says_so():
    book = RuleBook(
        "rules:\n  - id: a_001\n    tag: PROV\n    rule: 'x'\n    scoring: {route: judge}\n"
    )
    assert book.get("a_001").grounding() == "(no external citation; rule is self-grounding)"


def test_the_template_names_every_token():
    template = load_template()
    for token in (*RULE_TOKENS, *MOMENT_TOKENS):
        assert "{" + token + "}" in template
