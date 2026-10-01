"""Prompt templates, the shared prefix, and the contracts of inputs and replies."""

from __future__ import annotations

import pytest

from ad_for_edu import prompts
from ad_for_edu.core.errors import ContractError, MissingSourceError
from ad_for_edu.core.text import NO_SPEECH
from ad_for_edu.prompts.contracts import (
    CANDIDATES,
    CONTROLLED_PAIR,
    FOCUS_DIRECTIVES,
    REFERENCE,
    REFERENCE_VISUAL,
)
from ad_for_edu.prompts.templates import PromptTemplate, read_data, read_text
from ad_for_edu.standard import RuleBook

MOMENT = {
    "type": "figure",
    "transcript_window": "so the energy of the cell is produced in this organelle",
    "slide_ocr": "Mitochondrion: cristae, matrix",
    "pause_after": 1.5,
    "reachable_rung": 3,
    "described_so_far": "(none yet)",
}


# ---------------------------------------------------------------- catalogue
def test_the_catalogue_is_consistent():
    prompts.check_catalogue()


def test_the_prompts_of_the_pipeline():
    assert prompts.names() == (
        "classify_moments",
        "reference_fixed_window",
        "reference_next_event_window",
        "reference_visual_condition",
        "candidates_per_call",
        "candidates_verbalized_sampling",
        "controlled_pair",
        "judge_video_pairwise",
        "judge_text_pairwise",
        "judge_sheet_pairwise",
        "judge_rubric",
        "judge_reference_rating",
        "judge_reference_rating_pooled",
        "student_system",
    )


def test_an_unknown_prompt_lists_the_known_ones():
    with pytest.raises(ContractError) as caught:
        prompts.template("reference")
    assert "reference_fixed_window" in str(caught.value)


def test_a_missing_resource_is_an_error():
    with pytest.raises(MissingSourceError):
        read_text("no_such_prompt")


@pytest.mark.parametrize("name", prompts.names())
def test_every_prompt_binds_with_exactly_its_placeholders(name):
    template = prompts.template(name)
    values = {placeholder: f"<{placeholder}>" for placeholder in template.placeholders}
    bound = template.bind(values)
    for placeholder in template.placeholders:
        assert f"<{placeholder}>" in bound
        assert "{" + placeholder + "}" not in bound


@pytest.mark.parametrize("name", [n for n in prompts.names() if prompts.template(n).placeholders])
def test_a_prompt_is_never_sent_half_bound(name):
    template = prompts.template(name)
    values = {placeholder: "x" for placeholder in template.placeholders[1:]}
    with pytest.raises(ContractError) as caught:
        template.bind(values)
    assert template.placeholders[0] in str(caught.value)


@pytest.mark.parametrize("name", prompts.names())
def test_a_value_without_a_placeholder_is_refused(name):
    template = prompts.template(name)
    values = {placeholder: "x" for placeholder in template.placeholders}
    values["not_a_placeholder_of_any_prompt"] = "x"
    with pytest.raises(ContractError):
        template.bind(values)


def test_literal_braces_of_a_reply_format_survive_binding():
    bound = prompts.template("reference_fixed_window").bind(MOMENT)
    assert '{"coverage": "FULLY_COVERED"|"PARTLY_COVERED"|"UNCOVERED"' in bound
    assert "{{" not in bound and "}}" not in bound


def test_a_bound_value_may_itself_contain_braces():
    values = dict(MOMENT, slide_ocr="f(x) = {x | x > 0}")
    assert "f(x) = {x | x > 0}" in prompts.template("reference_fixed_window").bind(values)


def test_the_system_prompt_is_sent_as_it_is():
    template = prompts.template("student_system")
    assert template.placeholders == ()
    assert template.bind() == template.text
    assert '{"emit": <true|false>' in template.text


def test_a_replace_template_must_contain_its_tokens():
    with pytest.raises(ContractError):
        PromptTemplate("broken", "no token here", style="replace", tokens=("types",))


def test_a_format_template_does_not_declare_tokens():
    with pytest.raises(ContractError):
        PromptTemplate("broken", "{a}", style="format", tokens=("a",))


def test_an_unknown_style_is_refused():
    with pytest.raises(ContractError):
        PromptTemplate("broken", "{a}", style="jinja")


# ---------------------------------------------------------------- the two windows
def test_the_two_reference_prompts_differ_in_the_window_they_state():
    fixed = read_text("reference_fixed_window")
    next_event = read_text("reference_next_event_window")
    assert "-5 s / +8 s" in fixed
    assert "-5 s / +8 s" not in next_event
    assert "45" in next_event
    assert fixed != next_event


def test_the_visual_condition_prompt_names_its_closed_sets():
    text = read_text("reference_visual_condition")
    for value in ("NONE", "TEXT_ONLY", "VISUAL", "which_element", "colour_code"):
        assert value in text


# ---------------------------------------------------------------- resources with structure
def test_the_classifier_types_are_the_moment_types_plus_reject():
    from ad_for_edu.data.schema import MOMENT_TYPES, REJECT

    data = read_data("classify_types")
    assert tuple(data["order"]) == (*MOMENT_TYPES, REJECT)
    assert set(data["types"]) == set(data["order"])


def test_five_focus_directives():
    assert tuple(read_data("focus_directives")) == FOCUS_DIRECTIVES


# ---------------------------------------------------------------- prefix
def test_the_prefix_carries_the_rule_book_verbatim_and_counts_its_rules():
    book = RuleBook.load()
    prefix = prompts.shared_prefix(book)
    assert book.text in prefix
    assert "45 rules, verbatim" in prefix
    assert prefix.index("45 rules, verbatim") < prefix.index("OPERATING PRINCIPLES")
    assert prefix.count("OPERATING PRINCIPLES") == 1


def test_the_prefix_is_the_same_on_every_call():
    assert prompts.shared_prefix() == prompts.shared_prefix()


def test_the_prefix_reports_the_number_of_rules_it_carries():
    small = RuleBook(
        "rules:\n  - id: a_001\n    tag: DOC\n    rule: 'x'\n    scoring: {route: judge}\n"
    )
    assert "1 rules, verbatim" in prompts.shared_prefix(small)


def test_the_classifier_does_not_receive_the_prefix():
    assert not prompts.entry("classify_moments").prefixed
    assert prompts.entry("reference_fixed_window").prefixed
    assert prompts.entry("judge_video_pairwise").prefixed


# ---------------------------------------------------------------- input contracts
def test_a_complete_moment_satisfies_the_contract():
    prompts.check_inputs(REFERENCE, MOMENT)


def test_an_empty_transcript_window_is_refused():
    with pytest.raises(ContractError) as caught:
        prompts.check_inputs(REFERENCE, dict(MOMENT, transcript_window="  "))
    assert "transcript_window: empty" in str(caught.value)


def test_a_silent_window_is_stated_and_accepted():
    prompts.check_inputs(REFERENCE, dict(MOMENT, transcript_window=NO_SPEECH))


def test_a_missing_input_is_named():
    values = {key: value for key, value in MOMENT.items() if key != "pause_after"}
    with pytest.raises(ContractError) as caught:
        prompts.check_inputs(REFERENCE, values)
    assert "pause_after: missing" in str(caught.value)


def test_slide_text_may_be_empty():
    prompts.check_inputs(REFERENCE, dict(MOMENT, slide_ocr="", described_so_far=""))


@pytest.mark.parametrize("rung", [0, 5, "three"])
def test_a_rung_outside_the_ladder_is_refused(rung):
    with pytest.raises(ContractError):
        prompts.check_inputs(REFERENCE, dict(MOMENT, reachable_rung=rung))


def test_an_unknown_directive_band_or_axis_is_refused():
    candidate = dict(MOMENT, directive="REFERENT", band="M", band_min=16, band_max=25)
    prompts.check_inputs(CANDIDATES, candidate)
    with pytest.raises(ContractError):
        prompts.check_inputs(CANDIDATES, dict(candidate, directive="COLOUR"))
    with pytest.raises(ContractError):
        prompts.check_inputs(CANDIDATES, dict(candidate, band="XXL"))
    prompts.check_inputs(CONTROLLED_PAIR, dict(MOMENT, axis="non_redundancy"))
    with pytest.raises(ContractError):
        prompts.check_inputs(CONTROLLED_PAIR, dict(MOMENT, axis="redundancy"))


# ---------------------------------------------------------------- reply contracts
def test_a_verbatim_quotation_is_accepted():
    reply = {"emit": False, "coverage": "FULLY_COVERED", "covering_quote": "energy of the cell"}
    assert prompts.check_output(REFERENCE, reply, MOMENT) == []


def test_a_quotation_may_be_rewrapped_and_recased():
    reply = {"emit": False, "coverage": "FULLY_COVERED", "covering_quote": "Energy  of the\nCELL"}
    assert prompts.check_output(REFERENCE, reply, MOMENT) == []


def test_a_paraphrase_is_reported():
    reply = {"emit": False, "coverage": "FULLY_COVERED", "covering_quote": "the cell makes energy"}
    violations = prompts.check_output(REFERENCE, reply, MOMENT)
    assert len(violations) == 1 and "covering_quote" in violations[0]


def test_an_elided_quotation_is_accepted_in_order_only():
    source = "so the energy of the cell is produced in this organelle"
    assert prompts.quote_is_grounded("the energy ... in this organelle", source)
    assert prompts.quote_is_grounded("the energy … this organelle", source)
    assert not prompts.quote_is_grounded("in this organelle ... the energy", source)
    assert not prompts.quote_is_grounded("the energy ... in that organelle", source)


def test_an_empty_quotation_is_not_grounded():
    assert not prompts.quote_is_grounded("", "anything")
    assert not prompts.quote_is_grounded("...", "anything")


def test_a_quotation_against_a_silent_window_is_reported():
    reply = {"emit": False, "coverage": "FULLY_COVERED", "covering_quote": "energy of the cell"}
    violations = prompts.check_output(REFERENCE, reply, dict(MOMENT, transcript_window=NO_SPEECH))
    assert violations


def test_a_missing_reply_field_is_reported():
    assert prompts.check_output(REFERENCE, {"emit": True}, MOMENT) == [
        "missing output field 'coverage'"
    ]


def test_a_reply_value_outside_its_closed_set_is_reported():
    reply = {"emit": True, "coverage": "UNCOVERED", "content": "PICTURE", "unspoken_form": "size"}
    violations = prompts.check_output(REFERENCE_VISUAL, reply, MOMENT)
    assert len(violations) == 2
    assert any("content" in v for v in violations) and any("unspoken_form" in v for v in violations)


def test_a_null_in_a_closed_field_is_accepted():
    reply = {"emit": False, "coverage": "UNCOVERED", "content": "NONE", "unspoken_form": None}
    assert prompts.check_output(REFERENCE_VISUAL, reply, MOMENT) == []
