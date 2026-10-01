"""Reading replies: a clean verdict, or nothing."""

from __future__ import annotations

import pytest

from ad_for_edu.llm.replies import (
    RAW_CAP,
    extract_json,
    keep_raw,
    parse_choice,
    parse_scale,
    parse_verdict,
    strip_fences,
)


# ---------------------------------------------------------------- choices
@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("1", "1"),
        ("2", "2"),
        ("TIE", "tie"),
        ("tie", "tie"),
        (" **2** ", "2"),
        ('"1".', "1"),
        ("`TIE`!", "tie"),
    ],
)
def test_a_bare_verdict_is_read(reply, expected):
    assert parse_choice(reply) == expected


@pytest.mark.parametrize(
    "reply",
    [
        "The quantities differ between the two.",
        "Both list the properties of the cell.",
        "Their identities are swapped.",
    ],
)
def test_words_that_contain_tie_do_not_vote(reply):
    assert parse_choice(reply) is None


def test_the_word_tie_alone_is_a_tie():
    assert parse_choice("It is a tie between them.") == "tie"


def test_a_tie_is_recorded_as_the_same_value_everywhere():
    from ad_for_edu.data.schema import TIE as RECORDED
    from ad_for_edu.llm.replies import TIE as ANSWERED

    assert ANSWERED == RECORDED


@pytest.mark.parametrize(
    "reply",
    [
        "Description 1 is better than Description 2.",
        "Description 1 is worse than Description 2.",
        "Option 1 fails; option 2 succeeds.",
        "Between 1 and 2, I choose 2.",
        "1 or 2? A tie.",
    ],
)
def test_prose_that_names_both_options_is_refused(reply):
    assert parse_choice(reply) is None


def test_a_short_reply_naming_one_option_is_read():
    assert parse_choice("Description 2.") == "2"
    assert parse_choice("I choose option 1") == "1"


def test_a_long_reply_naming_one_option_is_refused():
    reply = "After weighing the wording and the timing of both, the better one here is 2"
    assert len(reply.split()) > 8
    assert parse_choice(reply) is None


@pytest.mark.parametrize("reply", [None, "", "   ", "3", "neither"])
def test_no_verdict_is_none_and_never_a_tie(reply):
    assert parse_choice(reply) is None


# ---------------------------------------------------------------- scales
@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("4", 4),
        ("4 out of 5", 4),
        ("3/5", 3),
        ("I rate this 2 on the 1-5 scale.", 2),
        ("Score: 5", 5),
        ("2 on a scale from 1 to 5", 2),
    ],
)
def test_the_bound_of_the_scale_is_not_the_rating(reply, expected):
    assert parse_scale(reply, 1, 5) == expected


def test_a_rating_on_a_scale_from_zero_to_ten():
    assert parse_scale("Rating on the 0-10 scale: 7", 0, 10) == 7
    assert parse_scale("7/10", 0, 10) == 7


@pytest.mark.parametrize("reply", ["between 2 and 4", "3 or 4", "2, then 3"])
def test_two_ratings_in_range_are_refused(reply):
    assert parse_scale(reply, 1, 5) is None


@pytest.mark.parametrize("reply", [None, "", "good", "9", "0", "-3"])
def test_no_rating_in_range_is_none_and_never_a_bound(reply):
    assert parse_scale(reply, 1, 5) is None


def test_the_same_rating_stated_twice_is_one_rating():
    assert parse_scale("4. Final answer: 4", 1, 5) == 4


# ---------------------------------------------------------------- structured replies
def test_a_plain_object_is_read():
    assert extract_json('{"emit": true, "rung": 3}') == {"emit": True, "rung": 3}


def test_a_preamble_and_a_fence_do_not_hide_the_payload():
    reply = 'I need to look at this carefully.\n```json\n[{"ad_text": "Two curves."}]\n```\nDone.'
    assert extract_json(reply) == [{"ad_text": "Two curves."}]


def test_brackets_in_the_preamble_do_not_defeat_the_search():
    reply = 'The prefix [after] applies to {this shape}.\n{"ad_text": "A cube.", "rung": 3}'
    assert extract_json(reply) == {"ad_text": "A cube.", "rung": 3}


def test_a_brace_inside_a_description_does_not_close_the_object():
    reply = '{"ad_text": "The set {x | x > 0} in red.", "rung": 1}'
    assert extract_json(reply)["ad_text"] == "The set {x | x > 0} in red."


def test_an_escaped_quote_inside_a_description_is_handled():
    reply = '{"ad_text": "The label reads \\"mass\\" in bold.", "rung": 1}'
    assert extract_json(reply)["ad_text"] == 'The label reads "mass" in bold.'


def test_the_largest_structure_wins_over_a_quoted_fragment():
    reply = 'For example {"a": 1}. The answer: {"ad_text": "A long description.", "rung": 2}'
    assert extract_json(reply) == {"ad_text": "A long description.", "rung": 2}


@pytest.mark.parametrize("reply", [None, "", "no structure here", '{"cut": "off', "]["])
def test_no_complete_structure_is_none(reply):
    assert extract_json(reply) is None


# ---------------------------------------------------------------- verdicts
def test_a_verdict_is_read_with_its_reasoning():
    reply = '```json\n{"verdict": "pass", "reasoning": "names the element"}\n```'
    assert parse_verdict(reply) == ("PASS", "names the element")
    assert parse_verdict('{"verdict": "FAIL"}') == ("FAIL", "")


@pytest.mark.parametrize(
    "reply", [None, "", "PASS", '{"verdict": "MAYBE"}', '{"result": "PASS"}', "[1, 2]"]
)
def test_anything_else_is_no_verdict(reply):
    assert parse_verdict(reply) == (None, None)


def test_fences_are_removed_from_both_ends():
    assert strip_fences("```json\n{}\n```") == "{}"
    assert strip_fences("{}") == "{}"


# ---------------------------------------------------------------- keeping the raw reply
def test_a_failed_reply_is_kept_for_diagnosis():
    record = keep_raw({"winner": None}, "I cannot decide between them because ...")
    assert record["raw"].startswith("I cannot decide")
    assert "raw_truncated" not in record


def test_a_long_reply_is_cut_and_marked():
    record = keep_raw({}, "x" * (RAW_CAP + 10))
    assert len(record["raw"]) == RAW_CAP and record["raw_truncated"] is True


def test_an_absent_reply_is_kept_as_empty():
    assert keep_raw({}, None)["raw"] == ""
