"""Identifiers, times and text helpers."""

from __future__ import annotations

import pytest

from ad_for_edu.core import ids, text, timecodes
from ad_for_edu.core.errors import ContractError


# ---------------------------------------------------------------- identifiers
def test_a_description_id_reduces_to_its_moment():
    assert ids.moment_of({"output_id": "lec-a#0017::writer-x"}) == "lec-a#0017"
    assert ids.moment_of("lec-a#0017::writer-x") == "lec-a#0017"


def test_a_prediction_and_a_description_resolve_to_the_same_moment():
    described = {"output_id": "lec-a#0017::writer-x"}
    predicted = {"moment_id": "lec-a#0017", "output_id": ""}
    assert ids.moment_of(described) == ids.moment_of(predicted)


def test_a_row_without_either_key_has_no_moment():
    assert ids.moment_of({"pair_id": "p1"}) == ""


def test_lecture_and_writer_are_read_from_the_id():
    assert ids.lecture_of("lec-a#g007") == "lec-a"
    assert ids.writer_of("lec-a#0017::writer-x") == "writer-x"
    assert ids.writer_of("lec-a#0017") is None


def test_moment_ids_are_built_per_channel_kind():
    assert ids.moment_id("lec-a", 17) == "lec-a#0017"
    assert ids.moment_id("lec-a", 7, from_gap=True) == "lec-a#g007"
    assert ids.is_gap_moment("lec-a#g007") and not ids.is_gap_moment("lec-a#0017")


@pytest.mark.parametrize("lecture", ["", "lec#a", "lec::a"])
def test_a_lecture_id_must_not_contain_a_separator(lecture):
    with pytest.raises(ContractError):
        ids.moment_id(lecture, 1)


# ---------------------------------------------------------------- times
def test_clock_times_have_three_fields_and_the_first_is_hours():
    assert timecodes.clock_to_seconds("00:09:39") == 579
    assert timecodes.clock_to_seconds("01:02:03.250") == 3723.25


@pytest.mark.parametrize("value", ["09:39", "abc", "", "00:61:00", "00:00:75", None])
def test_an_unreadable_clock_time_is_not_zero(value):
    assert timecodes.clock_to_seconds(value) == timecodes.UNPARSEABLE
    assert not timecodes.is_time(timecodes.clock_to_seconds(value))


@pytest.mark.parametrize("seconds", [0, 59, 60, 579, 3599, 3600, 3723, 86399])
def test_packed_names_round_trip(seconds):
    name = timecodes.seconds_to_packed(seconds)
    assert name.endswith(".jpg") and len(name) == 10
    assert timecodes.packed_to_seconds(name) == seconds


def test_a_packed_name_is_not_minutes_and_seconds():
    # 000939 is nine minutes thirty-nine seconds, never 939 seconds or 9 hours.
    assert timecodes.packed_to_seconds("000939.jpg") == 579


@pytest.mark.parametrize("name", ["93900.jpg", "0009391.jpg", "frame.jpg", "006100.jpg"])
def test_a_name_of_another_shape_is_not_a_time(name):
    assert timecodes.packed_to_seconds(name) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [(12, 12.0), (12.5, 12.5), ("12.5", 12.5), ("00:00:12", 12.0), (" 7 ", 7.0)],
)
def test_any_accepted_form_is_read(value, expected):
    assert timecodes.as_seconds(value) == expected


@pytest.mark.parametrize("value", [None, "", "soon", -1, "-3", True])
def test_anything_else_is_unparseable(value):
    assert timecodes.as_seconds(value) == timecodes.UNPARSEABLE


def test_clock_round_trip():
    assert timecodes.seconds_to_clock(3723.9) == "01:02:03"
    assert timecodes.clock_to_seconds(timecodes.seconds_to_clock(3723)) == 3723


# ---------------------------------------------------------------- text
def test_the_delivery_prefix_is_not_part_of_the_description():
    assert text.clean_description("[after] A bar chart appears.") == "A bar chart appears."
    assert text.clean_description("  [BEFORE]  Two curves.") == "Two curves."
    assert text.word_count("[after] A bar chart appears.") == 4
    assert text.clean_description(None) == ""


def test_a_marker_is_never_slide_content():
    context = {"slide_ocr": text.NO_SLIDE_TEXT, "what_on_screen": "A labelled diagram"}
    assert text.slide_premise(context) == "A labelled diagram"
    assert text.content_terms(text.NO_SLIDE_TEXT) == []


def test_a_marker_is_removed_inside_a_longer_string():
    joined = f"{text.NO_SLIDE_TEXT} Membrane potential"
    assert "slide" not in text.strip_sentinels(joined).lower()
    assert text.content_terms(joined) == ["Membrane", "potential"]


def test_a_marker_matches_whole_strings_only_in_is_sentinel():
    assert text.is_sentinel("  (No Speech In This Window) ")
    assert not text.is_sentinel(f"{text.NO_SPEECH} and more")


def test_content_terms_are_distinct_ordered_and_capped():
    terms = text.content_terms("Axon axon the Dendrite and soma, soma-body x1 ab", cap=3)
    assert terms == ["Axon", "Dendrite", "soma"]


def test_content_terms_skip_short_tokens_and_function_words():
    assert text.content_terms("the is of an ion for all") == ["ion"]


def test_an_empty_premise_stays_empty():
    assert text.slide_premise(None) == ""
    assert text.slide_premise({"slide_ocr": "", "what_on_screen": None}) == ""


def test_token_sets_and_overlap():
    left = text.token_set("The slide title: Action Potential")
    right = text.token_set("action potential (phase 2)")
    assert left == {"the", "slide", "title", "action", "potential"}
    assert text.jaccard(left, right) == pytest.approx(2 / 7)
    assert text.jaccard(left, frozenset()) == 0.0
    assert text.jaccard(left, left) == 1.0


def test_following_terms_takes_a_bounded_window():
    assert text.following_terms("one two-three x four5 six seven", 3) == "one two-three four5"
