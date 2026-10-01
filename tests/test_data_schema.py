"""Row classes: required fields, closed sets, fields carried through."""

from __future__ import annotations

import pytest

from ad_for_edu.core.errors import ContractError
from ad_for_edu.data import schema


def test_a_reference_row_knows_its_moment_lecture_and_writer():
    row = schema.ReferenceRow.from_row(
        {
            "output_id": "lec-a#0017::writer-x",
            "type": "figure",
            "emit": True,
            "ad_text": "Two curves cross at the threshold.",
            "family": "writer-x",
        }
    )
    assert row.moment_id == "lec-a#0017"
    assert row.lecture == "lec-a"
    assert row.writer == "writer-x"
    assert row.describes


def test_silence_and_empty_text_are_not_descriptions():
    silent = schema.ReferenceRow(output_id="lec-a#0001::w", type="slide", emit=False)
    blank = schema.ReferenceRow(output_id="lec-a#0001::w", type="slide", emit=True, ad_text="  ")
    assert not silent.describes and not blank.describes


def test_unknown_fields_are_carried_through_unchanged():
    source = {
        "output_id": "lec-a#0017::writer-x",
        "type": "figure",
        "emit": False,
        "coverage": "PARTLY",
        "n_families_attempted": 3,
    }
    row = schema.ReferenceRow.from_row(source)
    assert row.extra == {"coverage": "PARTLY", "n_families_attempted": 3}
    written = row.to_row()
    assert written["coverage"] == "PARTLY" and written["output_id"] == source["output_id"]


def test_a_missing_required_field_is_named():
    with pytest.raises(ContractError) as caught:
        schema.ReferenceRow.from_row({"output_id": "lec-a#0017::w", "type": "figure"})
    assert "emit" in str(caught.value)


def test_a_moment_type_outside_the_closed_set_is_refused():
    with pytest.raises(ContractError) as caught:
        schema.Moment(id="lec-a#0001", lecture="lec-a", t=1.0, type="gesture")
    assert "gesture" in str(caught.value)


def test_there_are_twelve_moment_types_and_six_categories():
    assert len(schema.MOMENT_TYPES) == 12 and len(set(schema.MOMENT_TYPES)) == 12
    assert schema.CATEGORIES == (
        "style",
        "terminology",
        "length",
        "deixis",
        "faithfulness",
        "non_redundancy",
    )
    assert schema.REJECT not in schema.MOMENT_TYPES


def test_a_pair_axis_must_be_a_category_or_absent():
    natural = schema.PreferencePair(pair_id="p1", moment_id="lec-a#0001", a="x", b="y", source="p3")
    assert not natural.controlled
    controlled = schema.PreferencePair(
        pair_id="p2", moment_id="lec-a#0001", a="x", b="y", source="p4", axis="non_redundancy"
    )
    assert controlled.controlled
    with pytest.raises(ContractError):
        schema.PreferencePair(
            pair_id="p3", moment_id="lec-a#0001", a="x", b="y", source="p4", axis="redundancy"
        )


def test_a_rater_label_is_a_side_or_a_tie():
    assert schema.RaterLabel(pair_id="p1", rater_id="R1", side="a").decided
    assert not schema.RaterLabel(pair_id="p1", rater_id="R1", side="tie").decided
    with pytest.raises(ContractError):
        schema.RaterLabel(pair_id="p1", rater_id="R1", side="both")


def test_a_failed_verdict_is_recognised():
    failed = schema.Verdict.from_row({"pair_id": "p1", "order": 0, "error": "timeout"})
    answered = schema.Verdict.from_row({"pair_id": "p1", "order": 1, "winner": "a"})
    assert failed.failed and failed.winner is None
    assert not answered.failed


def test_a_lecture_belongs_to_one_side_of_the_split():
    entry = schema.LectureEntry.from_row(
        {"lecture_id": "lec-a", "video": "lec-a.mp4", "course": "biology", "split": "test"}
    )
    assert entry.split == "test" and entry.renders_cursor is None
    with pytest.raises(ContractError):
        schema.LectureEntry(lecture_id="lec-a", video="v", course="c", split="dev")


@pytest.mark.parametrize("value", [None, "", "   ", "None", " None "])
def test_an_unfilled_description_is_recognised(value):
    assert schema.is_unfilled(value)
    assert not schema.Prediction(
        output_id="lec-a#0001", moment_id="lec-a#0001", emit=True, ad_text=value
    ).filled


def test_a_description_about_nothing_is_still_filled():
    assert not schema.is_unfilled("None of the bars exceeds ten.")
