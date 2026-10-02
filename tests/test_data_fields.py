"""Reading a dataset that is not this project's, and the one way that goes wrong quietly.

The failure these tests exist for: the novelty term groups a system's descriptions by
lecture, and the lecture used to come from splitting the moment id. An identifier from
another dataset has nothing to split on, so every moment became its own lecture, nothing
could be seen to repeat, and the factor was 1 everywhere. Every number stayed plausible
and the term had stopped working.

So two things are asserted here: a foreign dataset can be read at all, and the term still
fires on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ad_for_edu.compliance import build_scorer, build_sequence_factor, score_predictions
from ad_for_edu.compliance.sequence import NoveltyKnee, SequenceItem
from ad_for_edu.core.errors import ContractError, SettingsError
from ad_for_edu.data.fields import DEFAULTS, FieldMap, assert_lectures_group

#: A dataset with none of this project's conventions.
FOREIGN = [
    {
        "id": f"{course.lower()}-clip{index:03d}",
        "course": course,
        "start": 12.5 * index,
        "kind": "chart",
        "ocr": "Pressure against volume",
        "screen": "A line chart",
        "said": "and the curve bends here",
        "gap": 3.0,
        "their_own_column": "kept",
    }
    for course in ("PHYS-1", "CHEM-2")
    for index in range(1, 4)
]
MAPPING = [
    "moment_id=id",
    "lecture=course",
    "time=start",
    "type=kind",
    "slide_text=ocr",
    "on_screen=screen",
    "transcript_window=said",
    "pause_after=gap",
    "text=narration",
]


# ------------------------------------------------------------------- the map
def test_with_no_map_the_defaults_are_this_package_s_own_names():
    assert FieldMap().columns == DEFAULTS
    assert FieldMap().renamed == {}


def test_a_map_renames_only_what_it_is_told_to():
    found = FieldMap.parse(["moment_id=id", "lecture=course"])
    assert found.renamed == {"moment_id": "id", "lecture": "course"}
    assert found.column("type") == DEFAULTS["type"]


def test_a_field_this_package_does_not_read_is_refused():
    with pytest.raises(SettingsError, match="does not read a field"):
        FieldMap.parse(["vibes=x"])


def test_a_mapping_without_a_column_is_refused():
    with pytest.raises(SettingsError, match="is not name=column"):
        FieldMap.parse(["moment_id="])


def test_a_foreign_row_comes_back_under_this_package_s_names():
    fields = FieldMap.parse(MAPPING)
    row = fields.canonical(FOREIGN[0])
    assert row["moment_id"] == "phys-1-clip001"
    assert row["lecture"] == "PHYS-1"
    assert row["t"] == pytest.approx(12.5)
    assert row["slide_ocr"] == "Pressure against volume"
    assert row["transcript_window"] == "and the curve bends here"
    assert row["pause_after"] == pytest.approx(3.0)


def test_a_dataset_s_own_columns_survive_the_translation():
    row = FieldMap.parse(MAPPING).canonical(FOREIGN[0])
    assert row["their_own_column"] == "kept"


def test_a_row_with_no_identifier_cannot_be_joined():
    with pytest.raises(ContractError, match="cannot be joined"):
        FieldMap.parse(["moment_id=id"]).canonical({"course": "PHYS-1"})


def test_a_map_can_be_read_from_a_file(tmp_path: Path):
    path = tmp_path / "fields.yaml"
    path.write_text("moment_id: id\nlecture: course\n", encoding="utf-8")
    assert FieldMap.load(path).renamed == {"moment_id": "id", "lecture": "course"}


def test_a_file_naming_an_unknown_field_is_refused(tmp_path: Path):
    path = tmp_path / "fields.yaml"
    path.write_text("vibes: x\n", encoding="utf-8")
    with pytest.raises(SettingsError):
        FieldMap.load(path)


# ------------------------------------------- the lecture has to group something
def test_a_named_lecture_column_groups_the_moments():
    fields = FieldMap.parse(MAPPING)
    found = assert_lectures_group(fields.rows(FOREIGN), fields=fields)
    assert found == {"moments": 6, "lectures": 2, "moments_per_lecture": 3.0}


def test_an_unnamed_lecture_is_refused_rather_than_scored():
    """The failure this guard exists for. Every moment its own lecture, so no repetition."""
    bare = FieldMap.parse(["moment_id=id"])
    with pytest.raises(ContractError, match="its own lecture"):
        assert_lectures_group(bare.rows(FOREIGN), fields=bare)


def test_the_refusal_says_which_column_to_set():
    bare = FieldMap.parse(["moment_id=id"])
    with pytest.raises(ContractError, match="Set the field map's 'lecture'"):
        assert_lectures_group(bare.rows(FOREIGN), fields=bare)


def test_this_package_s_own_identifiers_need_no_map():
    ours = [{"moment_id": f"lec-a#{i:04d}", "t": 10.0 * i} for i in range(1, 5)]
    plain = FieldMap()
    assert assert_lectures_group(plain.rows(ours))["lectures"] == 1


def test_one_moment_is_not_treated_as_a_failure_to_group():
    one = [{"moment_id": "lec-a#0001", "t": 1.0}]
    assert assert_lectures_group(FieldMap().rows(one))["moments"] == 1


def test_nothing_to_group_is_refused():
    with pytest.raises(ContractError, match="no rows"):
        assert_lectures_group([])


# --------------------------------- the novelty term fires on a foreign dataset
def test_a_sequence_item_takes_the_lecture_it_is_given():
    given = SequenceItem("clip-1", 1.0, "text", in_lecture="PHYS-1")
    assert given.lecture == "PHYS-1"


def test_without_one_it_falls_back_to_this_package_s_identifiers():
    assert SequenceItem("lec-a#0001", 1.0, "text").lecture == "lec-a"
    # And a foreign identifier falls back to itself, which is why the lecture is passed.
    assert SequenceItem("clip-1", 1.0, "text").lecture == "clip-1"


def test_the_term_fires_on_foreign_identifiers_when_the_lecture_is_carried():
    """Identical descriptions in one course must be seen to repeat."""
    factor = NoveltyKnee(0.4)
    same = "Pressure against volume."
    carried = [
        SequenceItem(f"clip-{i}", float(i), same, in_lecture="PHYS-1") for i in range(1, 4)
    ]
    cut = [e.factor for e in factor.effects(carried).values() if e.factor < 1.0]
    assert len(cut) == 2  # the first has nothing to repeat

    # The same items without the lecture: three lectures of one, so nothing repeats.
    alone = [SequenceItem(f"clip-{i}", float(i), same) for i in range(1, 4)]
    assert all(e.factor == 1.0 for e in factor.effects(alone).values())


def test_scoring_a_foreign_system_carries_the_lecture_through(tmp_path: Path):
    from ad_for_edu.data.schema import MomentContext

    fields = FieldMap.parse(MAPPING)
    rows = fields.rows(FOREIGN)
    contexts = {
        row["moment_id"]: MomentContext(
            moment_id=row["moment_id"],
            type=row["type"],
            pause_after=row["pause_after"],
            slide_ocr=row["slide_ocr"],
            what_on_screen=row["what_on_screen"],
            transcript_window=row["transcript_window"],
        )
        for row in rows
    }
    same = "Pressure against volume."
    predictions = [
        {"output_id": row["moment_id"], "moment_id": row["moment_id"], "ad_text": same}
        for row in rows
    ]
    times = {row["moment_id"]: row["t"] for row in rows}
    lectures = {row["moment_id"]: row["lecture"] for row in rows}
    scored = score_predictions(
        predictions,
        contexts,
        build_scorer("mechanical"),
        sequence_factor=build_sequence_factor(),
        times=times,
        lectures=lectures,
        system="theirs",
    )
    def was_cut(row):
        factor = row.get("novelty_factor")
        # Explicitly against None: a factor of 0.0 is falsy, and `or 1.0` would read
        # the most heavily cut description of all as uncut.
        return factor is not None and factor < 1.0

    cut = [r for r in scored if was_cut(r)]
    assert len(cut) == 4  # six moments, two courses, the first of each is novel
    assert all("cross_cutting_006" in r["broken"] for r in cut)

    # Without the lectures the same system looks compliant, which is the silent failure.
    unaware = score_predictions(
        predictions,
        contexts,
        build_scorer("mechanical"),
        sequence_factor=build_sequence_factor(),
        times=times,
        system="theirs",
    )
    assert not [r for r in unaware if was_cut(r)]


# --------------------------------------------------- end to end on the command line
def test_the_diagnostic_runs_on_a_foreign_dataset(tmp_path: Path, monkeypatch, capsys):
    from ad_for_edu.cli.main import main

    for name in ("data", "work"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("AD_FOR_EDU_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AD_FOR_EDU_WORK_DIR", str(tmp_path / "work"))

    moments = tmp_path / "moments.jsonl"
    moments.write_text(
        "".join(json.dumps(row) + "\n" for row in FOREIGN), encoding="utf-8"
    )
    predictions = tmp_path / "predictions_theirs.jsonl"
    predictions.write_text(
        "".join(
            json.dumps({"id": row["id"], "narration": "Pressure against volume."}) + "\n"
            for row in FOREIGN
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "diagnose",
                "--predictions",
                str(predictions),
                "--moments",
                str(moments),
                "--fields",
                *MAPPING,
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    # Six descriptions, all identical, two courses: four break the repetition rule.
    assert "| cross_cutting_006 |" in printed
    assert "4 |" in printed


def test_the_diagnostic_refuses_a_foreign_dataset_with_no_lecture(
    tmp_path: Path, monkeypatch, capsys
):
    from ad_for_edu.cli.main import main

    for name in ("data", "work"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("AD_FOR_EDU_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AD_FOR_EDU_WORK_DIR", str(tmp_path / "work"))
    moments = tmp_path / "moments.jsonl"
    moments.write_text(
        "".join(json.dumps(row) + "\n" for row in FOREIGN), encoding="utf-8"
    )
    predictions = tmp_path / "predictions_theirs.jsonl"
    predictions.write_text(
        "".join(
            json.dumps({"id": row["id"], "narration": "x."}) + "\n" for row in FOREIGN
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "diagnose",
                "--predictions",
                str(predictions),
                "--moments",
                str(moments),
                "--fields",
                "moment_id=id",
                "text=narration",
            ]
        )
        == 1
    )
    assert "its own lecture" in capsys.readouterr().err
