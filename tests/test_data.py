"""Dataset sources, the catalogue of files, the manifest, the split and the media layout."""

from __future__ import annotations

import json

import pytest

from ad_for_edu.core.errors import ContractError, MissingSourceError, SettingsError
from ad_for_edu.data import (
    DATASET_SOURCES,
    SPLIT_POLICIES,
    DataCatalog,
    DatasetSplit,
    LectureManifest,
    ManifestSplit,
    MediaLayout,
    assert_disjoint_lectures,
    assert_disjoint_moments,
    assert_no_test_lecture,
)
from ad_for_edu.data.sources import LocalDirSource

MANIFEST = """lecture_id,video,course,split,renders_cursor
lec-a,lec-a.mp4,biology,train,true
lec-b,lec-b.mp4,biology,train,false
lec-c,lec-c.mp4,algorithms,test,
lec-d,lec-d.mp4,algorithms,test,yes
"""


@pytest.fixture
def manifest(tmp_path) -> LectureManifest:
    path = tmp_path / "lectures.csv"
    path.write_text(MANIFEST, encoding="utf-8")
    return LectureManifest.load(path)


def rows(*moments: str) -> list[dict]:
    return [{"moment_id": moment} for moment in moments]


# ---------------------------------------------------------------- sources
def test_the_two_sources():
    assert DATASET_SOURCES.names() == ("hub", "local_dir")


def test_a_directory_serves_files_in_the_layout_of_a_dataset(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "table.parquet").write_bytes(b"x")
    source = LocalDirSource()
    assert source.fetch(str(tmp_path), "data/table.parquet").read_bytes() == b"x"
    assert source.fetch_all(str(tmp_path), tmp_path / "unused") == tmp_path


def test_a_missing_file_or_directory_is_an_error(tmp_path):
    source = LocalDirSource()
    with pytest.raises(MissingSourceError):
        source.fetch(str(tmp_path), "absent.jsonl")
    with pytest.raises(MissingSourceError):
        source.fetch(str(tmp_path / "absent"), "rows.jsonl")


@pytest.mark.parametrize("name", ["../outside.jsonl", "a/../../b.jsonl"])
def test_a_file_name_cannot_leave_its_dataset(tmp_path, name):
    with pytest.raises(SettingsError):
        LocalDirSource().fetch(str(tmp_path), name)


# ---------------------------------------------------------------- catalogue
@pytest.fixture
def catalog(shipped_configs) -> DataCatalog:
    return DataCatalog.load(shipped_configs / "data.yaml", source=LocalDirSource())


def test_loading_the_catalogue_needs_no_environment(catalog):
    assert "reference_rows" in catalog.roles() and "rated_pairs" in catalog.roles()


def test_a_dataset_is_named_by_the_environment_only_when_it_is_read(catalog, tmp_path, monkeypatch):
    with pytest.raises(SettingsError) as caught:
        catalog.dataset_file("reference_rows")
    assert "AD_FOR_EDU_REFERENCES_DATASET" in str(caught.value)
    (tmp_path / "gold_eval.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("AD_FOR_EDU_REFERENCES_DATASET", str(tmp_path))
    assert catalog.dataset_file("reference_rows") == tmp_path / "gold_eval.jsonl"


def test_reading_one_dataset_does_not_need_the_name_of_another(catalog, tmp_path, monkeypatch):
    (tmp_path / "gold_eval.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("AD_FOR_EDU_REFERENCES_DATASET", str(tmp_path))
    monkeypatch.delenv("AD_FOR_EDU_PREFERENCES_DATASET", raising=False)
    assert catalog.dataset_file("reference_rows").is_file()


def test_a_file_name_with_a_placeholder_needs_its_value(catalog, tmp_path, monkeypatch):
    (tmp_path / "machine_labels_model-x.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("AD_FOR_EDU_PREFERENCES_DATASET", str(tmp_path))
    found = catalog.dataset_file("machine_labels", annotator="model-x")
    assert found.name == "machine_labels_model-x.jsonl"
    with pytest.raises(SettingsError) as caught:
        catalog.dataset_file("machine_labels")
    assert "annotator" in str(caught.value)


def test_an_unknown_role_lists_the_known_ones(catalog):
    with pytest.raises(SettingsError) as caught:
        catalog.dataset_file("references")
    assert "reference_rows" in str(caught.value)


def test_local_and_output_locations_hang_off_the_two_directories(catalog, tmp_path, monkeypatch):
    monkeypatch.setenv("AD_FOR_EDU_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AD_FOR_EDU_WORK_DIR", str(tmp_path / "work"))
    assert catalog.local_path("keyframes") == tmp_path / "data" / "preprocessing" / "keyframes"
    assert catalog.output_path("references") == tmp_path / "work" / "references"
    with pytest.raises(SettingsError):
        catalog.local_path("frames")


def test_the_shipped_settings_name_no_dataset_and_no_absolute_path(shipped_configs):
    text = (shipped_configs / "data.yaml").read_text(encoding="utf-8")
    values = [line.split(":", 1)[1].strip() for line in text.splitlines() if "id:" in line]
    assert values and all(value.startswith("${") and value.endswith("}") for value in values)


def test_a_role_registered_twice_is_refused(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text(
        "datasets:\n"
        "  one: {id: a, files: {rows: x.jsonl}}\n"
        "  two: {id: b, files: {rows: y.jsonl}}\n"
        "local: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(SettingsError) as caught:
        DataCatalog.load(path, source=LocalDirSource()).dataset_file("rows")
    assert "more than one" in str(caught.value)


# ---------------------------------------------------------------- manifest
def test_the_manifest_lists_lectures_with_their_side_and_course(manifest):
    assert len(manifest) == 4
    assert manifest.on_side("train") == ("lec-a", "lec-b")
    assert manifest.on_side("test") == ("lec-c", "lec-d")
    assert manifest.courses() == {
        "algorithms": ("lec-c", "lec-d"),
        "biology": ("lec-a", "lec-b"),
    }


def test_a_lecture_not_probed_has_no_pointer_fact(manifest):
    assert manifest.cursor_facts() == {"lec-a": True, "lec-b": False, "lec-d": True}
    assert manifest.get("lec-c").renders_cursor is None


def test_an_unlisted_lecture_has_no_side(manifest):
    with pytest.raises(ContractError) as caught:
        manifest.get("lec-z")
    assert "never assumed" in str(caught.value)


def test_whole_courses_are_held_out(manifest, tmp_path):
    assert manifest.check_courses_are_not_split() == []
    mixed = tmp_path / "mixed.csv"
    mixed.write_text(MANIFEST.replace("lec-b.mp4,biology,train", "lec-b.mp4,biology,test"), "utf-8")
    assert LectureManifest.load(mixed).check_courses_are_not_split() == ["biology"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (("algorithms,test,yes", "algorithms,dev,yes"), "split"),
        (("algorithms,test,yes", "algorithms,test,maybe"), "renders_cursor"),
        (("lec-d,lec-d.mp4", "lec-a,lec-d.mp4"), "twice"),
    ],
)
def test_a_malformed_manifest_is_refused(tmp_path, change, message):
    path = tmp_path / "lectures.csv"
    path.write_text(MANIFEST.replace(*change), encoding="utf-8")
    with pytest.raises(ContractError) as caught:
        LectureManifest.load(path)
    assert message in str(caught.value)


def test_a_manifest_without_a_required_column_is_refused(tmp_path):
    path = tmp_path / "lectures.csv"
    path.write_text("lecture_id,video,split\nlec-a,lec-a.mp4,train\n", encoding="utf-8")
    with pytest.raises(ContractError) as caught:
        LectureManifest.load(path)
    assert "course" in str(caught.value)


def test_an_empty_manifest_is_refused(tmp_path):
    path = tmp_path / "lectures.csv"
    path.write_text("lecture_id,video,course,split,renders_cursor\n", encoding="utf-8")
    with pytest.raises(ContractError):
        LectureManifest.load(path)


# ---------------------------------------------------------------- split
def test_the_two_split_policies():
    assert SPLIT_POLICIES.names() == ("dataset_split", "manifest")


def test_the_side_of_a_row_is_the_side_of_its_lecture(manifest):
    policy = ManifestSplit(manifest)
    assert policy.side("lec-a") == "train" and policy.is_test("lec-c")
    assert policy.side_of_row({"output_id": "lec-c#0004::writer-x"}) == "test"
    assert policy.side_of_row({"moment_id": "lec-a#g002"}) == "train"
    assert policy.side_of_row({"lecture": "lec-d", "pair_id": "p1"}) == "test"


def test_rows_are_selected_by_side(manifest):
    policy = ManifestSplit(manifest)
    given = rows("lec-a#0001", "lec-c#0001", "lec-b#0002", "lec-d#0009")
    assert [row["moment_id"] for row in policy.select(given, "test")] == [
        "lec-c#0001",
        "lec-d#0009",
    ]
    with pytest.raises(ContractError):
        policy.select(given, "dev")


def test_a_lecture_of_unknown_side_stops_the_selection(manifest):
    with pytest.raises(ContractError):
        ManifestSplit(manifest).select(rows("lec-a#0001", "lec-z#0001"), "train")


def test_the_published_files_define_a_split():
    policy = DatasetSplit(rows("lec-a#0001", "lec-a#0002", "lec-b#0001"), rows("lec-c#0001"))
    assert policy.side("lec-b") == "train" and policy.side("lec-c") == "test"
    with pytest.raises(ContractError):
        policy.side("lec-z")


def test_a_lecture_on_both_sides_of_the_published_files_is_refused():
    with pytest.raises(ContractError) as caught:
        DatasetSplit(rows("lec-a#0001", "lec-b#0001"), rows("lec-b#0007"))
    assert "lec-b" in str(caught.value)


def test_an_empty_side_is_refused():
    with pytest.raises(ContractError):
        DatasetSplit(rows("lec-a#0001"), [])


def test_training_material_from_a_test_lecture_stops_the_run(manifest):
    policy = ManifestSplit(manifest)
    assert_no_test_lecture(rows("lec-a#0001", "lec-b#0001"), policy)
    with pytest.raises(ContractError) as caught:
        assert_no_test_lecture(rows("lec-a#0001", "lec-c#0003"), policy)
    assert "lec-c" in str(caught.value)


# ---------------------------------------------------------------- disjointness
def test_a_shared_moment_is_found():
    with pytest.raises(ContractError) as caught:
        assert_disjoint_moments(
            rows("lec-a#0001", "lec-a#0002"),
            rows("lec-a#0002", "lec-b#0001"),
            first_name="rating pool",
            second_name="training pairs",
        )
    assert "lec-a#0002" in str(caught.value)


def test_sets_on_the_same_lectures_without_a_shared_moment_pass():
    assert_disjoint_moments(
        rows("lec-a#0001", "lec-b#0001"),
        rows("lec-a#0002", "lec-b#0002"),
        first_name="rating pool",
        second_name="training pairs",
    )


def test_an_empty_overlap_between_different_identifiers_proves_nothing():
    with pytest.raises(ContractError) as caught:
        assert_disjoint_moments(
            rows("lec-a#0001", "lec-b#0001"),
            [{"moment_id": "lecture_a_seg001"}, {"moment_id": "lecture_b_seg004"}],
            first_name="rating pool",
            second_name="training pairs",
        )
    assert "share no lecture" in str(caught.value)


def test_sets_meant_to_come_from_different_lectures_may_share_none():
    assert_disjoint_moments(
        rows("lec-a#0001"),
        rows("lec-c#0001"),
        first_name="training pairs",
        second_name="development pairs",
        same_lectures=False,
    )


def test_a_set_without_moments_cannot_be_compared():
    with pytest.raises(ContractError):
        assert_disjoint_moments(rows("lec-a#0001"), [], first_name="a", second_name="b")
    with pytest.raises(ContractError):
        assert_disjoint_moments(
            rows("lec-a#0001"), [{"pair_id": "p1"}], first_name="a", second_name="b"
        )


def test_a_shared_lecture_is_found():
    assert_disjoint_lectures(
        rows("lec-a#0001"), rows("lec-c#0001"), first_name="training", second_name="development"
    )
    with pytest.raises(ContractError) as caught:
        assert_disjoint_lectures(
            rows("lec-a#0001", "lec-b#0002"),
            rows("lec-b#0009"),
            first_name="training",
            second_name="development",
        )
    assert "lec-b" in str(caught.value)


# ---------------------------------------------------------------- media
@pytest.fixture
def layout(tmp_path) -> MediaLayout:
    for name in ("transcripts", "vad", "ocr", "keyframes", "videos"):
        (tmp_path / name).mkdir()
    (tmp_path / "transcripts" / "lec-a.json").write_text(
        json.dumps(
            {
                "words": [
                    {"start": 1.0, "word": " so "},
                    {"start": 1.4, "word": "here"},
                    {"word": "untimed"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "vad" / "lec-a.json").write_text(
        json.dumps({"silences": [{"start": 2.0, "end": 3.5}, {"start": 9.0}]}), encoding="utf-8"
    )
    (tmp_path / "ocr" / "lec-a.json").write_text(
        json.dumps(
            {
                "frames": [
                    {"timestamp": "00:01:00", "text": " Second slide "},
                    {"timestamp": "00:00:10", "text": "First slide"},
                    {"timestamp": "unreadable", "text": "dropped"},
                ]
            }
        ),
        encoding="utf-8",
    )
    frames = tmp_path / "keyframes" / "lec-a"
    frames.mkdir()
    for name in ("000100.jpg", "000010.jpg", "notes.txt"):
        (frames / name).write_bytes(b"x")
    (tmp_path / "videos" / "lec-a.mp4").write_bytes(b"x")
    return MediaLayout(
        transcripts=tmp_path / "transcripts",
        speech_gaps=tmp_path / "vad",
        slide_text=tmp_path / "ocr",
        keyframes=tmp_path / "keyframes",
        videos=tmp_path / "videos",
    )


def test_words_gaps_and_slide_text_are_read_in_seconds(layout):
    lecture = layout.lecture("lec-a")
    assert [(word.start, word.text) for word in lecture.words] == [(1.0, "so"), (1.4, "here")]
    assert [(gap.start, gap.end, gap.length) for gap in lecture.gaps] == [(2.0, 3.5, 1.5)]
    assert [(frame.time, frame.text) for frame in lecture.slide_text] == [
        (10, "First slide"),
        (60, "Second slide"),
    ]


def test_keyframes_are_ordered_by_the_time_in_their_name(layout):
    frames = layout.lecture("lec-a").keyframes
    assert [frame.time for frame in frames] == [10, 60]
    assert [frame.path.name for frame in frames] == ["000010.jpg", "000100.jpg"]


def test_the_recording_is_found_by_the_name_of_the_lecture(layout):
    assert layout.lecture("lec-a").video.name == "lec-a.mp4"
    assert layout.lecture("lec-b").video is None


@pytest.mark.parametrize("artefact", ["words", "gaps", "slide_text", "keyframes"])
def test_a_missing_artefact_is_an_error_and_never_an_empty_value(layout, artefact):
    with pytest.raises(MissingSourceError) as caught:
        getattr(layout.lecture("lec-b"), artefact)
    assert "lec-b" in str(caught.value)
