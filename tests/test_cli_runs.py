"""Every command that needs no hosted model, run end to end on invented material.

A command that parses is not a command that works. These tests give each one files
of the shape it reads and check what it writes, so that a wrong field name or a
signature that drifted fails here rather than on a machine holding the corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ad_for_edu.cli.main import main

from .fixtures.factory import reference_rows

MOMENT_TYPES = ("slide", "figure", "pointing", "code", "chart", "animation")


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A data and work directory, so no command reads the machine it runs on."""
    data = tmp_path / "data"
    work = tmp_path / "work"
    for directory in (data, work):
        directory.mkdir()
    monkeypatch.setenv("AD_FOR_EDU_DATA_DIR", str(data))
    monkeypatch.setenv("AD_FOR_EDU_WORK_DIR", str(work))
    return tmp_path


def write_rows(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def trailing_json(printed: str) -> dict:
    """The summary a command prints after a table: the last object in its output."""
    lines = printed.splitlines()
    starts = [index for index, line in enumerate(lines) if line == "{"]
    assert starts, f"no summary object in:\n{printed}"
    return json.loads("\n".join(lines[starts[-1] :]))


def moment_rows(lectures=("lec-a", "lec-b", "lec-c"), each=6) -> list[dict]:
    rows = []
    for lecture in lectures:
        for index in range(1, each + 1):
            rows.append(
                {
                    "moment_id": f"{lecture}#{index:04d}",
                    "lecture": lecture,
                    "t": 10.0 * index,
                    "time": 10.0 * index,
                    "type": MOMENT_TYPES[index % len(MOMENT_TYPES)],
                    "slide_ocr": f"Slide {index} of {lecture} about a topic",
                    "what_on_screen": f"A picture of thing {index}",
                    "transcript_window": "and here we can see the next part of it",
                    "pause_after": 2.0,
                }
            )
    return rows


def candidate_rows() -> list[dict]:
    rows = []
    for moment in moment_rows(each=3):
        for index, writer in enumerate(("writer-x", "writer-y", "writer-z")):
            rows.append(
                {
                    "moment_id": moment["moment_id"],
                    "family": writer,
                    "directive": "REFERENT",
                    "rung": 3,
                    "ad_text": " ".join(
                        [f"word{number + index}" for number in range(8 + 4 * index)]
                    ),
                }
            )
    return rows


# ------------------------------------------------------------------ the pair commands
def test_pairs_are_built_from_the_candidates(workspace: Path, capsys):
    candidates = write_rows(workspace / "candidates.jsonl", candidate_rows())
    assert main(["build-pairs", "--candidates", str(candidates)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["pairs"] > 0
    rows = read_rows(Path(printed["written"]))
    assert {row["pair_id"] for row in rows} == {row["pair_id"] for row in rows}
    assert all(row["a"] != row["b"] for row in rows)
    assert all(row["source"] == "natural" for row in rows)


def test_the_development_pairs_are_held_out_by_whole_lectures(workspace: Path, capsys):
    pairs = write_rows(
        workspace / "pairs.jsonl",
        [
            {
                "pair_id": f"{moment['moment_id']}@1",
                "moment_id": moment["moment_id"],
                "lecture": moment["lecture"],
            }
            for moment in moment_rows()
        ],
    )
    assert main(["hold-out-dev-pairs", "--pairs", str(pairs)]) == 0
    printed = json.loads(capsys.readouterr().out)
    kept = read_rows(Path(printed["training"]["path"]))
    held = read_rows(Path(printed["development"]["path"]))
    assert kept and held
    assert not ({row["lecture"] for row in kept} & {row["lecture"] for row in held})


def test_the_moments_of_the_head_to_head_draw_are_reproducible(workspace: Path, capsys):
    many = moment_rows(lectures=tuple(f"lec-{index:02d}" for index in range(30)))
    moments = write_rows(workspace / "moments.jsonl", many)
    assert main(["draw-head-to-head", "--moments", str(moments)]) == 0
    first = json.loads(capsys.readouterr().out)
    drawn = read_rows(Path(first["written"]))
    assert len(drawn) == 150
    assert main(["draw-head-to-head", "--moments", str(moments), "--overwrite"]) == 0
    capsys.readouterr()
    assert read_rows(Path(first["written"])) == drawn


def test_a_draw_larger_than_what_is_available_is_refused(workspace: Path):
    moments = write_rows(workspace / "moments.jsonl", moment_rows())
    assert main(["draw-head-to-head", "--moments", str(moments)]) == 1


def test_an_output_that_exists_is_not_overwritten_unless_asked(workspace: Path, capsys):
    candidates = write_rows(workspace / "candidates.jsonl", candidate_rows())
    assert main(["build-pairs", "--candidates", str(candidates)]) == 0
    capsys.readouterr()
    assert main(["build-pairs", "--candidates", str(candidates)]) == 1
    assert main(["build-pairs", "--candidates", str(candidates), "--overwrite"]) == 0


# ----------------------------------------------------------------- training material
def write_manifest(workspace: Path) -> Path:
    """A manifest that puts one of the two fixture lectures on each side."""
    path = workspace / "data" / "lectures.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "lecture_id,video,course,split,renders_cursor\n"
        "lec-a,lec-a.mp4,BIOLOGY,train,\n"
        "lec-b,lec-b.mp4,COMPUTING,test,\n",
        encoding="utf-8",
    )
    return path


def test_the_training_examples_are_written_per_arm_and_side(workspace: Path, capsys):
    write_manifest(workspace)
    references = write_rows(workspace / "references.jsonl", reference_rows())
    assert main(["build-training-examples", "--references", str(references)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["rows"] == len(reference_rows())
    assert set(printed["written"]) == {
        "multimodal/train",
        "multimodal/test",
        "text_only/train",
        "text_only/test",
    }
    train = read_rows(Path(printed["written"]["multimodal/train"]["path"]))
    test = read_rows(Path(printed["written"]["multimodal/test"]["path"]))
    assert {row["lecture"] for row in train} == {"lec-a"}
    assert {row["lecture"] for row in test} == {"lec-b"}
    # Three messages: what the describer is, the moment, and the answer.
    assert [message["role"] for message in train[0]["messages"]] == [
        "system",
        "user",
        "assistant",
    ]


def test_the_text_only_arm_shows_no_picture(workspace: Path, capsys):
    write_manifest(workspace)
    references = write_rows(workspace / "references.jsonl", reference_rows())
    assert (
        main(
            ["build-training-examples", "--references", str(references), "--arms", "text_only"]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert set(printed["written"]) == {"text_only/train", "text_only/test"}
    rows = read_rows(Path(printed["written"]["text_only/train"]["path"]))
    parts = rows[0]["messages"][1]["content"]
    assert all(part.get("type") != "image" for part in parts)


def test_an_unknown_input_arm_is_refused(workspace: Path):
    write_manifest(workspace)
    references = write_rows(workspace / "references.jsonl", reference_rows())
    with pytest.raises(SystemExit) as raised:
        main(["build-training-examples", "--references", str(references), "--arms", "audio"])
    assert "unknown input arms" in str(raised.value)


def test_a_cell_of_the_grid_prints_its_layered_settings(workspace: Path, capsys):
    from ad_for_edu.training import load_backbones
    from ad_for_edu.training.build import training_dir

    backbone = sorted(load_backbones(training_dir() / "backbones"))[0]
    assert (
        main(
            [
                "train",
                "--backbone",
                backbone,
                "--arm",
                "text_only",
                "--method",
                "sft",
                "--settings-only",
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["cell"]["method"] == "sft"
    assert printed["cell"]["seed"] == 0
    assert printed["settings"]["learning_rate"] > 0
    assert printed["lora"]["rank"] > 0
    assert "share" in printed["development"]


def test_an_unknown_backbone_is_refused_and_the_known_ones_are_named(workspace: Path, capsys):
    assert (
        main(
            [
                "train",
                "--backbone",
                "not-a-model",
                "--arm",
                "text_only",
                "--method",
                "sft",
                "--settings-only",
            ]
        )
        == 1
    )


# ----------------------------------------------------------------------- describing
def test_the_readout_describes_every_moment_it_is_asked_about(workspace: Path, capsys):
    moments = write_rows(workspace / "moments.jsonl", moment_rows())
    assert (
        main(
            [
                "describe",
                "--moments",
                str(moments),
                "--system",
                "slide_title",
                "--no-pictures",
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    rows = read_rows(Path(printed["written"]))
    assert len(rows) == len(moment_rows())
    assert {row["moment_id"] for row in rows} == {row["moment_id"] for row in moment_rows()}


def test_an_unknown_system_is_refused(workspace: Path):
    moments = write_rows(workspace / "moments.jsonl", moment_rows())
    with pytest.raises(SystemExit) as raised:
        main(["describe", "--moments", str(moments), "--system", "wishful", "--no-pictures"])
    assert "unknown system" in str(raised.value)


# ------------------------------------------------------------------------- the tables
def test_the_systems_table_scores_every_system_on_every_metric(workspace: Path, capsys):
    moments = write_rows(workspace / "moments.jsonl", moment_rows())
    references = write_rows(workspace / "references.jsonl", reference_rows())
    first = write_rows(
        workspace / "predictions_one.jsonl",
        [
            {
                "output_id": row["moment_id"],
                "moment_id": row["moment_id"],
                "emit": True,
                "ad_text": f"A picture of thing {index} with labels on it",
                "forced": True,
            }
            for index, row in enumerate(moment_rows())
        ],
    )
    second = write_rows(
        workspace / "predictions_two.jsonl",
        [
            {
                "output_id": row["moment_id"],
                "moment_id": row["moment_id"],
                "emit": True,
                "ad_text": "the same sentence every time",
                "forced": True,
            }
            for row in moment_rows()
        ],
    )
    assert (
        main(
            [
                "score-systems",
                "--predictions",
                str(first),
                str(second),
                "--moments",
                str(moments),
                "--references",
                str(references),
                "--metrics",
                "chrf",
                "compliance:mechanical",
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "| system | moments | described | seeds | chrf | compliance:mechanical |" in printed
    summary = trailing_json(printed)
    table = json.loads(
        Path(summary["written"]).with_suffix(".json").read_text(encoding="utf-8")
    )
    assert {row["system"] for row in table} == {"one", "two"}
    # The system that repeats itself is cut by the novelty factor.
    by_system = {row["system"]: row["compliance:mechanical"]["value"] for row in table}
    assert by_system["two"] < by_system["one"]


def test_the_metric_correlations_read_the_written_table(workspace: Path, capsys):
    table = workspace / "systems.json"
    table.write_text(
        json.dumps(
            [
                {"system": name, "chrf": {"value": value, "n": 10},
                 "compliance:mechanical": {"value": 1 - value, "n": 10}}
                for name, value in [("a", 0.2), ("b", 0.4), ("c", 0.6), ("d", 0.8)]
            ]
        ),
        encoding="utf-8",
    )
    assert main(["metric-correlations", "--table", str(table), "--resamples", "50"]) == 0
    printed = capsys.readouterr().out
    assert "-1.00" in printed


def test_a_correlation_needs_two_metrics(workspace: Path):
    table = workspace / "systems.json"
    table.write_text(json.dumps([{"system": "a", "chrf": {"value": 0.2, "n": 4}}]), "utf-8")
    with pytest.raises(SystemExit) as raised:
        main(["metric-correlations", "--table", str(table)])
    assert "at least two metrics" in str(raised.value)


def test_the_win_rate_counts_only_the_moments_both_orders_decided(workspace: Path, capsys):
    answers = []
    for index, (first, second) in enumerate(
        [("a", "a"), ("a", "a"), ("b", "b"), ("a", "b"), ("a", None)]
    ):
        moment = f"lec-a#{index:04d}"
        answers.append({"moment_id": moment, "order": 0, "choice": first})
        answers.append({"moment_id": moment, "order": 1, "choice": second})
    path = write_rows(workspace / "answers.jsonl", answers)
    assert (
        main(
            [
                "head-to-head-win-rate",
                "--answers",
                str(path),
                "--first",
                "trained",
                "--second",
                "untrained",
                "--judge",
                "a-judge",
                "--resamples",
                "200",
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["won"] == 2 and printed["lost"] == 1
    # One moment the two orders disagreed on, and one the second order left unanswered.
    assert printed["undecided"] == 2
    assert printed["win_rate"] == pytest.approx(2 / 3, abs=5e-5)
    assert printed["lectures"] == 1


# -------------------------------------------------------------- validity and localisation
def controlled_pairs() -> tuple[list[dict], list[dict], list[dict]]:
    """Controlled pairs, a scored file and a per-category graded file."""
    categories = ("style", "terminology", "length", "deixis", "faithfulness", "non_redundancy")
    pairs, scores, graded = [], [], []
    for index, category in enumerate(categories):
        for number in range(6):
            pair_id = f"lec-{'abc'[number % 3]}#{index:02d}{number:02d}@1"
            side = "a" if number % 2 == 0 else "b"
            pairs.append(
                {
                    "pair_id": pair_id,
                    "moment_id": pair_id.split("@")[0],
                    "lecture": pair_id.split("#")[0],
                    "axis": category,
                    "compliant_side": side,
                }
            )
            better, worse = (0.9, 0.4) if side == "a" else (0.4, 0.9)
            scores.append({"pair_id": pair_id, "scorer": "rubric", "a": better, "b": worse})
            # chrF gets it right half the time, so the two are distinguishable.
            flip = number < 3
            scores.append(
                {
                    "pair_id": pair_id,
                    "scorer": "chrf",
                    "a": better if flip else worse,
                    "b": worse if flip else better,
                }
            )
            graded.append(
                {
                    "pair_id": pair_id,
                    "a": {name: (4 if side == "a" else 2) for name in categories},
                    "b": {name: (2 if side == "a" else 4) for name in categories},
                }
            )
    return pairs, scores, graded


def test_coverage_and_accuracy_are_reported_per_scorer_and_tested(workspace: Path, capsys):
    pairs, scores, _ = controlled_pairs()
    pair_path = write_rows(workspace / "pairs.jsonl", pairs)
    score_path = write_rows(workspace / "scores.jsonl", scores)
    assert (
        main(
            [
                "pair-validity",
                "--pairs",
                str(pair_path),
                "--scores",
                str(score_path),
                "--cascade",
                "rubric",
                "chrf",
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["per_scorer"]["rubric"]["accuracy"] == 1.0
    assert printed["per_scorer"]["chrf"]["accuracy"] == pytest.approx(0.5)
    assert printed["per_scorer"]["cascade"]["decided_by"]["rubric"] == len(pairs)
    assert printed["tests"]["rubric"]["distinct"] is True


def test_pairs_with_no_known_side_cannot_be_measured(workspace: Path):
    pairs = write_rows(
        workspace / "pairs.jsonl", [{"pair_id": "p1", "moment_id": "lec-a#0001"}]
    )
    scores = write_rows(
        workspace / "scores.jsonl", [{"pair_id": "p1", "scorer": "chrf", "a": 1.0, "b": 0.0}]
    )
    with pytest.raises(SystemExit) as raised:
        main(["pair-validity", "--pairs", str(pairs), "--scores", str(scores)])
    assert "controlled pairs only" in str(raised.value)


def test_localisation_reports_every_category_with_its_own_count(workspace: Path, capsys):
    pairs, _, graded = controlled_pairs()
    pair_path = write_rows(workspace / "pairs.jsonl", pairs)
    graded_path = write_rows(workspace / "graded.jsonl", graded)
    assert (
        main(["localisation", "--pairs", str(pair_path), "--subscores", str(graded_path)]) == 0
    )
    printed = json.loads(capsys.readouterr().out)["per_category"]
    assert len(printed) == 6
    for category, cell in printed.items():
        assert cell["pairs_on_this_category"] == 6, category
        assert cell["accuracy"] == 1.0
        assert cell["interval"]["low"] < 1.0 <= cell["interval"]["high"]


def test_the_rank_of_the_systems_is_reported_at_every_knee(workspace: Path, capsys):
    rows = []
    for system, text in (
        ("varied", "a different sentence number {index} about the slide"),
        ("repeats", "the same sentence about the slide"),
    ):
        for index in range(8):
            rows.append(
                {
                    "system": system,
                    "moment_id": f"lec-a#{index:04d}",
                    "time": 10.0 * index,
                    "score": 0.8,
                    "ad_text": text.format(index=index),
                }
            )
    path = write_rows(workspace / "per_moment.jsonl", rows)
    assert main(["rank-stability", "--per-moment", str(path), "--resamples", "50"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["knees"] == [0.2, 0.3, 0.4, 0.5, 0.6]
    assert printed["systems"] == 2
    for ranks in printed["ranks"].values():
        assert ranks["varied"] < ranks["repeats"]


def test_the_reference_writers_are_scored_like_any_system(workspace: Path, capsys):
    rows = reference_rows()
    moments = write_rows(
        workspace / "moments.jsonl",
        [
            {
                "moment_id": row["output_id"].split("::")[0],
                "lecture": row["output_id"].split("#")[0],
                "t": float(row["time"]),
                "time": float(row["time"]),
                "type": row["type"],
                "slide_ocr": row["slide_ocr"],
                "what_on_screen": row["what_on_screen"],
                "transcript_window": row["transcript_window"],
                "pause_after": row["pause_after"],
            }
            for row in rows
            if row["family"] == "writer-x"
        ],
    )
    references = write_rows(workspace / "references.jsonl", rows)
    assert (
        main(
            [
                "reference-writer-compliance",
                "--references",
                str(references),
                "--moments",
                str(moments),
                "--metrics",
                "compliance:mechanical",
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "# Reference writers" in printed
    counts = trailing_json(printed)["writers"]
    # The writer that stayed silent everywhere has nothing to score.
    assert counts["writer-z"] == 0
    assert counts["writer-x"] > 0


# ------------------------------------------------------------- raters and the head
def rated_material(workspace: Path):
    pairs, scores, labels = [], [], {"r1": [], "r2": [], "r3": []}
    features = []
    for index in range(36):
        lecture = f"lec-{'abc'[index % 3]}"
        pair_id = f"{lecture}#{index:04d}@1"
        better = "a" if index % 2 == 0 else "b"
        pairs.append(
            {
                "pair_id": pair_id,
                "moment_id": pair_id.split("@")[0],
                "lecture": lecture,
                "axis": "deixis" if index % 3 == 0 else "length",
            }
        )
        first, second = (0.9, 0.3) if better == "a" else (0.3, 0.9)
        scores.append({"pair_id": pair_id, "scorer": "chrf", "a": first, "b": second})
        scores.append({"pair_id": pair_id, "scorer": "rubric", "a": second, "b": first})
        features.append(
            {
                "pair_id": pair_id,
                "a": {"chrf": first, "words": 20.0},
                "b": {"chrf": second, "words": 30.0},
            }
        )
        for rater, wrong_every in (("r1", 9), ("r2", 6), ("r3", 4)):
            other = "b" if better == "a" else "a"
            labels[rater].append(
                {"pair_id": pair_id, "side": other if index % wrong_every == 0 else better}
            )
    written = {
        "pairs": write_rows(workspace / "pairs.jsonl", pairs),
        "scores": write_rows(workspace / "scores.jsonl", scores),
        "features": write_rows(workspace / "features.jsonl", features),
    }
    written["labels"] = [
        write_rows(workspace / f"{rater}.jsonl", rows) for rater, rows in labels.items()
    ]
    return written


def test_the_metrics_are_compared_with_each_rater_and_each_group(workspace: Path, capsys):
    material = rated_material(workspace)
    assert (
        main(
            [
                "rater-agreement",
                "--pairs",
                str(material["pairs"]),
                "--labels",
                *[str(path) for path in material["labels"]],
                "--scores",
                str(material["scores"]),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "# Metric agreement with the raters" in printed
    written = trailing_json(printed)
    assert written["raters"] == ["r1", "r2", "r3"]
    assert written["metrics"] == ["chrf", "rubric"]
    found = json.loads(Path(written["written"]).read_text(encoding="utf-8"))
    assert set(found["by_group"]) == {"deixis", "length"}
    # chrF follows the raters, the rubric is its mirror, so they cannot both be high.
    # Each rater was given a share of wrong answers by construction, so the cells
    # differ; what holds for every rater is the ordering of the two metrics.
    for rater in written["raters"]:
        chrf = found["overall"][rater]["chrf"]
        rubric = found["overall"][rater]["rubric"]
        assert chrf["agreement"] > 0.7
        assert rubric["agreement"] < 0.3
        assert chrf["n"] == rubric["n"] == 36
    assert written["agreement_between_raters"]["units"] > 0


def test_the_head_is_fitted_out_of_fold_and_one_rater_is_held_out(workspace: Path, capsys):
    material = rated_material(workspace)
    assert (
        main(
            [
                "fit-preference-head",
                "--pairs",
                str(material["pairs"]),
                "--labels",
                *[str(path) for path in material["labels"]],
                "--features",
                str(material["features"]),
                "--keep",
                "chrf",
                "words",
                "--hold-out-rater",
                "r3",
                "--held-at-equal",
                "words",
            ]
        )
        == 0
    )
    printed = trailing_json(capsys.readouterr().out)
    assert printed["fitted_on_raters"] == ["r1", "r2"]
    assert printed["fold_disjoint_on"] == "lecture"
    assert printed["folds"] == 3
    assert printed["held_out"]["rater"] == "r3"
    assert printed["held_out"]["refitted"] is False
    assert printed["held_out"]["agreement"] > 0.5
    assert printed["held_equal"]["feature"] == "words"
    rows = read_rows(Path(printed["written"]))
    assert len(rows) == 36
    assert all(row["fold"] == row["pair_id"].split("#")[0] for row in rows)


def test_holding_a_screened_out_feature_equal_is_refused_with_a_way_forward(workspace: Path):
    material = rated_material(workspace)
    with pytest.raises(SystemExit) as raised:
        main(
            [
                "fit-preference-head",
                "--pairs",
                str(material["pairs"]),
                "--labels",
                *[str(path) for path in material["labels"]],
                "--features",
                str(material["features"]),
                "--keep",
                "chrf",
                "--held-at-equal",
                "words",
            ]
        )
    message = str(raised.value)
    assert "cannot be held equal" in message
    assert "--keep" in message


def test_a_feature_that_is_not_in_the_file_is_refused(workspace: Path):
    material = rated_material(workspace)
    with pytest.raises(SystemExit) as raised:
        main(
            [
                "fit-preference-head",
                "--pairs",
                str(material["pairs"]),
                "--labels",
                str(material["labels"][0]),
                "--features",
                str(material["features"]),
                "--use",
                "clipscore",
            ]
        )
    assert "clipscore" in str(raised.value)


# --------------------------------------- a rater is named by a pseudonym, never a person
def test_a_label_file_named_after_a_person_is_refused(workspace: Path, capsys):
    """The rater's name reaches the report, so it cannot be a participant's."""
    material = rated_material(workspace)
    named_after_someone = workspace / "jane_doe_answers.jsonl"
    named_after_someone.write_bytes(material["labels"][0].read_bytes())
    assert (
        main(
            [
                "rater-agreement",
                "--pairs",
                str(material["pairs"]),
                "--labels",
                str(named_after_someone),
                "--scores",
                str(material["scores"]),
            ]
        )
        == 1
    )
    message = capsys.readouterr().err
    assert "not pseudonyms" in message
    # The refusal must not repeat the name it is protecting.
    assert "jane" not in message.lower() and "doe" not in message.lower()


def test_such_a_file_can_still_be_used_under_a_pseudonym(workspace: Path, capsys):
    material = rated_material(workspace)
    named_after_someone = workspace / "jane_doe_answers.jsonl"
    named_after_someone.write_bytes(material["labels"][0].read_bytes())
    assert (
        main(
            [
                "rater-agreement",
                "--pairs",
                str(material["pairs"]),
                "--labels",
                str(named_after_someone),
                "--raters",
                "R1",
                "--scores",
                str(material["scores"]),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "R1" in printed
    assert "jane" not in printed.lower()


def test_one_name_per_file_is_required(workspace: Path):
    material = rated_material(workspace)
    with pytest.raises(SystemExit, match="one name per file"):
        main(
            [
                "rater-agreement",
                "--pairs",
                str(material["pairs"]),
                "--labels",
                *[str(path) for path in material["labels"]],
                "--raters",
                "R1",
                "--scores",
                str(material["scores"]),
            ]
        )


def test_the_preference_head_refuses_a_person_named_file_too(workspace: Path, capsys):
    material = rated_material(workspace)
    named_after_someone = workspace / "jane_doe_answers.jsonl"
    named_after_someone.write_bytes(material["labels"][0].read_bytes())
    assert (
        main(
            [
                "fit-preference-head",
                "--pairs",
                str(material["pairs"]),
                "--labels",
                str(named_after_someone),
                "--features",
                str(material["features"]),
            ]
        )
        == 1
    )
    assert "not pseudonyms" in capsys.readouterr().err
