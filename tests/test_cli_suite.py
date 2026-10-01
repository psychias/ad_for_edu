"""The evaluation runner and the per-rule diagnostic, run end to end.

The suite's job is to run several stages and be honest about which ones it could not.
So these tests check the negative half as carefully as the positive: a stage whose
input is absent must be named, and the paid stages must be priced once and left alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ad_for_edu.cli.main import main

SLIDE = "Mitochondrion cristae matrix"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("data", "work"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("AD_FOR_EDU_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AD_FOR_EDU_WORK_DIR", str(tmp_path / "work"))
    return tmp_path


def rows(path: Path, values) -> Path:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
    )
    return path


def moment_rows(kinds=("figure", "pointing", "slide", "slide")):
    return [
        {
            "moment_id": f"lec-a#{index:04d}",
            "lecture": "lec-a",
            "t": 10.0 * index,
            "time": 10.0 * index,
            "type": kind,
            "slide_ocr": SLIDE,
            "what_on_screen": "A labelled organelle",
            "transcript_window": "and so we go on",
            "pause_after": 2.0,
        }
        for index, kind in enumerate(kinds, start=1)
    ]


def readout_predictions(count: int = 4):
    """A system that copies the slide, so it repeats itself across the lecture."""
    return [
        {
            "output_id": f"lec-a#{index:04d}::readout",
            "moment_id": f"lec-a#{index:04d}",
            "emit": True,
            "ad_text": SLIDE + ".",
            "forced": True,
        }
        for index in range(1, count + 1)
    ]


def material(workspace: Path):
    return {
        "moments": rows(workspace / "moments.jsonl", moment_rows()),
        "predictions": rows(
            workspace / "predictions_readout.jsonl", readout_predictions()
        ),
        "references": rows(
            workspace / "references.jsonl",
            [
                {
                    "output_id": f"lec-a#{index:04d}::writer-x",
                    "moment_id": f"lec-a#{index:04d}",
                    "emit": True,
                    "ad_text": f"A labelled organelle, view {index}.",
                    "family": "writer-x",
                    "type": "figure",
                    "time": 10.0 * index,
                    "slide_ocr": SLIDE,
                    "what_on_screen": "A labelled organelle",
                    "transcript_window": "and so we go on",
                    "pause_after": 2.0,
                }
                for index in range(1, 5)
            ],
        ),
    }


def trailing_json(printed: str) -> dict:
    lines = printed.splitlines()
    starts = [index for index, line in enumerate(lines) if line == "{"]
    assert starts, f"no summary object in:\n{printed}"
    return json.loads("\n".join(lines[starts[-1] :]))


# ------------------------------------------------------------------- the diagnostic
def test_the_diagnostic_names_the_rules_a_slide_readout_breaks(workspace: Path, capsys):
    given = material(workspace)
    assert (
        main(
            [
                "diagnose",
                "--predictions",
                str(given["predictions"]),
                "--moments",
                str(given["moments"]),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    summary = trailing_json(printed)
    found = json.loads(
        Path(summary["written"]).with_suffix(".json").read_text(encoding="utf-8")
    )
    # It copies the slide, so it repeats itself and breaks the rules about repeating.
    assert found["per_rule"]["cross_cutting_006"]["broken_by"] > 0
    assert found["per_rule"]["slide_text_003"]["broken_by"] > 0
    # And it breaks none of the per-description mechanical rules, which is the point:
    # mechanical compliance alone does not catch it.
    for rule_id in ("cross_cutting_002", "cross_cutting_004", "length_004"):
        assert found["per_rule"][rule_id]["broken_by"] == 0
    assert "which rule" not in printed  # the report states rules, it does not ask


def test_the_diagnostic_says_which_rules_the_mode_could_not_test(workspace: Path, capsys):
    given = material(workspace)
    main(
        [
            "diagnose",
            "--predictions",
            str(given["predictions"]),
            "--moments",
            str(given["moments"]),
        ]
    )
    printed = capsys.readouterr().out
    assert "Not tested here" in printed
    assert "figure_001" in printed
    summary = trailing_json(printed)
    assert summary["rules_this_mode_cannot_reach"]


def test_the_diagnostic_takes_no_argument_that_changes_the_metric(workspace: Path):
    """The metric has one definition, so there is no flag that scores it differently."""
    from ad_for_edu.cli.main import build_parser

    parser = build_parser()
    given = material(workspace)
    with pytest.raises(SystemExit) as raised:
        parser.parse_args(
            [
                "diagnose",
                "--predictions",
                str(given["predictions"]),
                "--moments",
                str(given["moments"]),
                "--no-novelty",
            ]
        )
    assert raised.value.code != 0


def test_the_diagnostic_writes_both_a_report_and_its_numbers(workspace: Path, capsys):
    given = material(workspace)
    main(
        [
            "diagnose",
            "--predictions",
            str(given["predictions"]),
            "--moments",
            str(given["moments"]),
        ]
    )
    summary = trailing_json(capsys.readouterr().out)
    written = Path(summary["written"])
    assert written.suffix == ".md" and written.is_file()
    assert written.with_suffix(".json").is_file()


# ----------------------------------------------------------------------- the runner
def test_the_suite_runs_the_stages_its_inputs_allow(workspace: Path, capsys):
    given = material(workspace)
    assert (
        main(
            [
                "evaluate",
                "--predictions",
                str(given["predictions"]),
                "--moments",
                str(given["moments"]),
                "--references",
                str(given["references"]),
                "--metrics",
                "chrf",
                "compliance:mechanical",
                "--offline",
            ]
        )
        == 0
    )
    summary = trailing_json(capsys.readouterr().out)
    assert "score-systems" in summary["ran"]
    assert "diagnose" in summary["ran"]
    assert "metric-correlations" in summary["ran"]
    written = Path(summary["wrote_into"])
    assert (written / "systems.md").is_file()
    assert (written / "systems.json").is_file()
    assert (written / "rules_broken_mechanical.md").is_file()
    assert (written / "evaluation_run.json").is_file()


def test_a_stage_with_no_input_is_named_rather_than_skipped_quietly(
    workspace: Path, capsys
):
    given = material(workspace)
    main(
        [
            "evaluate",
            "--predictions",
            str(given["predictions"]),
            "--moments",
            str(given["moments"]),
            "--metrics",
            "compliance:mechanical",
            "--offline",
        ]
    )
    summary = trailing_json(capsys.readouterr().out)
    absent = summary["did_not_run"]
    assert "pair-validity" in absent
    assert "pairs" in absent["pair-validity"]
    assert "rater-agreement" in absent
    # Every stage is accounted for: it either ran or said why not.
    assert set(summary["ran"]) & set(absent) == set()


def test_the_record_of_the_run_lists_every_stage(workspace: Path, capsys):
    given = material(workspace)
    main(
        [
            "evaluate",
            "--predictions",
            str(given["predictions"]),
            "--moments",
            str(given["moments"]),
            "--metrics",
            "compliance:mechanical",
            "--offline",
        ]
    )
    summary = trailing_json(capsys.readouterr().out)
    record = json.loads(
        (Path(summary["wrote_into"]) / "evaluation_run.json").read_text(encoding="utf-8")
    )
    named = {entry["stage"] for entry in record["stages"]}
    assert named == set(summary["ran"]) | set(summary["did_not_run"])
    assert len(named) == 9  # the free stages, with no paid one offline


# ------------------------------------------------------------------ the spend gate
def test_the_suite_prices_every_paid_stage_once_and_builds_no_client(
    workspace: Path, capsys, monkeypatch
):
    import ad_for_edu.llm as llm_package

    assert hasattr(llm_package, "LLMClient")
    built = []

    class Refuses:
        def __init__(self, *args, **kwargs):
            built.append(args)
            raise AssertionError("a client was built without approval")

    monkeypatch.setattr(llm_package, "LLMClient", Refuses)

    given = material(workspace)
    drawn = rows(
        workspace / "head_to_head.jsonl",
        [{"moment_id": f"lec-a#{index:04d}"} for index in range(1, 5)],
    )
    assert (
        main(
            [
                "evaluate",
                "--predictions",
                str(given["predictions"]),
                "--moments",
                str(given["moments"]),
                "--head-to-head",
                str(drawn),
                "--judge",
                "gpt-5.5",
                "--metrics",
                "compliance:mechanical",
            ]
        )
        == 0
    )
    summary = trailing_json(capsys.readouterr().out)
    assert set(summary["would_run"]) == {
        "score-rubric",
        "rate-against-references",
        "judge-head-to-head",
    }
    assert summary["total_dollars"] > 0
    assert "no --approve-spend" in summary["note"]
    assert built == []
    # It stopped before any table was written.
    written = Path(workspace / "work" / "evaluation")
    assert [path.name for path in written.iterdir()] == ["spend_estimate.json"]


def test_offline_does_not_price_anything(workspace: Path, capsys):
    given = material(workspace)
    drawn = rows(workspace / "head_to_head.jsonl", [{"moment_id": "lec-a#0001"}])
    assert (
        main(
            [
                "evaluate",
                "--predictions",
                str(given["predictions"]),
                "--moments",
                str(given["moments"]),
                "--head-to-head",
                str(drawn),
                "--judge",
                "gpt-5.5",
                "--metrics",
                "compliance:mechanical",
                "--offline",
            ]
        )
        == 0
    )
    summary = trailing_json(capsys.readouterr().out)
    assert "would_run" not in summary
    assert "score-rubric" not in summary["ran"]
    assert "score-rubric" not in summary["did_not_run"]


def test_the_estimate_is_one_total_over_the_stages_not_one_each(
    workspace: Path, capsys
):
    given = material(workspace)
    assert (
        main(
            [
                "evaluate",
                "--predictions",
                str(given["predictions"]),
                "--moments",
                str(given["moments"]),
                "--metrics",
                "compliance:mechanical",
            ]
        )
        == 0
    )
    summary = trailing_json(capsys.readouterr().out)
    per_stage = sum(line["dollars"] for line in summary["per_stage"])
    assert summary["total_dollars"] == pytest.approx(round(per_stage, 2))
