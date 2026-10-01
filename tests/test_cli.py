"""The command line: the parser, the shared options, and the gate on spending.

The important property here is negative. A command that can call a hosted model must
not be able to call one without being told to, and the test for that is not that it
declines politely: it is that the object which makes calls is never built.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ad_for_edu.cli.common import COMMANDS, Command
from ad_for_edu.cli.main import build_parser, main
from ad_for_edu.core.spend import APPROVE_DEST, APPROVE_FLAG

#: The commands this repository offers. A command added or renamed fails here first.
EXPECTED = (
    "annotate-rated-pairs",
    "build-pairs",
    "build-training-examples",
    "check-standard",
    "classify-moments",
    "describe",
    "detect-moments",
    "diagnose",
    "draw-head-to-head",
    "draw-rating-pool",
    "evaluate",
    "fit-preference-head",
    "generate-candidates",
    "generate-controlled-pairs",
    "generate-references",
    "head-to-head-win-rate",
    "hold-out-dev-pairs",
    "judge-head-to-head",
    "judge-pairs",
    "list-strategies",
    "localisation",
    "metric-correlations",
    "pair-validity",
    "preprocess",
    "probe-cursor",
    "rank-stability",
    "rate-against-references",
    "rater-agreement",
    "reference-writer-compliance",
    "score-rubric",
    "score-systems",
    "train",
)

#: The commands that can call a hosted model.
PAID = (
    "annotate-rated-pairs",
    "classify-moments",
    "evaluate",
    "generate-candidates",
    "generate-controlled-pairs",
    "generate-references",
    "judge-head-to-head",
    "judge-pairs",
    "preprocess",
    "rate-against-references",
    "score-rubric",
)


# ------------------------------------------------------------------ the command set
def test_every_command_is_registered_once_and_no_others():
    assert COMMANDS.names() == EXPECTED


def test_every_command_says_what_it_does_and_whether_it_can_spend():
    for name in COMMANDS.names():
        command = COMMANDS.create(name)
        assert isinstance(command, Command)
        assert command.name == name, f"{name} answers to a different name"
        assert command.help and command.help[0].islower()
        assert isinstance(command.paid, bool)


def test_the_paid_commands_are_the_stated_ones():
    found = tuple(name for name in COMMANDS.names() if COMMANDS.create(name).paid)
    assert found == PAID


# ---------------------------------------------------------------------- the parser
def test_the_parser_carries_every_command():
    parser = build_parser()
    for name in EXPECTED:
        parsed = parser.parse_args([name, *_required_for(name)])
        assert parsed.command == name


def test_every_command_takes_the_shared_options():
    parser = build_parser()
    for name in EXPECTED:
        parsed = parser.parse_args([name, *_required_for(name)])
        assert parsed.config_dir is None
        assert parsed.out is None
        assert parsed.overwrite is False


def test_only_a_paid_command_offers_the_approval_flag():
    parser = build_parser()
    for name in EXPECTED:
        parsed = parser.parse_args([name, *_required_for(name)])
        has_flag = hasattr(parsed, APPROVE_DEST)
        assert has_flag == (name in PAID), f"{name} has the wrong flag"


def test_an_unknown_command_is_refused_with_a_non_zero_status():
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(["invent-a-benchmark"])
    assert raised.value.code != 0


def test_no_command_prints_the_help_and_fails():
    assert main([]) == 2


def _required_for(name: str) -> list[str]:
    """The arguments a command cannot be parsed without."""
    needed: dict[str, list[str]] = {
        "annotate-rated-pairs": ["--pairs", "p.jsonl", "--model", "a-model"],
        "build-pairs": ["--candidates", "c.jsonl"],
        "classify-moments": ["--events", "e.jsonl"],
        "describe": ["--moments", "m.jsonl", "--system", "slide_title"],
        "diagnose": ["--predictions", "p.jsonl", "--moments", "m.jsonl"],
        "draw-head-to-head": ["--moments", "m.jsonl"],
        "draw-rating-pool": ["--pairs", "p.jsonl"],
        "fit-preference-head": [
            "--pairs", "p.jsonl", "--labels", "r1.jsonl", "--features", "f.jsonl",
        ],
        "generate-candidates": ["--moments", "m.jsonl"],
        "generate-controlled-pairs": ["--moments", "m.jsonl"],
        "generate-references": ["--moments", "m.jsonl"],
        "head-to-head-win-rate": [
            "--answers", "a.jsonl", "--first", "one", "--second", "two", "--judge", "j",
        ],
        "hold-out-dev-pairs": ["--pairs", "p.jsonl"],
        "judge-head-to-head": ["--moments", "m.jsonl", "--judge", "a-model"],
        "judge-pairs": ["--pairs", "p.jsonl"],
        "localisation": ["--pairs", "p.jsonl", "--subscores", "s.jsonl"],
        "metric-correlations": ["--table", "t.json"],
        "pair-validity": ["--pairs", "p.jsonl", "--scores", "s.jsonl"],
        "rank-stability": ["--per-moment", "p.jsonl"],
        "rate-against-references": ["--predictions", "p.jsonl"],
        "rater-agreement": [
            "--pairs", "p.jsonl", "--labels", "r1.jsonl", "--scores", "s.jsonl",
        ],
        "reference-writer-compliance": ["--moments", "m.jsonl"],
        "score-rubric": ["--predictions", "p.jsonl"],
        "score-systems": ["--predictions", "p.jsonl", "--moments", "m.jsonl"],
        "train": ["--backbone", "b", "--arm", "text_only", "--method", "sft"],
    }
    return needed.get(name, [])


# ------------------------------------------------------------------ the spend gate
def test_a_paid_command_stops_before_building_a_client(monkeypatch, tmp_path, capsys):
    """The gate holds because the client is never constructed, not because it declines.

    The command is stopped where the estimate is printed, which is a success: the
    person asked what the work would cost and was told. So the status is zero, the
    estimate is on the output, and nothing that can make a call was built.
    """
    import ad_for_edu.llm as llm_package

    # The thing being patched has to be the thing the command imports, or this test
    # would pass with no client anywhere.
    assert hasattr(llm_package, "LLMClient")
    built = []

    class Refuses:
        def __init__(self, *args, **kwargs):
            built.append(args)
            raise AssertionError("a client was built without approval")

    monkeypatch.setattr(llm_package, "LLMClient", Refuses)

    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        json.dumps({"pair_id": "lec-a#0001@1", "moment_id": "lec-a#0001"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as raised:
        main(["judge-pairs", "--pairs", str(pairs)])
    assert raised.value.code == 0
    printed = capsys.readouterr().out
    assert "SPEND ESTIMATE" in printed
    assert "no --approve-spend" in printed
    assert built == []


def test_the_estimate_names_the_calls_and_the_model(tmp_path, capsys):
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        "\n".join(
            json.dumps({"pair_id": f"lec-a#0001@{index}", "moment_id": "lec-a#0001"})
            for index in range(1, 4)
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        main(["judge-pairs", "--pairs", str(pairs)])
    printed = capsys.readouterr().out
    # Three pairs, judged in both presentation orders.
    assert " 6 " in printed
    assert "judge_pairs" in printed


def test_the_approval_flag_is_spelled_the_same_everywhere():
    parser = build_parser()
    parsed = parser.parse_args(["judge-pairs", "--pairs", "p.jsonl", APPROVE_FLAG])
    assert getattr(parsed, APPROVE_DEST) is True


# --------------------------------------------------------- commands that need nothing
def test_the_strategies_can_be_listed_without_any_data(capsys):
    assert main(["list-strategies"]) == 0
    printed = capsys.readouterr().out
    assert "metric:" in printed
    assert "  chrf" in printed


def test_one_family_can_be_listed_on_its_own(capsys):
    assert main(["list-strategies", "--family", "metric"]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("metric:")
    assert "component:" not in printed


def test_an_unknown_family_fails_and_names_the_known_ones(capsys):
    assert main(["list-strategies", "--family", "vibes"]) == 1
    assert "known:" in capsys.readouterr().out


def test_the_standard_can_be_checked_without_any_data(capsys, tmp_path):
    assert main(["check-standard"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["rules"] == 45
    assert printed["judge_prompts"] == 42
    assert printed["routes"] == {
        "mechanical": 14,
        "judge": 17,
        "penalty": 2,
        "unscored": 12,
    }


def test_the_standard_check_writes_where_it_is_told(tmp_path):
    target = tmp_path / "standard.json"
    assert main(["check-standard", "--out", str(target)]) == 0
    assert json.loads(target.read_text(encoding="utf-8"))["rules"] == 45


# --------------------------------------------------------------- settings directory
def test_a_settings_directory_that_is_not_there_is_refused(tmp_path, capsys):
    missing = tmp_path / "nowhere"
    with pytest.raises(Exception) as raised:
        main(["check-standard", "--config-dir", str(missing)])
    assert "not a directory" in str(raised.value)


def test_a_named_settings_directory_is_used(tmp_path, monkeypatch, shipped_configs: Path):
    import shutil

    elsewhere = tmp_path / "settings"
    shutil.copytree(shipped_configs, elsewhere)
    assert main(["list-strategies", "--config-dir", str(elsewhere)]) == 0
    from ad_for_edu.core.settings import config_dir

    assert config_dir() == elsewhere
