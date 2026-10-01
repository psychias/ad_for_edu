"""The spend gate, write-once outputs and run records."""

from __future__ import annotations

import io
import json

import pytest

from ad_for_edu.core import outputs, runs, secrets, spend
from ad_for_edu.core.errors import (
    ContractError,
    MissingSourceError,
    OutputExistsError,
    SettingsError,
    SpendNotApprovedError,
)
from ad_for_edu.core.io import append_jsonl, index_by, read_jsonl, write_jsonl


def estimate() -> spend.SpendEstimate:
    est = spend.SpendEstimate("references", free_columns=("chrF",))
    est.add("lec-a", "writer-x", 10, 0.02)
    est.add("lec-a", "writer-y", 10, 0.05, measured=False)
    return est


# ---------------------------------------------------------------- spend
def test_without_approval_the_estimate_is_shown_and_the_run_stops_successfully():
    stream = io.StringIO()
    with pytest.raises(SystemExit) as stopped:
        spend.require_approval(estimate(), approved=False, out=stream)
    assert stopped.value.code == 0
    shown = stream.getvalue()
    assert "writer-x" in shown and "writer-y" in shown and "chrF" in shown
    assert "nothing was called" in shown


def test_with_approval_a_token_is_issued():
    stream = io.StringIO()
    token = spend.require_approval(estimate(), approved=True, out=stream)
    assert isinstance(token, spend.SpendApproval)
    assert token.calls == 20 and token.total == pytest.approx(0.70)


def test_a_token_cannot_be_made_by_hand():
    with pytest.raises(SpendNotApprovedError):
        spend.SpendApproval(object(), "references", 1.0, 1)


def test_the_estimate_adds_up_per_line():
    est = estimate()
    assert est.calls == 20
    assert est.total == pytest.approx(10 * 0.02 + 10 * 0.05)
    assert not est.all_measured
    assert "assumed" in est.render()


def test_a_local_stage_has_a_free_token():
    token = spend.free_approval("local rating")
    assert token.total == 0.0 and token.calls == 0


# ---------------------------------------------------------------- outputs
def test_an_existing_output_is_never_replaced(tmp_path):
    target = tmp_path / "out" / "rows.jsonl"
    assert outputs.guard_new(target) == target
    write_jsonl(target, [{"a": 1}])
    with pytest.raises(OutputExistsError):
        outputs.guard_new(target)
    assert read_jsonl(target) == [{"a": 1}]


def test_an_empty_file_or_directory_may_be_written(tmp_path):
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert outputs.guard_new(empty_file) == empty_file
    assert outputs.guard_new(empty_dir) == empty_dir


def test_overwriting_must_be_asked_for(tmp_path):
    target = tmp_path / "rows.jsonl"
    write_jsonl(target, [{"a": 1}])
    assert outputs.guard_new(target, overwrite=True) == target


def test_a_failed_row_is_asked_again_on_resume(tmp_path):
    target = tmp_path / "rows.jsonl"
    append_jsonl(target, {"id": "m1", "emit": True})
    append_jsonl(target, {"id": "m2", "error": "timeout"})
    append_jsonl(target, {"id": "m3", "emit": False})
    assert outputs.completed_keys(target, lambda row: row["id"]) == {"m1", "m3"}


def test_an_unparsed_row_is_asked_again_on_resume(tmp_path):
    target = tmp_path / "rows.jsonl"
    append_jsonl(target, {"id": "m1", "winner": "a"})
    append_jsonl(target, {"id": "m2", "winner": None})
    done = outputs.completed_keys(
        target, lambda row: row["id"], is_complete=lambda row: row["winner"] is not None
    )
    assert done == {"m1"}


def test_no_file_means_nothing_is_done(tmp_path):
    assert outputs.completed_keys(tmp_path / "absent.jsonl", lambda row: row["id"]) == set()


# ---------------------------------------------------------------- io
def test_a_repeated_key_is_an_error(tmp_path):
    with pytest.raises(ContractError):
        index_by([{"id": 1}, {"id": 1}], "id")
    with pytest.raises(ContractError):
        index_by([{"other": 1}], "id")


def test_a_broken_line_names_its_position(tmp_path):
    target = tmp_path / "rows.jsonl"
    target.write_text('{"a": 1}\n\n{"a": \n', encoding="utf-8")
    with pytest.raises(ContractError) as caught:
        read_jsonl(target)
    assert ":3:" in str(caught.value)


def test_a_missing_row_file_is_an_error(tmp_path):
    with pytest.raises(MissingSourceError):
        read_jsonl(tmp_path / "absent.jsonl")


# ---------------------------------------------------------------- runs
def test_each_seed_has_its_own_directory(tmp_path):
    first = runs.RunKey("model-4b", "text_only", "sft", 0).directory(tmp_path)
    second = runs.RunKey("model-4b", "text_only", "sft", 1).directory(tmp_path)
    assert first != second
    assert first.name == "seed0" and first.parent.name == "sft"
    assert first.parent.parent.name == "model-4b-text_only"


def test_a_smoke_run_never_shares_a_directory_with_a_real_one(tmp_path):
    key = runs.RunKey("model-4b", "text_only", "sft", 0)
    assert key.directory(tmp_path, smoke=True) != key.directory(tmp_path)


@pytest.mark.parametrize(
    "fields",
    [
        {"backbone": "vendor/model"},
        {"backbone": ""},
        {"arm": "a" + chr(92) + "b"},
        {"seed": -1},
        {"seed": True},
        {"seed": 1.0},
    ],
)
def test_a_run_key_is_validated(fields):
    values = {"backbone": "model-4b", "arm": "text_only", "method": "sft", "seed": 0}
    values.update(fields)
    with pytest.raises(ContractError):
        runs.RunKey(**values)


def test_the_record_is_written_at_start_and_completed_at_the_end(tmp_path):
    key = runs.RunKey("model-4b", "text_only", "dpo", 2)
    settings = {"model_id": "vendor/model-4b", "revision": "abc", "lr": 1.5e-5}
    runs.write_record(tmp_path, key, settings)
    started = runs.read_record(tmp_path)
    assert started["status"] == "running" and started["ended_at"] is None
    runs.write_record(tmp_path, key, settings, status="done", extra={"steps": 686})
    ended = runs.read_record(tmp_path)
    assert ended["status"] == "done" and ended["steps"] == 686
    assert ended["started_at"] == started["started_at"]
    assert ended["wall_seconds"] is not None
    assert ended["method"] == "dpo" and ended["seed"] == 2


def test_the_record_names_no_machine_or_person(tmp_path):
    key = runs.RunKey("model-4b", "text_only", "sft", 0)
    runs.write_record(tmp_path, key, {"model_id": "vendor/model-4b"})
    record = json.loads((tmp_path / runs.RECORD_NAME).read_text(encoding="utf-8"))
    assert not {"hostname", "user", "git_sha", "platform"} & set(record)


def test_the_settings_digest_ignores_key_order():
    assert runs.settings_digest({"a": 1, "b": 2}) == runs.settings_digest({"b": 2, "a": 1})
    assert runs.settings_digest({"a": 1}) != runs.settings_digest({"a": 2})


def test_a_directory_without_a_record_is_not_a_run(tmp_path):
    with pytest.raises(MissingSourceError):
        runs.read_record(tmp_path)
    with pytest.raises(MissingSourceError):
        runs.update_record(tmp_path, anything=1)


def test_the_latest_checkpoint_is_the_highest_number(tmp_path):
    for number in (100, 50, 1000, 200):
        (tmp_path / f"checkpoint-{number}").mkdir()
    (tmp_path / "checkpoint-final").mkdir()
    assert runs.latest_checkpoint(tmp_path).name == "checkpoint-1000"
    assert runs.latest_checkpoint(tmp_path / "checkpoint-50") is None


# ---------------------------------------------------------------- secrets
def test_a_missing_credential_names_its_variable():
    with pytest.raises(SettingsError) as caught:
        secrets.credential(secrets.GEMINI)
    assert secrets.GEMINI in str(caught.value)


def test_the_env_file_fills_only_what_is_unset(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nA_TEST_KEY='from-file'\nB_TEST_KEY=other\n\nbroken line\n", "utf-8")
    monkeypatch.setenv("B_TEST_KEY", "from-environment")
    monkeypatch.delenv("A_TEST_KEY", raising=False)
    assert secrets.load_env_file(env) == 1
    assert secrets.credential("A_TEST_KEY") == "from-file"
    assert secrets.credential("B_TEST_KEY") == "from-environment"
