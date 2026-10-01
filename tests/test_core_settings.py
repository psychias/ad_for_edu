"""Settings: closed keys, environment values without defaults, no absolute paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from ad_for_edu.core import settings
from ad_for_edu.core.errors import SettingsError


@dataclass(frozen=True)
class Inner:
    seconds: float
    label: str = "x"


@dataclass(frozen=True)
class Outer:
    name: str
    inner: Inner
    sizes: tuple[int, ...] = ()
    options: dict = field(default_factory=dict)


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_a_file_binds_to_its_dataclass(tmp_path):
    source = write(tmp_path / "s.yaml", "name: a\ninner:\n  seconds: 1.5\nsizes: [1, 2]\n")
    bound = settings.load_settings(source, Outer)
    assert bound == Outer("a", Inner(1.5), (1, 2))


def test_an_unknown_key_fails_and_lists_the_known_keys(tmp_path):
    source = write(tmp_path / "s.yaml", "name: a\ninner:\n  seconds: 1\n  labell: y\n")
    with pytest.raises(SettingsError) as caught:
        settings.load_settings(source, Outer)
    message = str(caught.value)
    assert "labell" in message and "label" in message and "seconds" in message


def test_a_missing_required_key_fails(tmp_path):
    source = write(tmp_path / "s.yaml", "inner:\n  seconds: 1\n")
    with pytest.raises(SettingsError) as caught:
        settings.load_settings(source, Outer)
    assert "name" in str(caught.value)


def test_a_variable_is_taken_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_PLACE", "materials")
    source = write(tmp_path / "s.yaml", "name: ${SOME_PLACE}/x\ninner:\n  seconds: 1\n")
    assert settings.load_settings(source, Outer).name == "materials/x"


def test_an_unset_variable_fails_and_names_the_variable(tmp_path):
    source = write(tmp_path / "s.yaml", "name: ${NOT_SET_ANYWHERE}\ninner:\n  seconds: 1\n")
    with pytest.raises(SettingsError) as caught:
        settings.load_settings(source, Outer)
    assert "NOT_SET_ANYWHERE" in str(caught.value)


def test_an_empty_variable_counts_as_unset(monkeypatch):
    monkeypatch.setenv("AD_FOR_EDU_DATA_DIR", "   ")
    with pytest.raises(SettingsError):
        settings.data_dir()


@pytest.mark.parametrize(
    "value",
    ["/srv/material", "C:/material", "D:" + chr(92) + "material", "~/material"],
)
def test_an_absolute_path_is_refused(tmp_path, value):
    source = tmp_path / "s.yaml"
    source.write_text(f"name: '{value}'\ninner:\n  seconds: 1\n", encoding="utf-8")
    with pytest.raises(SettingsError) as caught:
        settings.load_settings(source, Outer)
    assert "absolute path" in str(caught.value)


def test_an_absolute_path_is_found_inside_nested_values():
    with pytest.raises(SettingsError) as caught:
        settings.reject_absolute_paths({"a": [{"b": "/etc/x"}]}, where="file")
    assert "file.a[0].b" in str(caught.value)


def test_a_model_id_with_a_slash_is_not_a_path():
    settings.reject_absolute_paths({"model_id": "vendor/model-4b"}, where="file")


def test_a_section_is_selected_before_binding(tmp_path):
    source = write(tmp_path / "s.yaml", "one:\n  seconds: 2\ntwo:\n  seconds: 3\n")
    assert settings.load_settings(source, Inner, section="two") == Inner(3)
    with pytest.raises(SettingsError) as caught:
        settings.load_settings(source, Inner, section="three")
    assert "one" in str(caught.value) and "two" in str(caught.value)


def test_a_missing_file_is_an_error(tmp_path):
    with pytest.raises(SettingsError):
        settings.load_yaml(tmp_path / "absent.yaml")


def test_layers_merge_nested_mappings_and_replace_the_rest():
    merged = settings.merge(
        {"a": 1, "lora": {"r": 16, "alpha": 32}, "targets": ["q", "k"]},
        {"lora": {"r": 8}, "targets": ["v"]},
    )
    assert merged == {"a": 1, "lora": {"r": 8, "alpha": 32}, "targets": ["v"]}


def test_the_config_directory_can_be_named_by_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("AD_FOR_EDU_CONFIG_DIR", str(tmp_path))
    assert settings.config_dir() == tmp_path
