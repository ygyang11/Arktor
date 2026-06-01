import json
from pathlib import Path

import pytest

from agent_cli import theme as theme_module


@pytest.fixture(autouse=True)
def isolated_prefs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    prefs = tmp_path / "cli-prefs.json"
    monkeypatch.setattr("agent_cli.runtime.prefs.PREFS_PATH", prefs)
    return prefs


def test_load_saved_theme_returns_default_when_no_file() -> None:
    assert theme_module.load_saved_theme() is theme_module.DEFAULT_THEME


def test_load_saved_theme_returns_default_when_corrupt(isolated_prefs: Path) -> None:
    isolated_prefs.write_text("{{{ not json")
    assert theme_module.load_saved_theme() is theme_module.DEFAULT_THEME


def test_load_saved_theme_returns_default_when_unknown_name(
    isolated_prefs: Path,
) -> None:
    isolated_prefs.write_text(json.dumps({"theme": "ghost-theme"}))
    assert theme_module.load_saved_theme() is theme_module.DEFAULT_THEME


def test_save_theme_writes_prefs_file(isolated_prefs: Path) -> None:
    theme_module.save_theme(theme_module.DEFAULT_THEME.name)
    data = json.loads(isolated_prefs.read_text())
    assert data == {"theme": theme_module.DEFAULT_THEME.name}


def test_save_theme_rejects_unknown_name() -> None:
    with pytest.raises(KeyError, match="Unknown theme"):
        theme_module.save_theme("nonexistent")


def test_save_theme_preserves_other_prefs(isolated_prefs: Path) -> None:
    isolated_prefs.parent.mkdir(parents=True, exist_ok=True)
    isolated_prefs.write_text(json.dumps({"foo": "bar"}))
    theme_module.save_theme(theme_module.DEFAULT_THEME.name)
    data = json.loads(isolated_prefs.read_text())
    assert data["foo"] == "bar"
    assert data["theme"] == theme_module.DEFAULT_THEME.name


def test_available_names_sorted() -> None:
    names = theme_module.available_names()
    assert names == sorted(names)
    assert theme_module.DEFAULT_THEME.name in names


def test_every_registered_theme_round_trips(isolated_prefs: Path) -> None:
    for name in theme_module.available_names():
        theme_module.save_theme(name)
        assert theme_module.load_saved_theme() is theme_module.THEMES[name]
