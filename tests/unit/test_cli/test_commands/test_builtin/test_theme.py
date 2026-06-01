from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_cli import theme as theme_module
from agent_cli.commands.builtin.theme import CMD

from ..conftest import render_output


@pytest.fixture(autouse=True)
def isolated_prefs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    prefs = tmp_path / "cli-prefs.json"
    monkeypatch.setattr("agent_cli.runtime.prefs.PREFS_PATH", prefs)
    return prefs


async def test_no_arg_lists_available_themes() -> None:
    ctx = MagicMock()
    result = await CMD.handler(ctx, "")
    assert result.output is not None
    rendered = render_output(result.output)
    assert "Available themes:" in rendered
    assert theme_module.DEFAULT_THEME.name in rendered


async def test_unknown_theme_lists_available() -> None:
    ctx = MagicMock()
    result = await CMD.handler(ctx, "nonexistent-theme")
    assert result.output is not None
    rendered = render_output(result.output)
    assert "Unknown theme: nonexistent-theme" in rendered
    assert "Available:" in rendered


async def test_known_theme_persists_and_instructs_restart(
    isolated_prefs: Path,
) -> None:
    ctx = MagicMock()
    result = await CMD.handler(ctx, theme_module.DEFAULT_THEME.name)
    assert result.output is not None
    assert "Restart to apply" in render_output(result.output)
    assert isolated_prefs.exists()
    assert theme_module.load_saved_theme() is theme_module.DEFAULT_THEME


def test_theme_command_metadata() -> None:
    assert CMD.name == "/theme"
    assert "theme" in CMD.description.lower()
