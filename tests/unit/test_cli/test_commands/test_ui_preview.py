"""Tests for the session-list preview formatter (CLI-layer transform)."""
from __future__ import annotations

from agent_cli.commands.ui import _format_session_preview
from agent_cli.render.notices import format_shell_run


def test_preview_plain_text_returns_collapsed() -> None:
    assert _format_session_preview("hello world") == "hello world"


def test_preview_collapses_whitespace() -> None:
    raw = "line1\n\nline2\twith   spaces"
    assert _format_session_preview(raw) == "line1 line2 with spaces"


def test_preview_truncates_to_limit() -> None:
    raw = "a" * 200
    result = _format_session_preview(raw)
    assert len(result) == 60
    assert result == "a" * 60


def test_preview_shell_run_returns_bang_form() -> None:
    raw = format_shell_run("cd Agent-Harness", 0, "")
    assert _format_session_preview(raw) == "! cd Agent-Harness"


def test_preview_shell_run_with_output_still_shows_command() -> None:
    raw = format_shell_run("ls", 0, "a\nb\nc")
    assert _format_session_preview(raw) == "! ls"


def test_preview_shell_run_failure_shows_command() -> None:
    raw = format_shell_run("nope", 127, "not found")
    assert _format_session_preview(raw) == "! nope"


def test_preview_shell_run_long_command_truncated() -> None:
    long_cmd = "echo " + "x" * 200
    raw = format_shell_run(long_cmd, 0, "")
    result = _format_session_preview(raw)
    assert len(result) == 60
    assert result.startswith("! echo ")


def test_preview_empty_string_returns_empty() -> None:
    assert _format_session_preview("") == ""
