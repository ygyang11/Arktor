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


# ── envelope unwrapping via peel_user_command ────────────────────────


def test_preview_init_envelope_collapses_to_slash_init() -> None:
    from agent_cli.commands.builtin.init import _INIT_NEW

    raw = _INIT_NEW.format(focus="\n\nFocus: auth tests")
    assert _format_session_preview(raw) == "/init auth tests"


def test_preview_review_envelope_collapses_to_slash_review() -> None:
    from agent_cli.commands.builtin.review import _REVIEW_PROMPT

    raw = _REVIEW_PROMPT.format(target="src/agent_cli")
    assert _format_session_preview(raw) == "/review src/agent_cli"


def test_preview_skill_envelope_collapses_to_slash_command() -> None:
    raw = (
        "look at history\n\n"
        "<system-reminder>The user has explicitly requested the web-search "
        "skill. Apply the skill instructions below to address their "
        "request.</system-reminder>\n\n"
        '<skill-loaded name="web-search">\nbody\n</skill-loaded>'
    )
    assert _format_session_preview(raw) == "/web-search look at history"


def test_preview_skill_envelope_no_args() -> None:
    raw = (
        "<system-reminder>The user has explicitly requested the humanizer "
        "skill. Apply the skill instructions below to address their "
        "request.</system-reminder>\n\n"
        '<skill-loaded name="humanizer">\nbody\n</skill-loaded>'
    )
    assert _format_session_preview(raw) == "/humanizer"


# ── home_relative_path ───────────────────────────────────────────────


def test_home_relative_path_strips_home() -> None:
    import os
    from pathlib import Path
    from agent_cli.commands.ui import home_relative_path

    home = Path.home()
    p = home / ".agent-harness" / "exports" / "file.md"
    assert home_relative_path(p) == "~/.agent-harness/exports/file.md"
    assert home_relative_path(str(p)) == "~/.agent-harness/exports/file.md"


def test_home_relative_path_absolute_when_outside_home() -> None:
    from agent_cli.commands.ui import home_relative_path

    assert home_relative_path("/tmp/x.md") == "/tmp/x.md"
