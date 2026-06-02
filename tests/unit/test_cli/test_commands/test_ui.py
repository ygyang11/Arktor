"""Tests for the session-list preview formatter (CLI-layer transform)."""
from __future__ import annotations

from agent_cli.commands.ui import _format_session_preview
from agent_cli.runtime.shell import format_shell_run


def test_preview_plain_text_returns_collapsed() -> None:
    assert _format_session_preview("hello world") == "hello world"


def test_preview_collapses_whitespace() -> None:
    raw = "line1\n\nline2\twith   spaces"
    assert _format_session_preview(raw) == "line1 line2 with spaces"


def test_preview_truncates_long_prose() -> None:
    from agent_harness.utils.token_counter import count_tokens

    raw = ("hello world this is a long preview sentence " * 30).strip()
    result = _format_session_preview(raw)
    assert result.endswith("…")
    assert count_tokens(result) <= 30
    assert count_tokens(raw) > 30  # sanity: input exceeded budget


def test_preview_shell_run_returns_bang_form() -> None:
    raw = format_shell_run("cd Arktor", 0, "")
    assert _format_session_preview(raw) == "! cd Arktor"


def test_preview_shell_run_with_output_still_shows_command() -> None:
    raw = format_shell_run("ls", 0, "a\nb\nc")
    assert _format_session_preview(raw) == "! ls"


def test_preview_shell_run_failure_shows_command() -> None:
    raw = format_shell_run("nope", 127, "not found")
    assert _format_session_preview(raw) == "! nope"


def test_preview_shell_run_long_command_truncated() -> None:
    from agent_harness.utils.token_counter import count_tokens

    long_cmd = "echo " + " ".join(f"word{i}" for i in range(80))
    raw = format_shell_run(long_cmd, 0, "")
    result = _format_session_preview(raw)
    assert result.startswith("! echo ")
    assert result.endswith("…")
    assert count_tokens(result) <= 30


def test_preview_empty_string_returns_empty() -> None:
    assert _format_session_preview("") == ""


def test_preview_strips_drift_reminder() -> None:
    drift = (
        "<system-reminder>\nNote: the following files changed on disk since "
        "you last read them — ...\n\n- x.py (modified)\n</system-reminder>"
    )
    assert _format_session_preview(f"fix the bug\n\n{drift}") == "fix the bug"


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
    p = home / ".arktor" / "exports" / "file.md"
    assert home_relative_path(p) == "~/.arktor/exports/file.md"
    assert home_relative_path(str(p)) == "~/.arktor/exports/file.md"


def test_home_relative_path_absolute_when_outside_home() -> None:
    from agent_cli.commands.ui import home_relative_path

    assert home_relative_path("/tmp/x.md") == "/tmp/x.md"
