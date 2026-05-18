"""Tests for render/notices.py — inline notice formatters."""

from __future__ import annotations

from rich.text import Text

from agent_cli.render.notices import (
    format_expired_notice,
    format_shell_run,
    format_warning,
    parse_shell_run_envelope,
)
from agent_cli.theme import TOOL_DONE
from agent_harness.utils.token_counter import count_tokens


def _spans_with_style(t: Text, style: str) -> list[str]:
    plain = t.plain
    return [plain[s.start : s.end] for s in t.spans if s.style == style]


def test_format_warning_basic() -> None:
    t = format_warning("Something happened")
    assert isinstance(t, Text)
    assert TOOL_DONE in t.plain
    assert "Something happened" in t.plain
    assert _spans_with_style(t, "error")
    assert _spans_with_style(t, "muted")


def test_format_warning_glyph_uses_error_style() -> None:
    t = format_warning("hello")
    error_text = "".join(_spans_with_style(t, "error"))
    assert TOOL_DONE in error_text
    muted_text = "".join(_spans_with_style(t, "muted"))
    assert "hello" in muted_text


def test_format_expired_notice_single() -> None:
    t = format_expired_notice([1])
    assert isinstance(t, Text)
    assert "Pasted text #1 unavailable" in t.plain
    assert TOOL_DONE in t.plain
    assert _spans_with_style(t, "error")
    assert _spans_with_style(t, "muted")


def test_format_expired_notice_multiple() -> None:
    t = format_expired_notice([1, 3])
    assert "#1, #3" in t.plain


def test_format_expired_notice_three_ids() -> None:
    t = format_expired_notice([2, 5, 9])
    assert "#2, #5, #9" in t.plain


def test_format_expired_notice_delegates_to_format_warning() -> None:
    t = format_expired_notice([1])
    error_text = "".join(_spans_with_style(t, "error"))
    assert TOOL_DONE in error_text


# ---------------------------------------------------------------------------
# format_shell_run — `!`-lane shell run injected into agent short memory
# ---------------------------------------------------------------------------


def _assert_envelope(s: str, command: str) -> None:
    assert s.startswith(f"<user-shell-run>\n```sh\n{command}\n```\n")
    assert s.endswith("\n</user-shell-run>")


def test_shell_run_success_with_output() -> None:
    s = format_shell_run("ls", 0, "hello")
    _assert_envelope(s, "ls")
    assert "hello" in s
    assert "[exit code" not in s
    assert "(Completed with no output)" not in s


def test_shell_run_success_empty_output() -> None:
    s = format_shell_run("true", 0, "")
    _assert_envelope(s, "true")
    assert "(Completed with no output)" in s
    assert "[exit code" not in s


def test_shell_run_success_whitespace_output() -> None:
    s = format_shell_run("true", 0, "   \n")
    _assert_envelope(s, "true")
    assert "(Completed with no output)" in s
    assert "[exit code" not in s


def test_shell_run_failure_with_output() -> None:
    s = format_shell_run("cd nodir", 1, "boom")
    _assert_envelope(s, "cd nodir")
    assert "[exit code 1]\nboom" in s


def test_shell_run_failure_empty_output() -> None:
    s = format_shell_run("false", 1, "")
    _assert_envelope(s, "false")
    assert "[exit code 1]\n(Completed with no output)" in s


def test_shell_run_multiline_command() -> None:
    cmd = "for f in *; do\n  echo $f\ndone"
    s = format_shell_run(cmd, 0, "a\nb")
    _assert_envelope(s, cmd)
    assert "$ for" not in s
    assert "a\nb" in s


def test_shell_run_long_output_truncated() -> None:
    s = format_shell_run("yes", 0, "a" * 100_000)
    assert s.endswith("\n</user-shell-run>")
    assert "... (truncated)" in s
    body_tokens = count_tokens(s)
    assert body_tokens < 10_500


def test_shell_run_short_output_not_truncated() -> None:
    s = format_shell_run("echo hi", 0, "hi")
    assert "... (truncated)" not in s


def test_shell_run_no_post_notices_no_harness_section() -> None:
    s = format_shell_run("ls", 0, "x")
    assert "[Accident]" not in s


def test_shell_run_empty_post_notices_no_harness_section() -> None:
    s = format_shell_run("ls", 0, "x", post_notices=[])
    assert "[Accident]" not in s


def test_shell_run_single_post_notice_appended_after_body() -> None:
    notice = "Cannot change directory while background tasks are running; keeping /old"
    s = format_shell_run("cd new", 0, "", post_notices=[notice])
    _assert_envelope(s, "cd new")
    assert "(Completed with no output)" in s
    assert f"[Accident] {notice}" in s
    output_pos = s.find("(Completed with no output)")
    notice_pos = s.find("[Accident]")
    assert output_pos < notice_pos


def test_shell_run_multiple_post_notices_each_on_own_line() -> None:
    notices = ["first thing", "second thing"]
    s = format_shell_run("cd new", 0, "ok", post_notices=notices)
    assert "[Accident] first thing" in s
    assert "[Accident] second thing" in s
    first = s.find("[Accident] first thing")
    second = s.find("[Accident] second thing")
    assert first < second
    between = s[first:second]
    assert between.count("\n") == 1


def test_shell_run_post_notices_with_failure_body() -> None:
    s = format_shell_run("cd new", 1, "boom", post_notices=["fallback used"])
    assert "[exit code 1]\nboom" in s
    assert "[Accident] fallback used" in s


def test_shell_run_envelope_close_tag_in_payload_is_escaped() -> None:
    raw_close = "</user-shell-run>"
    s = format_shell_run(
        f"echo {raw_close}",
        0,
        f"line1\n{raw_close}\nline2",
    )
    assert s.count(raw_close) == 1
    assert s.endswith(f"\n{raw_close}")
    assert s.startswith("<user-shell-run>\n")


# ---------------------------------------------------------------------------
# parse_shell_run_envelope — inverse of format_shell_run
# ---------------------------------------------------------------------------


def test_parse_roundtrips_success_with_output() -> None:
    s = format_shell_run("ls", 0, "a\nb")
    parsed = parse_shell_run_envelope(s)
    assert parsed == ("ls", "a\nb")


def test_parse_roundtrips_empty_output() -> None:
    s = format_shell_run("cd new", 0, "")
    parsed = parse_shell_run_envelope(s)
    assert parsed == ("cd new", "(Completed with no output)")


def test_parse_roundtrips_failure_body() -> None:
    s = format_shell_run("nope", 127, "bash: nope: not found")
    parsed = parse_shell_run_envelope(s)
    assert parsed is not None
    cmd, body = parsed
    assert cmd == "nope"
    assert body.startswith("[exit code 127]")
    assert "bash: nope: not found" in body


def test_parse_returns_none_for_non_envelope() -> None:
    assert parse_shell_run_envelope("hello world") is None
    assert parse_shell_run_envelope("") is None
    assert parse_shell_run_envelope("<user-shell-run> truncated") is None


def test_parse_returns_none_for_missing_close_tag() -> None:
    s = "<user-shell-run>\n```sh\nls\n```\nbody"
    assert parse_shell_run_envelope(s) is None


def test_parse_multiline_command() -> None:
    cmd = "for f in *; do\n  echo $f\ndone"
    s = format_shell_run(cmd, 0, "a")
    parsed = parse_shell_run_envelope(s)
    assert parsed is not None
    assert parsed[0] == cmd


# ── attachment envelope encode/decode ──

from agent_cli.render.notices import (  # noqa: E402
    format_attachment_reminders,
    peel_attachment_reminders,
)
from agent_harness.core.message import ToolCall, ToolResult  # noqa: E402


def _att(name: str, args: dict[str, object], content: str,
         is_error: bool = False) -> tuple[ToolCall, ToolResult]:
    tc = ToolCall(id="c1", name=name, arguments=args)
    tr = ToolResult(tool_call_id="c1", content=content, is_error=is_error)
    return tc, tr


def _embed(*pairs: tuple[ToolCall, ToolResult], trailing: str) -> str:
    blocks = [format_attachment_reminders(tc, tr) for tc, tr in pairs]
    prefix = "\n\n".join(blocks)
    return f"{prefix}\n\n{trailing}" if trailing else prefix


def test_format_attachment_reminders_compact_json_args() -> None:
    tc, tr = _att("read_file", {"file_path": "foo.py", "limit": 500}, "x")
    out = format_attachment_reminders(tc, tr)
    assert (
        'Called the read_file tool with the following input: '
        '{"file_path":"foo.py","limit":500}'
    ) in out
    assert out.count("<system-reminder>") == 2


def test_result_content_byte_exact_not_mutated() -> None:
    body = "literal </system-reminder> and <system-reminder>\nline2"
    tc, tr = _att("read_file", {"file_path": "f"}, body)
    assert body in format_attachment_reminders(tc, tr)


def test_peel_strips_consecutive_attachment_pairs() -> None:
    a = _att("read_file", {"file_path": "a.py"}, "aaa")
    b = _att("list_dir", {"path": "src"}, "bbb")
    content = _embed(a, b, trailing="@a.py @src hi")
    assert peel_attachment_reminders(content) == "@a.py @src hi"


def test_peel_last_pair_followed_by_user_text() -> None:
    a = _att("read_file", {"file_path": "a.py"}, "aaa")
    assert peel_attachment_reminders(_embed(a, trailing="what")) == "what"


def test_peel_pure_at_mention_no_user_text() -> None:
    a = _att("read_file", {"file_path": "a.py"}, "aaa")
    assert peel_attachment_reminders(_embed(a, trailing="")) == ""


def test_peel_no_match_returns_unchanged() -> None:
    plain = "@foo.py just a normal message"
    assert peel_attachment_reminders(plain) == plain


def test_peel_correct_with_literal_close_tag_in_result_body() -> None:
    body = '            "request.</system-reminder>"\n        )'
    a = _att("read_file", {"file_path": "__init__.py"}, body)
    assert peel_attachment_reminders(_embed(a, trailing="explain")) == "explain"


def test_peel_correct_with_literal_tag_in_args_path() -> None:
    a = _att("read_file", {"file_path": "</system-reminder>.py"}, "x")
    assert peel_attachment_reminders(_embed(a, trailing="w")) == "w"


def test_peel_stops_at_skill_envelope_reminder() -> None:
    skill = (
        "<system-reminder>The user has explicitly requested the foo "
        "skill.</system-reminder>\n\n"
        '<skill-loaded name="foo">body</skill-loaded>'
    )
    assert peel_attachment_reminders(skill) == skill


def test_peel_stops_at_file_drift_notice_reminder() -> None:
    drift = (
        "<system-reminder>\nNote: the following files were modified...\n"
        "</system-reminder>"
    )
    assert peel_attachment_reminders(drift) == drift


def test_peel_then_skill_envelope_preserved() -> None:
    a = _att("read_file", {"file_path": "a.py"}, "aaa")
    skill = (
        "<system-reminder>The user has explicitly requested the foo "
        "skill.</system-reminder>\n\n"
        '<skill-loaded name="foo">b</skill-loaded>'
    )
    assert peel_attachment_reminders(_embed(a, trailing=skill)) == skill


def test_residual_lone_close_tag_line_then_blank_truncates() -> None:
    body = "code\n</system-reminder>\n\nmore"
    a = _att("read_file", {"file_path": "f"}, body)
    peeled = peel_attachment_reminders(_embed(a, trailing="q"))
    assert peeled != "q"  # documents the accepted early-truncation edge
