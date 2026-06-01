"""Tests for render/notices.py — inline notice formatters."""

from __future__ import annotations

from rich.text import Text

from agent_cli.render.notices import (
    format_expired_notice,
    format_warning,
    peel_drift_reminder,
    peel_reminders,
)
from agent_cli.theme import TOOL_DONE


def _spans_with_style(t: Text, style: str) -> list[str]:
    plain = t.plain
    return [plain[s.start : s.end] for s in t.spans if s.style == style]


def test_peel_drift_reminder_strips_trailing_block() -> None:
    notice = (
        "<system-reminder>\nNote: the following files changed on disk since "
        "you last read them — ...\n\n- src/x.py (modified)\n</system-reminder>"
    )
    content = f"fix the parser\n\n{notice}"
    assert peel_drift_reminder(content) == "fix the parser"


def test_peel_drift_reminder_noop_without_block() -> None:
    assert peel_drift_reminder("just a plain message") == "just a plain message"


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


def test_peel_reminders_strips_attachment_and_drift() -> None:
    a = _att("read_file", {"file_path": "a.py"}, "x")
    drift = (
        "<system-reminder>\nNote: the following files changed on disk since "
        "you last read them — ...\n\n- x.py (modified)\n</system-reminder>"
    )
    content = _embed(a, trailing=f"fix the bug\n\n{drift}")
    assert peel_reminders(content) == "fix the bug"


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
