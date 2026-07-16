"""Tests for /export — transcript dump to markdown."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_cli.commands.builtin.export import CMD, _format_message, _format_tool_group
from agent_harness.core.message import Attachment, Message, ToolCall, ToolResult

from ..conftest import render_output


def _ctx_with_messages(msgs: list[Message], session_id: str = "sess") -> MagicMock:
    agent = MagicMock()
    agent.context.short_term_memory.get_context_messages = AsyncMock(return_value=msgs)
    return MagicMock(agent=agent, session_id=session_id)


def test_format_message_user_content() -> None:
    out = _format_message(Message.user("hello world"))
    assert out.startswith("## User\n\nhello world")


def test_format_message_assistant_with_tool_calls() -> None:
    tcs = [
        ToolCall(id="c1", name="read_file", arguments={"file_path": "a.py"}),
        ToolCall(id="c2", name="list_dir", arguments={"path": "src"}),
    ]
    out = _format_message(Message.assistant(content="thinking", tool_calls=tcs))
    assert "## Assistant" in out
    assert "thinking" in out
    assert "**Tool calls:**" in out
    assert '`read_file({"file_path": "a.py"})`' in out
    assert '`list_dir({"path": "src"})`' in out


def test_format_message_system_role() -> None:
    out = _format_message(Message.system("rules"))
    assert out.startswith("## System")


def test_format_tool_group_resolves_name_from_prev_assistant() -> None:
    prev = Message.assistant(content="", tool_calls=[
        ToolCall(id="c1", name="read_file", arguments={}),
    ])
    tool_msg = Message.tool(tool_call_id="c1", content="file body", is_error=False)
    out = _format_tool_group(prev, [tool_msg])
    assert out.startswith("## Tool")
    assert "**`read_file`**:" in out
    assert "file body" in out


def test_format_tool_group_marks_errors() -> None:
    prev = Message.assistant(content="", tool_calls=[
        ToolCall(id="c1", name="read_file", arguments={}),
    ])
    tool_msg = Message.tool(tool_call_id="c1", content="boom", is_error=True)
    out = _format_tool_group(prev, [tool_msg])
    assert "**`read_file`** (error):" in out


def test_format_tool_group_falls_back_when_no_prev_assistant() -> None:
    tool_msg = Message.tool(tool_call_id="orphan", content="stray", is_error=False)
    out = _format_tool_group(None, [tool_msg])
    assert "## Tool" in out
    assert "stray" in out
    # no `**...**:` label since name lookup failed
    assert "**`" not in out


async def test_export_writes_file_and_reports_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    msgs = [
        Message.user("ping"),
        Message.assistant("pong"),
    ]
    ctx = _ctx_with_messages(msgs, session_id="abc")
    result = await CMD.handler(ctx, "")

    out_dir = tmp_path / ".arktor" / "sessions" / "abc" / "export"
    files = list(out_dir.glob("*.md"))
    assert len(files) == 1
    body = files[0].read_text(encoding="utf-8")
    assert "## User" in body and "ping" in body
    assert "## Assistant" in body and "pong" in body
    rendered = render_output(result.output)
    assert "Exported" in rendered
    # path shown is home-relative
    assert "~/.arktor/sessions/abc/export/" in rendered


async def test_export_groups_consecutive_tool_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    tcs = [
        ToolCall(id="c1", name="read_file", arguments={"file_path": "a"}),
        ToolCall(id="c2", name="read_file", arguments={"file_path": "b"}),
    ]
    msgs = [
        Message.user("look at @a and @b"),
        Message.assistant(content="", tool_calls=tcs),
        Message.tool(tool_call_id="c1", content="A content", is_error=False),
        Message.tool(tool_call_id="c2", content="B content", is_error=False),
        Message.assistant("done"),
    ]
    ctx = _ctx_with_messages(msgs, session_id="grp")
    await CMD.handler(ctx, "")

    out_dir = tmp_path / ".arktor" / "sessions" / "grp" / "export"
    body = next(out_dir.glob("*.md")).read_text()
    # one ## Tool block, both results inside
    assert body.count("## Tool") == 1
    assert "A content" in body and "B content" in body
    # both tool calls resolve to read_file name labels
    assert body.count("**`read_file`**:") == 2


# ── user-block canonicalization (peel attachment + envelope) ──


def _att_user_msg(trailing: str) -> Message:
    from agent_cli.render.notices import format_attachment_reminders

    tc = ToolCall(id="c1", name="read_file", arguments={"file_path": "a.py"})
    tr = ToolResult(tool_call_id="c1", content="[a.py] lines 1-1 of 1\nx")
    block = format_attachment_reminders(tc, tr)
    return Message.user(f"{block}\n\n{trailing}")


def test_export_user_block_strips_attachment_reminders() -> None:
    out = _format_message(_att_user_msg("@a.py explain"))
    assert "<system-reminder>" not in out
    assert out.endswith("@a.py explain\n")


def test_export_user_block_strips_drift_reminder() -> None:
    drift = (
        "<system-reminder>\nNote: the following files changed on disk since "
        "you last read them — ...\n\n- x.py (modified)\n</system-reminder>"
    )
    out = _format_message(Message.user(f"fix the bug\n\n{drift}"))
    assert "fix the bug" in out
    assert "changed on disk" not in out


def test_export_user_block_canonicalizes_shell_run_envelope() -> None:
    from agent_cli.runtime.shell import format_shell_run

    msg = Message.user(format_shell_run("echo hi", 0, "hi"))
    out = _format_message(msg)
    assert "<user-shell-run>" not in out
    assert "! echo hi" in out


def test_export_user_block_canonicalizes_skill_envelope() -> None:
    envelope = (
        "<system-reminder>The user has explicitly requested the foo "
        "skill. Apply the skill instructions below to address their "
        'request.</system-reminder>\n\n<skill-loaded name="foo">b'
        "</skill-loaded>"
    )
    out = _format_message(Message.user(envelope))
    assert "/foo" in out
    assert "<skill-loaded" not in out


def test_export_user_block_plain_text_unchanged() -> None:
    out = _format_message(Message.user("just a plain question"))
    assert out == "## User\n\njust a plain question\n"


def test_export_assistant_block_unchanged() -> None:
    out = _format_message(Message.assistant("plain answer"))
    assert out == "## Assistant\n\nplain answer\n"


def test_export_goal_start_and_continuation_are_canonical() -> None:
    from agent_cli.runtime.goal import mode as goal_mode

    start = _format_message(Message.user(goal_mode.make_start_input("ship it")))
    continuation = _format_message(goal_mode.make_continuation_message(
        _goal_agent("ship it"),
        "gap",
        "run tests",
    ))

    assert "/goal ship it" in start
    assert "initial worker turn" not in start
    assert "◎ goal · continuing" in continuation
    assert "ship it" not in continuation
    assert "run tests" not in continuation


def _goal_agent(objective: str) -> MagicMock:
    from agent_cli.runtime.goal import mode as goal_mode
    from agent_harness.context.context import AgentContext

    agent = MagicMock()
    agent.context = AgentContext()
    agent._session_metadata_extras = {}
    goal_mode.begin(agent, objective)
    return agent


def test_export_user_attachments_section() -> None:
    att = Attachment(digest="a" * 64, mime="image/png", filename="shot.png", size=1024)
    out = _format_message(Message.user("see this", attachments=[att]))
    assert "## User" in out
    assert "see this" in out
    assert "**Attachments:**" in out
    assert "shot.png (image/png, 1.0KB, sha256:" + ("a" * 12) + "…)" in out


def test_export_tool_attachments_section() -> None:
    prev = Message.assistant(content="", tool_calls=[
        ToolCall(id="c1", name="web_fetch", arguments={"url": "u"}),
    ])
    att = Attachment(
        digest="b" * 64, mime="application/pdf",
        filename="doc.pdf", size=2 * 1024 * 1024,
    )
    tool_msg = Message.tool(tool_call_id="c1", content="ok", attachments=[att])
    out = _format_tool_group(prev, [tool_msg])
    assert "**`web_fetch`**:" in out
    assert "**Attached media:**" in out
    assert "doc.pdf (application/pdf, 2.0MB, sha256:" + ("b" * 12) + "…)" in out
