"""Tests for /export — transcript dump to markdown."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_cli.commands.builtin.export import CMD, _format_message, _format_tool_group
from agent_harness.core.message import Message, ToolCall, ToolResult

from .conftest import render_output


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

    out_dir = tmp_path / ".agent-harness" / "sessions" / "abc"
    files = list(out_dir.glob("export-*.md"))
    assert len(files) == 1
    body = files[0].read_text(encoding="utf-8")
    assert "## User" in body and "ping" in body
    assert "## Assistant" in body and "pong" in body
    rendered = render_output(result.output)
    assert "Exported" in rendered
    # path shown is home-relative
    assert "~/.agent-harness/sessions/abc/export-" in rendered


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

    out_dir = tmp_path / ".agent-harness" / "sessions" / "grp"
    body = next(out_dir.glob("export-*.md")).read_text()
    # one ## Tool block, both results inside
    assert body.count("## Tool") == 1
    assert "A content" in body and "B content" in body
    # both tool calls resolve to read_file name labels
    assert body.count("**`read_file`**:") == 2
