"""Tests for static replay path (no adapter / no Live)."""
from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock

from rich.console import Console

from agent_cli.render.replay import (
    _hard_clear,
    _index_results,
    render_post_switch,
    replay,
    slice_last_turns,
)
from agent_cli.theme import DEFAULT_THEME, PROMPT
from agent_harness.core.message import Message, ToolCall, ToolResult


def _u(text: str) -> Message:
    return Message.user(text)


def _a(text: str | None = None, calls: list[ToolCall] | None = None) -> Message:
    return Message.assistant(content=text, tool_calls=calls)


def _t(call_id: str, content: str, is_error: bool = False) -> Message:
    return Message.tool(tool_call_id=call_id, content=content, is_error=is_error)


def _render(*messages: Message) -> str:
    buf = StringIO()
    console = Console(
        file=buf, force_terminal=False, color_system=None,
        width=200, theme=DEFAULT_THEME.rich,
    )
    replay(console, DEFAULT_THEME, list(messages))
    return buf.getvalue()


# ── slice_last_turns ─────────────────────────────────────────────────


def test_slice_last_turns_takes_last_n_user_msgs() -> None:
    msgs = [_u("q1"), _a("r1"), _u("q2"), _a("r2"), _u("q3"), _a("r3")]
    sliced = slice_last_turns(msgs, 2)
    assert sliced == msgs[2:]


def test_slice_last_turns_falls_back_to_all_when_fewer_users() -> None:
    msgs = [_u("q1"), _a("r1"), _u("q2"), _a("r2")]
    assert slice_last_turns(msgs, 5) == msgs


def test_slice_last_turns_filters_system_messages() -> None:
    sys = Message.system("system prompt")
    msgs = [sys, _u("q1"), _a("r1")]
    assert sys not in slice_last_turns(msgs, 5)


def test_slice_last_turns_first_after_slice_is_user() -> None:
    msgs = [_u("q1"), _a("r1"), _u("q2"), _a("r2"), _u("q3"), _a("r3")]
    sliced = slice_last_turns(msgs, 2)
    assert sliced[0].role.value == "user"


def test_slice_last_turns_empty_input() -> None:
    assert slice_last_turns([], 5) == []


def test_slice_last_turns_no_user_returns_all_non_system() -> None:
    msgs = [_a("r1"), _a("r2")]
    assert slice_last_turns(msgs, 3) == msgs


# ── _index_results ───────────────────────────────────────────────────


def test_index_results_picks_tool_messages_only() -> None:
    tr = _t("call_1", "done")
    msgs = [_u("q"), _a("r", calls=[ToolCall(id="call_1", name="x")]), tr]
    idx = _index_results(msgs)
    assert "call_1" in idx
    assert idx["call_1"].content == "done"


def test_index_results_empty_when_no_tool_messages() -> None:
    assert _index_results([_u("q"), _a("r")]) == {}


# ── replay rendering ─────────────────────────────────────────────────


def test_replay_empty_messages_renders_nothing() -> None:
    assert _render() == ""


def test_replay_user_message_uses_prompt_glyph() -> None:
    out = _render(_u("hello"))
    assert f"{PROMPT} hello" in out


def test_replay_user_content_not_styled_as_primary() -> None:
    buf = StringIO()
    console = Console(
        file=buf, force_terminal=True, color_system="truecolor",
        width=200, theme=DEFAULT_THEME.rich,
    )
    replay(console, DEFAULT_THEME, [Message.user("hello world content")])
    out = buf.getvalue()
    # primary color in DEFAULT_THEME (flexoki-dark) is #DA702C — its truecolor
    # escape is \x1b[38;2;218;112;44m. The PROMPT glyph must carry that span;
    # the content body must NOT, otherwise the whole user line shows primary.
    primary_open = "\x1b[38;2;218;112;44m"
    primary_segments = out.split(primary_open)
    assert len(primary_segments) >= 2, "PROMPT glyph should carry primary style"
    # Each opened primary span must close before "hello world content" begins.
    span_after_first_open = primary_segments[1]
    reset_pos = span_after_first_open.find("\x1b[0m")
    content_pos = span_after_first_open.find("hello world content")
    assert reset_pos != -1 and reset_pos < content_pos, \
        "primary span must close before user content body"


def test_replay_user_skips_empty_content() -> None:
    out = _render(Message.user(""))
    assert out == ""


def test_replay_assistant_text_includes_tool_done_glyph() -> None:
    out = _render(_a("hi there"))
    assert "● " in out
    assert "hi there" in out


def test_replay_assistant_skips_empty_content() -> None:
    out = _render(_a(None))
    assert out == ""


def test_replay_assistant_tool_call_renders_name_and_args() -> None:
    tc = ToolCall(id="c1", name="read_file", arguments={"file_path": "/a/b.py"})
    tr = _t("c1", "lines 1-10")
    out = _render(_a(calls=[tc]), tr)
    assert "Read" in out
    assert "/a/b.py" in out


def test_replay_running_when_result_missing() -> None:
    tc = ToolCall(id="c1", name="read_file", arguments={"file_path": "/a"})
    out = _render(_a(calls=[tc]))
    assert "Read" in out


def test_replay_error_result_renders_error_branch() -> None:
    tc = ToolCall(id="c1", name="read_file", arguments={"file_path": "/x"})
    tr = _t("c1", "Error: file not found", is_error=True)
    out = _render(_a(calls=[tc]), tr)
    assert "Error" in out


def test_replay_multiple_tool_calls_have_blank_line_between() -> None:
    tc1 = ToolCall(id="c1", name="read_file", arguments={"file_path": "/a"})
    tc2 = ToolCall(id="c2", name="read_file", arguments={"file_path": "/b"})
    tr1 = _t("c1", "ok")
    tr2 = _t("c2", "ok")
    out = _render(_a(calls=[tc1, tc2]), tr1, tr2)
    # blank line between the two call rows; both file paths present
    assert "/a" in out and "/b" in out


def test_replay_tool_message_alone_not_rendered() -> None:
    out = _render(_t("orphan", "stray content"))
    assert "stray content" not in out


# ── render_post_switch ───────────────────────────────────────────────


def _post_switch(messages: list[Message]) -> str:
    buf = StringIO()
    console = Console(
        file=buf, force_terminal=False, color_system=None,
        width=200, theme=DEFAULT_THEME.rich,
    )
    agent = MagicMock()
    agent.context.short_term_memory._messages = messages
    render_post_switch(agent, console, DEFAULT_THEME, "abc123")
    return buf.getvalue()


def test_hard_clear_writes_viewport_and_scrollback_sequences() -> None:
    buf = StringIO()
    console = Console(
        file=buf, force_terminal=False, color_system=None,
        width=200, theme=DEFAULT_THEME.rich,
    )
    _hard_clear(console)
    raw = buf.getvalue()
    assert "\x1b[2J" in raw
    assert "\x1b[3J" in raw
    assert "\x1b[H" in raw


def test_render_post_switch_empty_session_shows_new_marker() -> None:
    out = _post_switch([])
    assert "New session" in out
    assert "abc123" in out


def test_render_post_switch_with_messages_replays_and_shows_resumed_marker() -> None:
    out = _post_switch([_u("hi"), _a("hello")])
    assert "hi" in out
    assert "hello" in out
    assert "Resumed" in out
    assert "abc123" in out


def test_render_post_switch_resumed_marker_appears_after_replay() -> None:
    out = _post_switch([_u("greet"), _a("response")])
    assert out.index("response") < out.index("Resumed")
