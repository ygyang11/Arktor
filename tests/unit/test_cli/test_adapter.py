import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_cli.adapter import CliAdapter
from agent_cli.render import status_lines as status_lines_mod
from agent_cli.theme import FLEXOKI_DARK
from agent_harness.core.errors import LLMConnectionError
from agent_harness.llm.types import LLMRetryInfo


def _adapter() -> CliAdapter:
    a = CliAdapter(MagicMock(), FLEXOKI_DARK)
    a.markdown = MagicMock()
    a.tool_display = MagicMock()
    return a


@pytest.fixture
def fast_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(status_lines_mod, "THINKING_DEBOUNCE_S", 0.0)
    monkeypatch.setattr(status_lines_mod, "THINKING_TICK_S", 0.005)


async def test_stream_delta_enters_markdown_phase() -> None:
    a = _adapter()
    await a.on_stream_delta("hi")
    assert a._phase == "markdown"
    a.markdown.update.assert_called_once_with("hi")


async def test_first_tool_call_transitions_markdown_to_tools() -> None:
    a = _adapter()
    await a.on_stream_delta("let me...")
    tc = MagicMock(id="t1")
    tc.name = "read_file"
    await a.on_tool_call(tc)
    assert a._phase == "tools"
    a.markdown.finalize.assert_called_once()
    a.tool_display.add_call.assert_called_once()


async def test_on_tool_call_suppressed_finalizes_markdown_not_aborts() -> None:
    a = _adapter()
    await a.on_stream_delta("hello ")
    assert a._phase == "markdown"
    tc = MagicMock(id="t1")
    tc.name = "sub_agent"
    await a.on_tool_call(tc)
    a.markdown.finalize.assert_called_once()
    a.markdown.abort.assert_not_called()
    a.tool_display.add_call.assert_not_called()
    assert a._phase == "none"


async def test_on_tool_call_suppressed_in_tools_phase_ends_tool_display() -> None:
    a = _adapter()
    tc1 = MagicMock(id="t1")
    tc1.name = "read_file"
    await a.on_tool_call(tc1)
    assert a._phase == "tools"
    tc2 = MagicMock(id="t2")
    tc2.name = "sub_agent"
    await a.on_tool_call(tc2)
    a.tool_display.end.assert_called_once()
    a.markdown.finalize.assert_not_called()
    a.markdown.abort.assert_not_called()
    assert a._phase == "none"


async def test_on_tool_call_suppressed_in_none_phase_is_noop() -> None:
    a = _adapter()
    tc = MagicMock(id="t1")
    tc.name = "sub_agent"
    await a.on_tool_call(tc)
    a.markdown.finalize.assert_not_called()
    a.markdown.abort.assert_not_called()
    a.tool_display.end.assert_not_called()
    assert a._phase == "none"


async def test_on_tool_call_todo_write_also_finalizes_markdown() -> None:
    a = _adapter()
    await a.on_stream_delta("ok ")
    tc = MagicMock(id="t1")
    tc.name = "todo_write"
    await a.on_tool_call(tc)
    a.markdown.finalize.assert_called_once()
    a.markdown.abort.assert_not_called()
    assert a._phase == "none"


async def test_suppressed_tool_promotes_live_tail_text_to_scrollback() -> None:
    import io as _io

    from rich.console import Console as _Console

    from agent_cli.render.markdown_stream import MarkdownStream

    buf = _io.StringIO()
    con = _Console(
        file=buf,
        force_terminal=True,
        color_system="truecolor",
        width=80,
        theme=FLEXOKI_DARK.rich,
    )
    a = CliAdapter(con, FLEXOKI_DARK)
    a.markdown = MarkdownStream(con, FLEXOKI_DARK)
    a.tool_display = MagicMock()

    await a.on_stream_delta("好的，再来 2 个！")
    tc = MagicMock(id="t1")
    tc.name = "sub_agent"
    await a.on_tool_call(tc)

    assert a._phase == "none"
    assert "好的，再来 2 个" in buf.getvalue(), (
        "finalize() path must promote Live tail into scrollback; "
        "a regression to abort() would drop this line"
    )


async def test_denied_tool_id_skips_on_tool_result() -> None:
    a = _adapter()
    denied = MagicMock(tool_call_id="t1", tool_name="x", reason="nope")
    await a.on_tool_denied(denied)
    assert "t1" in a._denied_ids
    assert a._phase == "tools"
    a.tool_display.add_denied.assert_called_once()

    result = MagicMock(tool_call_id="t1", is_error=True, content="x")
    await a.on_tool_result(result)
    a.tool_display.mark_result.assert_not_called()


async def test_end_step_closes_tool_live_and_flushes_todo() -> None:
    a = _adapter()
    tc = MagicMock(id="t1")
    tc.name = "read_file"
    await a.on_tool_call(tc)
    a.queue_todo([{"content": "a", "status": "pending"}], {"total": 1, "completed": 0})
    await a.end_step()
    a.tool_display.end.assert_called_once()
    a.tool_display.show_todos.assert_called_once()
    assert a._phase == "none"
    assert a._denied_ids == set()
    assert a._pending_todo is None


async def test_end_step_idempotent() -> None:
    a = _adapter()
    await a.end_step()
    await a.end_step()
    assert a._phase == "none"


async def test_end_run_omits_summary_below_one_minute() -> None:
    import time as _time
    a = _adapter()
    a.print_inline = AsyncMock()
    a.begin_run()
    a._run_started = _time.monotonic() - 59
    await a.end_run()
    a.print_inline.assert_not_called()
    assert a._run_started is None


async def test_end_run_prints_worked_for_summary_at_one_minute() -> None:
    import time as _time
    a = _adapter()
    a.print_inline = AsyncMock()
    a.begin_run()
    a._run_started = _time.monotonic() - 60
    await a.end_run()
    a.print_inline.assert_awaited_once()
    rendered = a.print_inline.await_args.args[0]
    assert "Worked for 1m 0s" in rendered
    assert "✻" in rendered
    assert a._run_started is None


async def test_end_run_format_uses_minutes() -> None:
    import time as _time
    a = _adapter()
    a.print_inline = AsyncMock()
    a._run_started = _time.monotonic() - 135
    await a.end_run()
    rendered = a.print_inline.await_args.args[0]
    assert "Worked for 2m 15s" in rendered


async def test_end_run_omits_summary_when_no_begin_run() -> None:
    a = _adapter()
    a.print_inline = AsyncMock()
    await a.end_run()
    a.print_inline.assert_not_called()


def test_fmt_duration_format_per_magnitude() -> None:
    from agent_cli.render.status_lines import fmt_duration
    assert fmt_duration(0) == "0s"
    assert fmt_duration(59) == "59s"
    assert fmt_duration(60) == "1m 0s"
    assert fmt_duration(3599) == "59m 59s"
    assert fmt_duration(3600) == "1h 0m"
    assert fmt_duration(3661) == "1h 1m"
    assert fmt_duration(86399) == "23h 59m"
    assert fmt_duration(86400) == "1d 0h"
    assert fmt_duration(90061) == "1d 1h"


async def test_end_step_closes_markdown_phase() -> None:
    a = _adapter()
    await a.on_stream_delta("hi")
    await a.end_step()
    a.markdown.finalize.assert_called_once()
    assert a._phase == "none"


async def test_pause_for_stdin_closes_active_live() -> None:
    a = _adapter()
    tc = MagicMock(id="t1")
    tc.name = "read_file"
    await a.on_tool_call(tc)
    await a.pause_for_stdin()
    a.tool_display.pause.assert_called_once()

    a2 = _adapter()
    await a2.on_stream_delta("x")
    await a2.pause_for_stdin()
    a2.markdown.pause.assert_called_once()


async def test_thinking_debounce_cancels_before_visible(
    fast_thinking: None,
) -> None:
    # Even with 0 debounce, immediate stop should cancel before the first tick
    # has a chance to call print. We call on_llm_call + stop back-to-back.
    a = _adapter()
    await a.on_llm_call()
    await a._thinking_line.stop()
    assert a._thinking_line.visible is False
    # console.print may or may not have been called once depending on scheduling;
    # what matters is that after stop, visible is False and no task is running.
    assert a._thinking_line.task is None


async def test_thinking_becomes_visible_after_debounce(
    fast_thinking: None,
) -> None:
    from agent_cli.render.status_lines import _THINKING_WORDS
    from agent_cli.theme import ELLIPSIS_FRAMES, SPINNER_FRAMES

    a = _adapter()
    await a.on_llm_call()
    # Let the loop fire at least once
    await asyncio.sleep(0.02)
    assert a._thinking_line.visible is True
    # Thinking bypasses Rich and writes to console.file directly.
    written = "".join(
        call.args[0] for call in a.console.file.write.call_args_list
    )
    # A flavor word from the pool
    assert any(w in written for w in _THINKING_WORDS)
    # Braille wave spinner frame
    assert any(g in written for g in SPINNER_FRAMES)
    # Breathing ellipsis (one of . / .. / …)
    assert any(e in written for e in ELLIPSIS_FRAMES)
    # Cancel hint removed; early frames don't show parens either
    assert "(ctrl+c to cancel)" not in written
    await a._thinking_line.stop()


async def test_stream_delta_stops_thinking(fast_thinking: None) -> None:
    a = _adapter()
    await a.on_llm_call()
    await asyncio.sleep(0.02)
    assert a._thinking_line.visible is True
    await a.on_stream_delta("hi")
    assert a._thinking_line.visible is False
    assert a._thinking_line.task is None
    assert a._phase == "markdown"


async def test_tool_call_stops_thinking(fast_thinking: None) -> None:
    a = _adapter()
    await a.on_llm_call()
    await asyncio.sleep(0.02)
    tc = MagicMock(id="t1")
    tc.name = "read_file"
    await a.on_tool_call(tc)
    assert a._thinking_line.visible is False
    assert a._thinking_line.task is None
    assert a._phase == "tools"


async def test_end_step_stops_thinking(fast_thinking: None) -> None:
    a = _adapter()
    await a.on_llm_call()
    await asyncio.sleep(0.02)
    await a.end_step()
    assert a._thinking_line.visible is False
    assert a._thinking_line.task is None


async def test_pause_for_stdin_stops_thinking(fast_thinking: None) -> None:
    a = _adapter()
    await a.on_llm_call()
    await asyncio.sleep(0.02)
    await a.pause_for_stdin()
    assert a._thinking_line.visible is False
    assert a._thinking_line.task is None


async def test_thinking_detail_shows_effort_past_threshold(
    monkeypatch: pytest.MonkeyPatch, fast_thinking: None,
) -> None:
    # Zero threshold so first tick already reveals the detail section.
    monkeypatch.setattr(status_lines_mod, "_DETAIL_AFTER_S", 0)
    a = CliAdapter(MagicMock(), FLEXOKI_DARK, effort="high")
    a.markdown = MagicMock()
    a.tool_display = MagicMock()
    await a.on_llm_call()
    await asyncio.sleep(0.02)
    written = "".join(c.args[0] for c in a.console.file.write.call_args_list)
    assert "thinking with high effort" in written
    await a._thinking_line.stop()


async def test_thinking_detail_omits_effort_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch, fast_thinking: None,
) -> None:
    monkeypatch.setattr(status_lines_mod, "_DETAIL_AFTER_S", 0)
    a = _adapter()   # default effort=None
    await a.on_llm_call()
    await asyncio.sleep(0.02)
    written = "".join(c.args[0] for c in a.console.file.write.call_args_list)
    assert "thinking with" not in written
    # elapsed still shown in parens
    assert "(" in written and "s)" in written
    await a._thinking_line.stop()


def test_thinking_phrase_and_color_per_tier(fast_thinking: None) -> None:
    a = _adapter()
    tl = a._thinking_line
    assert tl._phrase_and_color(0) is None
    assert tl._phrase_and_color(29) is None
    assert tl._phrase_and_color(30) == ("still working", None)
    assert tl._phrase_and_color(59) == ("still working", None)
    assert tl._phrase_and_color(60) == ("running long", tl._text_ansi)
    assert tl._phrase_and_color(119) == ("running long", tl._text_ansi)
    assert tl._phrase_and_color(120) == ("still pushing", tl._text_ansi)
    assert tl._phrase_and_color(299) == ("still pushing", tl._text_ansi)
    assert tl._phrase_and_color(300) == ("going strong", tl._accent)


async def test_thinking_phrase_renders_in_detail(
    monkeypatch: pytest.MonkeyPatch, fast_thinking: None,
) -> None:
    monkeypatch.setattr(status_lines_mod, "_DETAIL_AFTER_S", 0)
    monkeypatch.setattr(status_lines_mod, "_TIER_30_S", 0)
    a = _adapter()
    await a.on_llm_call()
    await asyncio.sleep(0.02)
    written = "".join(c.args[0] for c in a.console.file.write.call_args_list)
    assert "still working" in written
    await a._thinking_line.stop()


async def test_thinking_phrase_300s_uses_primary_color_segmentation(
    monkeypatch: pytest.MonkeyPatch, fast_thinking: None,
) -> None:
    monkeypatch.setattr(status_lines_mod, "_DETAIL_AFTER_S", 0)
    monkeypatch.setattr(status_lines_mod, "_TIER_300_S", 0)
    a = _adapter()
    await a.on_llm_call()
    await asyncio.sleep(0.02)
    written = "".join(c.args[0] for c in a.console.file.write.call_args_list)
    assert "going strong" in written
    assert a._thinking_line._accent in written
    await a._thinking_line.stop()


async def test_thinking_word_stable_within_run(fast_thinking: None) -> None:
    from agent_cli.render.status_lines import _THINKING_WORDS

    a = _adapter()
    await a.on_llm_call()
    await asyncio.sleep(0.03)     # several ticks
    written = "".join(c.args[0] for c in a.console.file.write.call_args_list)
    used = {w for w in _THINKING_WORDS if w in written}
    assert len(used) == 1, f"word should be fixed per-run, saw: {used}"
    await a._thinking_line.stop()


def _attachment_pair(tool_name: str, **args: object) -> tuple[MagicMock, MagicMock]:
    tc = MagicMock(id="t1", arguments=dict(args))
    tc.name = tool_name
    if tool_name == "read_file":
        path = args.get("file_path", "x")
        tr = MagicMock(
            tool_call_id="t1",
            content=f"[{path}] lines 1-3 of 3\na\nb\nc",
            is_error=False,
        )
    elif tool_name == "list_dir":
        tr = MagicMock(
            tool_call_id="t1", content="src/  (5 entries)\n  a\n  b", is_error=False,
        )
    else:
        tr = MagicMock(tool_call_id="t1", content="ok", is_error=False)
    return tc, tr


def _real_console_adapter() -> tuple[CliAdapter, list[str]]:
    """Adapter with a real Rich console writing to capture buffer."""
    import io as _io
    from rich.console import Console as _Console
    buf = _io.StringIO()
    con = _Console(file=buf, color_system=None, width=200, theme=FLEXOKI_DARK.rich)
    a = CliAdapter(con, FLEXOKI_DARK)
    a.markdown = MagicMock()
    a.tool_display = MagicMock()
    return a, [buf]  # buf in list to keep mutable handle


async def test_render_attachments_empty_skips() -> None:
    a, _ = _real_console_adapter()
    await a.render_attachments([])

    assert a.console.file.getvalue() == ""


async def test_render_attachments_header_and_rows() -> None:
    a, _ = _real_console_adapter()
    pair = _attachment_pair("read_file", file_path="src/foo.py")

    await a.render_attachments([pair])
    out = a.console.file.getvalue()

    assert "Loaded into context" in out
    assert "Read" in out
    assert "src/foo.py" in out
    assert "(3 lines)" in out


async def test_render_attachments_error_styling() -> None:
    a, _ = _real_console_adapter()
    tc = MagicMock(id="t1", arguments={"file_path": "missing.py"})
    tc.name = "read_file"
    tr = MagicMock(tool_call_id="t1", content="Error: not found", is_error=False)

    await a.render_attachments([(tc, tr)])
    out = a.console.file.getvalue()

    assert "Error" in out
    assert "missing.py" in out


async def test_render_attachments_multiple_rows() -> None:
    a, _ = _real_console_adapter()
    p1 = _attachment_pair("read_file", file_path="a.py")
    p2 = _attachment_pair("list_dir", path="src/")

    await a.render_attachments([p1, p2])
    out = a.console.file.getvalue()

    assert "a.py" in out
    assert "src/" in out
    assert out.count("⎿") >= 2


async def test_render_attachments_path_with_brackets_safe() -> None:
    a, _ = _real_console_adapter()
    tc = MagicMock(id="t1", arguments={"file_path": "src/[gen]/x.py"})
    tc.name = "read_file"
    tr = MagicMock(tool_call_id="t1", content="x\ny", is_error=False)

    await a.render_attachments([(tc, tr)])
    out = a.console.file.getvalue()

    assert "[gen]" in out


async def test_render_attachments_multibyte_path_safe() -> None:
    a, _ = _real_console_adapter()
    tc = MagicMock(id="t1", arguments={"file_path": "中文/foo.py"})
    tc.name = "read_file"
    tr = MagicMock(tool_call_id="t1", content="x", is_error=False)

    await a.render_attachments([(tc, tr)])
    out = a.console.file.getvalue()

    assert "中文/foo.py" in out


def _retry_info(
    *,
    attempt: int = 1,
    max_retries: int = 3,
    kind: str = "stream",
    error: Exception | None = None,
) -> LLMRetryInfo:
    return LLMRetryInfo(
        kind=kind,  # type: ignore[arg-type]
        attempt=attempt,
        max_retries=max_retries,
        wait=1.0,
        error=error or LLMConnectionError("transient"),
    )


async def test_on_retry_aborts_markdown_and_prints_separator(
    fast_thinking: None,
) -> None:
    a = _adapter()
    a.print_inline = AsyncMock()
    await a.on_stream_delta("Hello world")
    assert a._phase == "markdown"

    await a.on_retry(_retry_info())

    a.markdown.abort.assert_called_once()
    a.print_inline.assert_awaited_once()
    rendered = a.print_inline.await_args.args[0]
    assert "Retrying LLM (1/3)" in rendered
    assert "LLMConnectionError" in rendered
    assert a._phase == "none"
    assert a._thinking_line.task is None


async def test_on_retry_from_tools_phase_ends_tool_display(
    fast_thinking: None,
) -> None:
    a = _adapter()
    a.print_inline = AsyncMock()
    tc = MagicMock(id="t1")
    tc.name = "read_file"
    await a.on_tool_call(tc)
    assert a._phase == "tools"

    await a.on_retry(_retry_info())

    a.tool_display.end.assert_called_once()
    a.markdown.abort.assert_not_called()
    assert a._phase == "none"


async def test_on_retry_when_no_markdown_buffer_still_prints_and_stops_heartbeat(
    fast_thinking: None,
) -> None:
    a = _adapter()
    a.print_inline = AsyncMock()
    await a.on_llm_call()
    await asyncio.sleep(0.02)
    assert a._thinking_line.task is not None

    await a.on_retry(_retry_info(kind="generate", error=ConnectionError("dns")))

    a.markdown.abort.assert_not_called()
    a.print_inline.assert_awaited_once()
    assert a._thinking_line.task is None


async def test_on_retry_escapes_custom_exception_class_name(
    fast_thinking: None,
) -> None:
    a = _adapter()
    a.print_inline = AsyncMock()

    class WeirdName(Exception):
        ...

    # Pick a name that rich would interpret as markup (lowercase tag-like).
    WeirdName.__name__ = "Weird[bold red]Error"

    await a.on_retry(_retry_info(error=WeirdName()))

    rendered = a.print_inline.await_args.args[0]
    # rich.markup.escape prefixes the bracket with a backslash so the tag
    # is not parsed as active markup.
    assert r"Weird\[bold red]Error" in rendered
