import asyncio
import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.console import Console

from agent_cli.commands.builtin.compact import CMD
from agent_cli.theme import DEFAULT_THEME

from ..conftest import render_output


def _stub_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.console = Console(file=io.StringIO(), force_terminal=False, color_system=None, width=80)
    adapter.lock = MagicMock(return_value=asyncio.Lock())
    adapter.theme = DEFAULT_THEME
    return adapter


def _home_archive_path(name: str = "compression_round_1.md") -> str:
    """A path under $HOME so _display_archive renders it as `~/...`."""
    return str(
        Path.home() / ".agent-harness" / "sessions" / "test-session" / "compact" / name
    )


async def test_compact_passes_extra_instructions_and_saves() -> None:
    archive = _home_archive_path()
    compressor = MagicMock()
    compressor.compress = AsyncMock(return_value=["msg1"])
    compressor.take_last_result = MagicMock(return_value=MagicMock(
        original_count=10, compressed_count=3,
        archive_path=archive,
        llm_usage=MagicMock(total_tokens=0),
    ))
    agent = MagicMock()
    agent.context.short_term_memory.compressor = compressor
    agent.context.short_term_memory._messages = ["m1", "m2"]
    save = AsyncMock()
    ctx = MagicMock(agent=agent, save_session=save, adapter=_stub_adapter())

    result = await CMD.handler(ctx, "focus on auth")
    compressor.compress.assert_awaited_once()
    assert compressor.compress.await_args.kwargs["extra_instructions"] == "focus on auth"
    save.assert_awaited_once()
    out = render_output(result.output)
    assert "Compacted: 10 → 3 msgs" in out
    # archive detail: count + home-relative path so user can cat the file
    assert (
        "7 archived to ~/.agent-harness/sessions/test-session/compact/compression_round_1.md"
        in out
    )


async def test_compact_without_compressor_returns_message() -> None:
    agent = MagicMock()
    agent.context.short_term_memory.compressor = None
    ctx = MagicMock(agent=agent)
    result = await CMD.handler(ctx, "")
    assert "not enabled" in render_output(result.output).lower()


async def test_compact_save_cancel_propagates_without_repl_concerns() -> None:
    """save_session cancel inside /compact bubbles as CancelledError;
    outer _handle_line (tested separately) is what keeps REPL alive."""
    compressor = MagicMock()
    compressor.compress = AsyncMock(return_value=["compressed"])
    compressor.take_last_result = MagicMock(return_value=MagicMock(
        original_count=10, compressed_count=3,
        archive_path=None,
        llm_usage=MagicMock(total_tokens=0),
    ))
    agent = MagicMock()
    agent.context.short_term_memory.compressor = compressor
    agent.context.short_term_memory._messages = ["m1", "m2"]
    save = AsyncMock(side_effect=asyncio.CancelledError())
    ctx = MagicMock(agent=agent, save_session=save, adapter=_stub_adapter())

    with pytest.raises(asyncio.CancelledError):
        await CMD.handler(ctx, "")
    compressor.compress.assert_awaited_once()


async def test_compact_no_op_preserves_last_call_snapshot() -> None:
    """When compressor returns the unchanged list (`take_last_result() is None`),
    /compact must skip `replace_messages`/`save_session` — calling them would
    wipe `last_call`, which the status bar reads for the displayed token count
    (regression: bottom bar shows `—/Xm` until the next LLM turn).
    """
    compressor = MagicMock()
    original = ["m1", "m2"]
    compressor.compress = AsyncMock(return_value=original)
    compressor.take_last_result = MagicMock(return_value=None)
    agent = MagicMock()
    stm = agent.context.short_term_memory
    stm.compressor = compressor
    stm._messages = original
    save = AsyncMock()
    ctx = MagicMock(agent=agent, save_session=save, adapter=_stub_adapter())

    result = await CMD.handler(ctx, "")
    out = render_output(result.output)
    assert "Nothing to compact" in out
    stm.replace_messages.assert_not_called()
    save.assert_not_awaited()


async def test_compact_output_omits_archive_when_no_path() -> None:
    """Defensive: if archive_path is missing (e.g. session_id unset), the
    archive detail is suppressed rather than rendering a broken filename."""
    compressor = MagicMock()
    compressor.compress = AsyncMock(return_value=["msg1"])
    compressor.take_last_result = MagicMock(return_value=MagicMock(
        original_count=10, compressed_count=3,
        archive_path=None,
        llm_usage=MagicMock(total_tokens=0),
    ))
    agent = MagicMock()
    agent.context.short_term_memory.compressor = compressor
    agent.context.short_term_memory._messages = ["m1", "m2"]
    save = AsyncMock()
    ctx = MagicMock(agent=agent, save_session=save, adapter=_stub_adapter())

    result = await CMD.handler(ctx, "")
    out = render_output(result.output)
    assert "Compacted: 10 → 3 msgs" in out
    assert "archived" not in out


async def test_compact_archive_outside_home_falls_back_to_absolute() -> None:
    """`_display_archive` should keep an absolute path when archive_path
    isn't under $HOME (e.g. a config override pointing to /tmp/…)."""
    archive = "/tmp/agent-harness-archives/compression_round_1.md"
    compressor = MagicMock()
    compressor.compress = AsyncMock(return_value=["msg"])
    compressor.take_last_result = MagicMock(return_value=MagicMock(
        original_count=5, compressed_count=2,
        archive_path=archive,
        llm_usage=MagicMock(total_tokens=0),
    ))
    agent = MagicMock()
    agent.context.short_term_memory.compressor = compressor
    agent.context.short_term_memory._messages = ["m"]
    save = AsyncMock()
    ctx = MagicMock(agent=agent, save_session=save, adapter=_stub_adapter())

    result = await CMD.handler(ctx, "")
    out = render_output(result.output)
    assert f"archived to {archive}" in out
