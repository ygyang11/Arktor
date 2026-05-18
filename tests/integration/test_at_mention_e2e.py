"""End-to-end @-mention injection: real BaseAgent + mocked LLM verifies
the attachment is embedded as <system-reminder> blocks on the single user
message (no synthetic assistant/tool turn) before the first LLM call."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_app.tools.filesystem.list_dir import list_dir
from agent_app.tools.filesystem.read_file import read_file
from agent_cli.repl.mentions import expand_mentions
from agent_harness.agent.react import ReActAgent
from agent_harness.context.context import AgentContext
from agent_harness.core.message import Role


def _adapter() -> Any:
    a = MagicMock()
    a.render_attachments = AsyncMock()
    return a


def _cb(adapter: Any) -> Any:
    async def cb(a: Any, msg: Any, t: str) -> None:
        if msg.role != Role.USER:
            return
        await expand_mentions(a, adapter, t)
    return cb


@pytest.mark.asyncio
async def test_user_at_file_embeds_single_user_message(
    mock_llm, config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "foo.py").write_text("# hello\nprint('hi')\n")
    mock_llm.add_text_response("I read foo.py and it greets.")

    ctx = AgentContext.create(config)
    agent = ReActAgent(
        name="cli_agent", llm=mock_llm, tools=[read_file, list_dir], context=ctx,
    )
    adapter = _adapter()

    result = await agent.run("explain @foo.py", after_input_appended=_cb(adapter))

    assert result.output == "I read foo.py and it greets."

    first_call = mock_llm.call_history[0]
    roles = [m.role for m in first_call]
    assert Role.ASSISTANT not in roles
    assert Role.TOOL not in roles

    user_msg = next(m for m in first_call if m.role == Role.USER)
    assert "Called the read_file tool" in (user_msg.content or "")
    assert "hello" in (user_msg.content or "")
    assert (user_msg.content or "").endswith("explain @foo.py")


@pytest.mark.asyncio
async def test_multiple_mentions_embed_two_blocks_no_synthetic_turn(
    mock_llm, config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("a")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "b.py").write_text("b")
    mock_llm.add_text_response("done")

    ctx = AgentContext.create(config)
    agent = ReActAgent(
        name="cli_agent", llm=mock_llm, tools=[read_file, list_dir], context=ctx,
    )
    adapter = _adapter()

    await agent.run("look at @a.py and @src", after_input_appended=_cb(adapter))

    first_call = mock_llm.call_history[0]
    assert [m for m in first_call if m.role == Role.ASSISTANT] == []
    assert [m for m in first_call if m.role == Role.TOOL] == []

    user_msg = next(m for m in first_call if m.role == Role.USER)
    content = user_msg.content or ""
    assert "Called the read_file tool" in content
    assert "Called the list_dir tool" in content
    assert content.index("read_file") < content.index("list_dir")


@pytest.mark.asyncio
async def test_no_mentions_no_embed(
    mock_llm, config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    mock_llm.add_text_response("hi")

    ctx = AgentContext.create(config)
    agent = ReActAgent(
        name="cli_agent", llm=mock_llm, tools=[read_file, list_dir], context=ctx,
    )
    adapter = _adapter()

    await agent.run("just text", after_input_appended=_cb(adapter))

    first_call = mock_llm.call_history[0]
    user_msg = next(m for m in first_call if m.role == Role.USER)
    assert user_msg.content == "just text"
    assert "<system-reminder>" not in (user_msg.content or "")
    adapter.render_attachments.assert_not_called()


@pytest.mark.asyncio
async def test_nonexistent_at_silently_skipped(
    mock_llm, config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    mock_llm.add_text_response("ok")

    ctx = AgentContext.create(config)
    agent = ReActAgent(
        name="cli_agent", llm=mock_llm, tools=[read_file, list_dir], context=ctx,
    )
    adapter = _adapter()

    await agent.run("@nonexistent.py please", after_input_appended=_cb(adapter))

    first_call = mock_llm.call_history[0]
    assert [m for m in first_call if m.role == Role.ASSISTANT] == []
    user_msg = next(m for m in first_call if m.role == Role.USER)
    assert user_msg.content == "@nonexistent.py please"
