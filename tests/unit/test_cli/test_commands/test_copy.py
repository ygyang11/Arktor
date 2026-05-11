from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_cli.commands.builtin import copy as copy_mod
from agent_cli.commands.builtin.copy import CMD
from agent_harness.core.message import Message

from .conftest import render_output


def _ctx_with_messages(*assistant_texts: str) -> MagicMock:
    msgs = [Message.assistant(t) for t in assistant_texts]
    agent = MagicMock()
    agent.context.short_term_memory.get_context_messages = AsyncMock(return_value=msgs)
    return MagicMock(agent=agent)


@pytest.fixture(autouse=True)
def _stub_clipboard(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    captured: dict[str, str] = {}

    def fake_native(text: str) -> bool:
        captured["text"] = text
        return True

    monkeypatch.setattr(copy_mod, "_native_copy", fake_native)
    monkeypatch.setattr(copy_mod, "_osc52_copy", lambda text: True)
    return captured


async def test_copy_default_copies_last_assistant_reply(_stub_clipboard: dict[str, str]) -> None:
    ctx = _ctx_with_messages("first", "second", "third")
    result = await CMD.handler(ctx, "")
    assert "Copied last message" in render_output(result.output)
    assert _stub_clipboard["text"] == "third"


async def test_copy_with_index_picks_nth_from_last(_stub_clipboard: dict[str, str]) -> None:
    ctx = _ctx_with_messages("first", "second", "third")
    result = await CMD.handler(ctx, "2")
    assert "#2" in render_output(result.output)
    assert _stub_clipboard["text"] == "second"


async def test_copy_skips_user_messages_in_count(_stub_clipboard: dict[str, str]) -> None:
    msgs = [
        Message.assistant("a1"),
        Message.user("u1"),
        Message.assistant("a2"),
    ]
    agent = MagicMock()
    agent.context.short_term_memory.get_context_messages = AsyncMock(return_value=msgs)
    result = await CMD.handler(MagicMock(agent=agent), "")
    assert _stub_clipboard["text"] == "a2"


async def test_copy_index_out_of_range_returns_error() -> None:
    ctx = _ctx_with_messages("only one")
    result = await CMD.handler(ctx, "5")
    assert "Message #5 not found" in render_output(result.output)


async def test_copy_no_assistant_replies_returns_error() -> None:
    msgs = [Message.user("only user")]
    agent = MagicMock()
    agent.context.short_term_memory.get_context_messages = AsyncMock(return_value=msgs)
    result = await CMD.handler(MagicMock(agent=agent), "")
    assert "not found" in render_output(result.output)


async def test_copy_both_native_and_osc52_fail_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(copy_mod, "_native_copy", lambda text: False)
    monkeypatch.setattr(copy_mod, "_osc52_copy", lambda text: False)
    ctx = _ctx_with_messages("hi")
    result = await CMD.handler(ctx, "")
    out = render_output(result.output)
    assert "Could not access clipboard" in out


async def test_copy_falls_back_to_osc52_when_native_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(copy_mod, "_native_copy", lambda text: False)
    seen: list[str] = []

    def fake_osc52(text: str) -> bool:
        seen.append(text)
        return True

    monkeypatch.setattr(copy_mod, "_osc52_copy", fake_osc52)
    ctx = _ctx_with_messages("only")
    result = await CMD.handler(ctx, "")
    assert seen == ["only"]
    assert "Copied last message" in render_output(result.output)


async def test_copy_install_hint_is_platform_aware() -> None:
    import sys

    hint = copy_mod._install_hint()
    if sys.platform == "darwin":
        assert "pbcopy" in hint
    elif sys.platform == "linux":
        assert "wl-copy" in hint or "xclip" in hint
    elif sys.platform == "win32":
        assert "clip" in hint
