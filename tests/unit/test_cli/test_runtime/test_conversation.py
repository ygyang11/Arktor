"""Tests for runtime/conversation.py — _write / append_shell_run / tracker."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_cli.runtime.conversation import (
    _write,
    append_shell_run,
    drain_pending,
    refresh_system_prompt,
    reset_pending_tracker,
    use_pending_tracker,
)
from agent_harness.core.message import Message, Role


def _agent_with_memory() -> tuple[Any, list[Message]]:
    captured: list[Message] = []
    memory = MagicMock()

    async def add_message(msg: Message) -> None:
        captured.append(msg)

    memory.add_message = add_message
    agent = MagicMock()
    agent.context.short_term_memory = memory
    return agent, captured


class TestRefreshSystemPrompt:
    def test_rebuilds_and_assigns(self) -> None:
        agent = MagicMock()
        builder = MagicMock()
        builder.build = MagicMock(return_value="NEW_PROMPT")
        agent._prompt_builder = builder
        agent._make_builder_context = MagicMock(return_value={"key": "ctx"})
        agent.system_prompt = "OLD_PROMPT"

        refresh_system_prompt(agent)

        builder.build.assert_called_once_with({"key": "ctx"})
        assert agent.system_prompt == "NEW_PROMPT"

    def test_does_not_touch_memory_directly(self) -> None:
        agent = MagicMock()
        agent._prompt_builder.build = MagicMock(return_value="X")
        agent._make_builder_context = MagicMock(return_value={})
        msgs: list[Message] = [Message.system("OLD")]
        agent.context.short_term_memory._messages = msgs

        refresh_system_prompt(agent)

        assert msgs[0].content == "OLD"


# ── _write — user_content writer ──


@pytest.mark.asyncio
async def test_write_user_content_no_save() -> None:
    agent, captured = _agent_with_memory()

    await _write(agent, user_content="hello")

    assert len(captured) == 1
    assert captured[0].role == Role.USER
    assert captured[0].content == "hello"


@pytest.mark.asyncio
async def test_write_user_content_with_save_orders_memory_before_save() -> None:
    timeline: list[str] = []
    memory = MagicMock()

    async def add_message(msg: Message) -> None:
        timeline.append(f"add:{msg.role.value}")

    memory.add_message = add_message
    agent = MagicMock()
    agent.context.short_term_memory = memory

    async def save() -> None:
        timeline.append("save")

    await _write(agent, user_content="hi", save=save)

    assert timeline == ["add:user", "save"]


# ── append_shell_run — `!` lane injector ──


@pytest.mark.asyncio
async def test_append_shell_run_routes_format_and_save_to_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = MagicMock()
    captured_kwargs: dict[str, Any] = {}

    async def fake_write(_agent: Any, **kwargs: Any) -> None:
        captured_kwargs.update(kwargs)

    monkeypatch.setattr("agent_cli.runtime.conversation._write", fake_write)

    save = AsyncMock()
    await append_shell_run(
        agent, command="echo hi", exit_code=0, output="hi", save=save,
    )

    assert "user_content" in captured_kwargs
    assert captured_kwargs["save"] is save
    body = captured_kwargs["user_content"]
    assert isinstance(body, str)
    assert body.startswith("<user-shell-run>\n```sh\necho hi\n```\n")
    assert body.endswith("\n</user-shell-run>")
    assert "hi" in body


@pytest.mark.asyncio
async def test_append_shell_run_failure_includes_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = MagicMock()
    captured: dict[str, Any] = {}

    async def fake_write(_agent: Any, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("agent_cli.runtime.conversation._write", fake_write)

    save = AsyncMock()
    await append_shell_run(
        agent, command="false", exit_code=1, output="", save=save,
    )

    body = captured["user_content"]
    assert "[exit code 1]" in body
    assert "(Completed with no output)" in body


@pytest.mark.asyncio
async def test_append_shell_run_forwards_post_notices_to_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = MagicMock()
    captured: dict[str, Any] = {}

    async def fake_write(_agent: Any, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("agent_cli.runtime.conversation._write", fake_write)

    save = AsyncMock()
    notice = "cwd reverted"
    await append_shell_run(
        agent, command="cd nope", exit_code=0, output="", save=save,
        post_notices=[notice],
    )

    assert f"[Accident] {notice}" in captured["user_content"]


@pytest.mark.asyncio
async def test_append_shell_run_registers_with_pending_tracker() -> None:
    agent, _ = _agent_with_memory()
    save = AsyncMock()
    tracker: list[asyncio.Future[Any]] = []
    token = use_pending_tracker(tracker)
    try:
        await append_shell_run(
            agent, command="ls", exit_code=0, output="x", save=save,
        )
    finally:
        reset_pending_tracker(token)

    assert len(tracker) == 1
    assert tracker[0].done()


@pytest.mark.asyncio
async def test_append_shell_run_no_tracker_bound_is_noop() -> None:
    agent, captured = _agent_with_memory()
    save = AsyncMock()
    await append_shell_run(
        agent, command="ls", exit_code=0, output="x", save=save,
    )
    assert len(captured) == 1
    save.assert_awaited_once()


@pytest.mark.asyncio
async def test_append_shell_run_shield_completes_when_outer_cancelled() -> None:
    captured: list[Message] = []
    memory = MagicMock()

    async def add_message(msg: Message) -> None:
        await asyncio.sleep(0)
        captured.append(msg)

    memory.add_message = add_message
    agent = MagicMock()
    agent.context.short_term_memory = memory

    save_called = asyncio.Event()

    async def save() -> None:
        await asyncio.sleep(0)
        save_called.set()

    async def runner() -> None:
        await append_shell_run(
            agent, command="ls", exit_code=0, output="x", save=save,
        )

    task = asyncio.create_task(runner())
    await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert len(captured) == 1
    assert captured[0].role == Role.USER
    assert save_called.is_set()


# ── drain_pending ──


@pytest.mark.asyncio
async def test_drain_pending_empty_is_noop() -> None:
    pending: list[asyncio.Future[Any]] = []
    await drain_pending(pending)
    assert pending == []


@pytest.mark.asyncio
async def test_drain_pending_awaits_and_clears() -> None:
    completed: list[int] = []

    async def work(idx: int) -> None:
        await asyncio.sleep(0)
        completed.append(idx)

    pending: list[asyncio.Future[Any]] = [
        asyncio.ensure_future(work(0)),
        asyncio.ensure_future(work(1)),
    ]
    await drain_pending(pending)
    assert sorted(completed) == [0, 1]
    assert pending == []


@pytest.mark.asyncio
async def test_drain_pending_swallows_exceptions() -> None:
    async def explode() -> None:
        raise RuntimeError("boom")

    pending: list[asyncio.Future[Any]] = [asyncio.ensure_future(explode())]
    await drain_pending(pending)
    assert pending == []
