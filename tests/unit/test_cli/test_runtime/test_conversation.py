"""Tests for runtime/conversation.py — append_tool_turn + refresh_system_prompt."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_cli.runtime.conversation import (
    _write,
    append_shell_run,
    append_tool_turn,
    drain_pending,
    refresh_system_prompt,
)
from agent_harness.core.message import Message, Role, ToolCall, ToolResult


def _agent_with_memory() -> tuple[Any, list[Message]]:
    captured: list[Message] = []
    memory = MagicMock()

    async def add_message(msg: Message) -> None:
        captured.append(msg)

    memory.add_message = add_message
    agent = MagicMock()
    agent.context.short_term_memory = memory
    # Must be a real dict (not MagicMock) — pydantic validates provider_metadata
    agent.llm.synthetic_turn_sidecar = MagicMock(return_value={})
    return agent, captured


def _pair(idx: int, *, is_error: bool = False) -> tuple[ToolCall, ToolResult]:
    tc = ToolCall(id=f"id_{idx}", name="read_file", arguments={"file_path": f"f{idx}"})
    tr = ToolResult(tool_call_id=tc.id, content=f"content_{idx}", is_error=is_error)
    return tc, tr


@pytest.mark.asyncio
async def test_empty_pairs_writes_nothing() -> None:
    agent, captured = _agent_with_memory()
    render = AsyncMock()

    await append_tool_turn(agent, [], render=render)

    assert captured == []
    render.assert_not_called()


@pytest.mark.asyncio
async def test_single_pair_no_render_writes_assistant_then_tool() -> None:
    agent, captured = _agent_with_memory()
    pair = _pair(1)

    await append_tool_turn(agent, [pair])

    assert len(captured) == 2
    assert captured[0].role == Role.ASSISTANT
    assert captured[0].tool_calls == [pair[0]]
    assert captured[1].role == Role.TOOL
    assert captured[1].tool_result is not None
    assert captured[1].tool_result.tool_call_id == "id_1"


@pytest.mark.asyncio
async def test_render_runs_before_memory_write() -> None:
    agent, captured = _agent_with_memory()
    pair = _pair(1)
    timeline: list[str] = []

    async def render(items: list[tuple[ToolCall, ToolResult]]) -> None:
        timeline.append("render")

    orig_add = agent.context.short_term_memory.add_message

    async def add_message(msg: Message) -> None:
        timeline.append("write")
        await orig_add(msg)

    agent.context.short_term_memory.add_message = add_message

    await append_tool_turn(agent, [pair], render=render)

    assert timeline == ["render", "write", "write"]


@pytest.mark.asyncio
async def test_render_exception_skips_memory_write() -> None:
    agent, captured = _agent_with_memory()
    pair = _pair(1)

    async def render(items: list[tuple[ToolCall, ToolResult]]) -> None:
        raise RuntimeError("render fail")

    with pytest.raises(RuntimeError, match="render fail"):
        await append_tool_turn(agent, [pair], render=render)

    assert captured == []


@pytest.mark.asyncio
async def test_multi_pair_assistant_carries_all_tcs() -> None:
    agent, captured = _agent_with_memory()
    pairs = [_pair(1), _pair(2), _pair(3)]

    await append_tool_turn(agent, pairs)

    assert captured[0].role == Role.ASSISTANT
    assert captured[0].tool_calls is not None
    assert [tc.id for tc in captured[0].tool_calls] == ["id_1", "id_2", "id_3"]
    assert [m.tool_result.tool_call_id for m in captured[1:]] == [
        "id_1",
        "id_2",
        "id_3",
    ]


@pytest.mark.asyncio
async def test_synthesized_assistant_carries_provider_sidecar() -> None:
    """The fake assistant turn must inherit whatever the LLM declares as its
    synthetic_turn_sidecar — without it, reasoning-capable backends 400 on
    the next request with "reasoning_content must be passed back"."""
    agent, captured = _agent_with_memory()
    stamp = {"openai_chat": {"reasoning_content": "", "reasoning_details": []}}
    agent.llm.synthetic_turn_sidecar = MagicMock(return_value=stamp)
    pair = _pair(1)

    await append_tool_turn(agent, [pair])

    agent.llm.synthetic_turn_sidecar.assert_called_once()
    assert captured[0].role == Role.ASSISTANT
    assert captured[0].provider_metadata == stamp
    # tool result message does not get the stamp — only assistant does
    assert captured[1].role == Role.TOOL
    assert captured[1].provider_metadata == {}


@pytest.mark.asyncio
async def test_is_error_pass_through() -> None:
    agent, captured = _agent_with_memory()
    pair = _pair(1, is_error=True)

    await append_tool_turn(agent, [pair])

    assert captured[1].tool_result is not None
    assert captured[1].tool_result.is_error is True


@pytest.mark.asyncio
async def test_shield_completes_writes_when_outer_cancelled() -> None:
    agent, captured = _agent_with_memory()
    pairs = [_pair(1), _pair(2)]

    async def runner() -> None:
        await append_tool_turn(agent, pairs)

    task = asyncio.create_task(runner())
    await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert len(captured) == 3, (
        f"shield should let memory writes finish; got {len(captured)} entries"
    )


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


# ── Pending-write tracker (Phase 3) ──


from agent_cli.runtime.conversation import (
    reset_pending_tracker,
    use_pending_tracker,
)


@pytest.mark.asyncio
async def test_tracker_unbound_is_noop() -> None:
    agent, captured = _agent_with_memory()
    pair = _pair(1)

    await append_tool_turn(agent, [pair])

    assert len(captured) == 2


@pytest.mark.asyncio
async def test_tracker_bound_records_inner_write_task() -> None:
    agent, _ = _agent_with_memory()
    tracker: list[asyncio.Future[Any]] = []
    token = use_pending_tracker(tracker)
    try:
        await append_tool_turn(agent, [_pair(1)])
    finally:
        reset_pending_tracker(token)

    assert len(tracker) == 1
    assert tracker[0].done()


@pytest.mark.asyncio
async def test_tracker_drain_after_outer_cancel_completes_write() -> None:
    agent, captured = _agent_with_memory()
    tracker: list[asyncio.Future[Any]] = []

    async def runner() -> None:
        token = use_pending_tracker(tracker)
        try:
            await append_tool_turn(agent, [_pair(1), _pair(2)])
        finally:
            reset_pending_tracker(token)

    task = asyncio.create_task(runner())
    await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    pending = list(tracker)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    assert len(captured) == 3, (
        f"shield + tracker drain should let writes finish; got {len(captured)}"
    )


# ── _write — unified writer (tool_pairs / user_content / save) ──


@pytest.mark.asyncio
async def test_write_tool_pairs_no_save() -> None:
    agent, captured = _agent_with_memory()
    save = AsyncMock()

    await _write(agent, tool_pairs=[_pair(1), _pair(2)])

    assert [m.role for m in captured] == [Role.ASSISTANT, Role.TOOL, Role.TOOL]
    save.assert_not_called()


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


@pytest.mark.asyncio
async def test_write_tool_pairs_with_save_orders_all_memory_before_save() -> None:
    timeline: list[str] = []

    memory = MagicMock()

    async def add_message(msg: Message) -> None:
        timeline.append(f"add:{msg.role.value}")

    memory.add_message = add_message
    agent = MagicMock()
    agent.context.short_term_memory = memory
    agent.llm.synthetic_turn_sidecar = MagicMock(return_value={})

    async def save() -> None:
        timeline.append("save")

    await _write(agent, tool_pairs=[_pair(1), _pair(2)], save=save)

    assert timeline == ["add:assistant", "add:tool", "add:tool", "save"]


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
    assert "tool_pairs" not in captured_kwargs
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


@pytest.mark.asyncio
async def test_append_shell_run_registers_with_pending_tracker() -> None:
    agent, captured = _agent_with_memory()
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
        agent,
        command="cd nope",
        exit_code=0,
        output="",
        save=save,
        post_notices=[notice],
    )

    body = captured["user_content"]
    assert f"[Accident] {notice}" in body


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


@pytest.mark.asyncio
async def test_append_tool_turn_routes_pairs_to_write_without_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = MagicMock()
    captured: dict[str, Any] = {}

    async def fake_write(_agent: Any, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("agent_cli.runtime.conversation._write", fake_write)

    pairs = [_pair(1), _pair(2)]
    await append_tool_turn(agent, pairs)

    assert captured.get("tool_pairs") == pairs
    assert "user_content" not in captured
    assert captured.get("save") is None or "save" not in captured
