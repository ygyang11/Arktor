"""Tests for runtime/session.py.

Covers two concern groups against the same module:
- turn lifecycle: snapshot / drain / rollback / decision behaviour
- session persistence: make_save_session / switch_session plan_mode handling
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_cli.runtime import plan_mode
from agent_cli.runtime.session import (
    _TurnContext,
    make_save_session,
    rollback,
    should_rollback,
    switch_session,
    take_snapshot,
    transcript_changed,
)
from agent_harness.core.message import Message, Role
from agent_harness.llm.types import Usage


SYS_PROMPT = "You are a careful assistant."


def _agent(messages: list[Message], compressor: Any | None = None) -> Any:
    from agent_harness.memory.short_term import ShortTermMemory

    agent = MagicMock()
    agent.system_prompt = SYS_PROMPT
    agent._total_usage = Usage()
    stm = ShortTermMemory()
    stm._messages = messages
    stm.compressor = compressor
    agent.context.short_term_memory = stm
    return agent


def _compressor(count: int = 2, archives: list[str] | None = None) -> Any:
    c = MagicMock()
    c._compression_count = count
    c._archive_paths = list(archives or [])
    c._last_result = MagicMock()
    return c


# ── take_snapshot ──


def test_snapshot_messages_are_deep_copies() -> None:
    msgs = [Message.system(SYS_PROMPT), Message.user("hello")]
    agent = _agent(msgs)

    ctx = take_snapshot(agent)

    assert len(ctx.snapshot_messages) == 2
    msgs[1].content = "MUTATED"
    assert ctx.snapshot_messages[1].content == "hello"


def test_snapshot_ids_are_original_object_ids_not_copies() -> None:
    msgs = [Message.system(SYS_PROMPT), Message.user("hello")]
    agent = _agent(msgs)

    ctx = take_snapshot(agent)

    assert ctx.snapshot_ids == frozenset(id(m) for m in msgs)
    for copy in ctx.snapshot_messages:
        assert id(copy) not in ctx.snapshot_ids


def test_snapshot_main_system_id_locks_first_match() -> None:
    main = Message.system(SYS_PROMPT)
    other_sys = Message.system("a non-identity system message")
    msgs = [main, other_sys, Message.user("hi")]
    agent = _agent(msgs)

    ctx = take_snapshot(agent)

    assert ctx.main_system_id == id(main)


def test_snapshot_main_system_id_none_when_no_match() -> None:
    msgs = [Message.system("DIFFERENT"), Message.user("hi")]
    agent = _agent(msgs)

    ctx = take_snapshot(agent)

    assert ctx.main_system_id is None


def test_snapshot_main_system_id_none_when_system_prompt_empty() -> None:
    msgs = [Message.system("anything"), Message.user("hi")]
    agent = _agent(msgs)
    agent.system_prompt = ""

    ctx = take_snapshot(agent)

    assert ctx.main_system_id is None


def test_snapshot_compressor_state_none_when_absent() -> None:
    agent = _agent([Message.user("hi")], compressor=None)

    ctx = take_snapshot(agent)

    assert ctx.snapshot_compressor_state is None


def test_snapshot_compressor_state_captured_when_present() -> None:
    comp = _compressor(count=3, archives=["a.md", "b.md"])
    agent = _agent([Message.user("hi")], compressor=comp)

    ctx = take_snapshot(agent)

    assert ctx.snapshot_compressor_state == (3, ["a.md", "b.md"])


def test_snapshot_compressor_archives_are_copied() -> None:
    archives = ["a.md"]
    comp = _compressor(count=1, archives=archives)
    agent = _agent([Message.user("hi")], compressor=comp)

    ctx = take_snapshot(agent)
    archives.append("b.md")

    assert ctx.snapshot_compressor_state == (1, ["a.md"])


def test_snapshot_does_not_carry_total_usage_field() -> None:
    agent = _agent([Message.user("hi")])

    ctx = take_snapshot(agent)

    assert not hasattr(ctx, "snapshot_total_usage")


# ── transcript_changed ──


def _ctx_for(snapshot: list[Message], main_idx: int | None = 0) -> _TurnContext:
    return _TurnContext(
        snapshot_messages=[m.model_copy(deep=True) for m in snapshot],
        snapshot_compressor_state=None,
        snapshot_ids=frozenset(id(m) for m in snapshot),
        main_system_id=id(snapshot[main_idx]) if main_idx is not None and snapshot else None,
        fs_state={},
    )


def test_transcript_unchanged_when_only_appended() -> None:
    a = Message.system(SYS_PROMPT)
    b = Message.user("u1")
    ctx = _ctx_for([a, b], main_idx=0)
    new = Message.assistant("reply")

    assert transcript_changed(ctx, [a, b, new]) is False


def test_transcript_unchanged_when_main_system_replaced_via_model_copy() -> None:
    a = Message.system(SYS_PROMPT)
    b = Message.user("u1")
    ctx = _ctx_for([a, b], main_idx=0)
    new_main = a.model_copy(update={"content": SYS_PROMPT})

    assert transcript_changed(ctx, [new_main, b]) is False


def test_transcript_changed_when_non_system_message_removed() -> None:
    a = Message.system(SYS_PROMPT)
    b = Message.user("u1")
    c = Message.assistant("a1")
    ctx = _ctx_for([a, b, c], main_idx=0)

    assert transcript_changed(ctx, [a, c]) is True


def test_transcript_changed_when_compression_summary_removed() -> None:
    a = Message.system(SYS_PROMPT)
    summary = Message.system("## summary", metadata={"is_compression_summary": True})
    c = Message.user("u")
    ctx = _ctx_for([a, summary, c], main_idx=0)

    assert transcript_changed(ctx, [a, c]) is True


def test_transcript_changed_when_bg_result_removed() -> None:
    a = Message.system(SYS_PROMPT)
    bg = Message.system("[bg done]", metadata={"is_background_result": True})
    c = Message.user("u")
    ctx = _ctx_for([a, bg, c], main_idx=0)

    assert transcript_changed(ctx, [a, c]) is True


def test_transcript_changed_when_no_main_id_and_identity_msg_replaced() -> None:
    a = Message.system("DIFFERENT")
    b = Message.user("u1")
    ctx = _ctx_for([a, b], main_idx=None)
    new_a = a.model_copy(update={"content": "DIFFERENT"})

    assert transcript_changed(ctx, [new_a, b]) is True


# ── should_rollback ──


def test_should_rollback_false_when_committed() -> None:
    a = Message.system(SYS_PROMPT)
    ctx = _ctx_for([a], main_idx=0)
    ctx.committed = True

    assert should_rollback(ctx, [a]) is False


def test_should_rollback_true_when_unchanged_uncommitted() -> None:
    a = Message.system(SYS_PROMPT)
    b = Message.user("u")
    ctx = _ctx_for([a, b], main_idx=0)

    assert should_rollback(ctx, [a, b, Message.assistant("r")]) is True


def test_should_rollback_false_when_changed_uncommitted() -> None:
    a = Message.system(SYS_PROMPT)
    b = Message.user("u")
    ctx = _ctx_for([a, b], main_idx=0)

    assert should_rollback(ctx, [a]) is False


# ── rollback ──


@pytest.mark.asyncio
async def test_rollback_restores_messages_and_drops_new_non_bg() -> None:
    a = Message.system(SYS_PROMPT)
    b = Message.user("u1")
    snapshot = [a, b]
    new_user = Message.user("new")
    new_assistant = Message.assistant("reply")
    current = list(snapshot) + [new_user, new_assistant]
    agent = _agent(current)
    ctx = _ctx_for(snapshot, main_idx=0)
    save = AsyncMock()

    await rollback(agent, ctx, save)

    msgs = agent.context.short_term_memory._messages
    assert len(msgs) == 2
    assert [m.content for m in msgs] == [SYS_PROMPT, "u1"]
    save.assert_awaited_once()


@pytest.mark.asyncio
async def test_rollback_preserves_bg_results_added_during_turn() -> None:
    a = Message.system(SYS_PROMPT)
    b = Message.user("u1")
    snapshot = [a, b]
    new_assistant = Message.assistant("reply")
    bg = Message.system("[bg done]", metadata={"is_background_result": True})
    current = list(snapshot) + [new_assistant, bg]
    agent = _agent(current)
    ctx = _ctx_for(snapshot, main_idx=0)
    save = AsyncMock()

    await rollback(agent, ctx, save)

    msgs = agent.context.short_term_memory._messages
    assert len(msgs) == 3
    assert msgs[-1].metadata.get("is_background_result") is True
    assert msgs[-1].content == "[bg done]"


@pytest.mark.asyncio
async def test_rollback_restores_file_freshness(tmp_path: Any) -> None:
    from agent_app.observability import file_freshness as ff
    from agent_harness.context.variables import ContextVariables

    agent = _agent([Message.system(SYS_PROMPT), Message.user("u1")])
    agent.context.variables = ContextVariables()
    f = tmp_path / "x.txt"
    f.write_text("hello")
    ff.mark_read(agent, f)
    ctx = take_snapshot(agent)

    f.write_text("changed")
    ff.mark_seen(agent, f)
    assert ff.poll_drift(agent) == []

    await rollback(agent, ctx, AsyncMock())
    assert len(ff.poll_drift(agent)) == 1


@pytest.mark.asyncio
async def test_rollback_restores_compressor_state() -> None:
    comp = _compressor(count=5, archives=["x.md", "y.md", "z.md"])
    a = Message.system(SYS_PROMPT)
    snapshot = [a]
    agent = _agent(list(snapshot), compressor=comp)
    ctx = _TurnContext(
        snapshot_messages=[a.model_copy(deep=True)],
        snapshot_compressor_state=(2, ["x.md"]),
        snapshot_ids=frozenset({id(a)}),
        main_system_id=id(a),
        fs_state={},
    )
    save = AsyncMock()

    await rollback(agent, ctx, save)

    assert comp._compression_count == 2
    assert comp._archive_paths == ["x.md"]
    assert comp._last_result is None


@pytest.mark.asyncio
async def test_rollback_swallows_save_failure_in_memory_still_restored() -> None:
    a = Message.system(SYS_PROMPT)
    snapshot = [a]
    extra = Message.user("u")
    agent = _agent(list(snapshot) + [extra])
    ctx = _ctx_for(snapshot, main_idx=0)

    async def boom() -> None:
        raise RuntimeError("disk full")

    await rollback(agent, ctx, boom)

    msgs = agent.context.short_term_memory._messages
    assert len(msgs) == 1
    assert msgs[0].content == SYS_PROMPT


@pytest.mark.asyncio
async def test_rollback_does_not_touch_total_usage() -> None:
    a = Message.system(SYS_PROMPT)
    snapshot = [a]
    agent = _agent(list(snapshot))
    agent._total_usage = Usage(prompt_tokens=1500, completion_tokens=200, total_tokens=1700)
    ctx = _ctx_for(snapshot, main_idx=0)
    save = AsyncMock()

    await rollback(agent, ctx, save)

    assert agent._total_usage.prompt_tokens == 1500
    assert agent._total_usage.completion_tokens == 200
    assert agent._total_usage.total_tokens == 1700


# ── session persistence: make_save_session / switch_session plan_mode ─


def _agent_with_active_plan(active: bool) -> MagicMock:
    agent = MagicMock()
    agent.name = "x"
    agent._approval.mode = "auto"
    agent._approval.export_session_grants = MagicMock(return_value={})
    agent._session_created_at = None
    agent._session_metadata_extras = {"_plan_mode": active}
    agent.tool_registry.save_states = MagicMock(return_value={})

    state = MagicMock()
    state.metadata = {}

    def _to_session_state(session_id: str, **meta):
        state.created_at = None
        state.updated_at = None
        return state

    agent.context.to_session_state = _to_session_state

    plan_mode._active.clear()
    if active:
        plan_mode._active.add(id(agent))
    return agent


def teardown_function() -> None:
    plan_mode._active.clear()


async def test_save_session_records_plan_mode_true() -> None:
    agent = _agent_with_active_plan(True)
    backend = MagicMock()
    backend.session_id = "abc"
    backend.save_state = AsyncMock()

    save = make_save_session(agent, backend)
    await save()

    backend.save_state.assert_awaited_once()
    state = backend.save_state.await_args.args[0]
    assert state.metadata["_plan_mode"] is True


async def test_save_session_records_plan_mode_false() -> None:
    agent = _agent_with_active_plan(False)
    backend = MagicMock()
    backend.session_id = "abc"
    backend.save_state = AsyncMock()

    save = make_save_session(agent, backend)
    await save()

    state = backend.save_state.await_args.args[0]
    assert state.metadata["_plan_mode"] is False


async def test_switch_session_restores_plan_mode_from_metadata() -> None:
    agent = MagicMock()
    agent._collect_background_results = AsyncMock(return_value=False)
    agent._bg_manager.shutdown = AsyncMock()
    agent._bg_manager.get_all = MagicMock(return_value=[])
    agent._bg_manager._tasks = {}
    agent._sandbox.stop = AsyncMock()
    agent.apply_session_state = AsyncMock()

    loaded_state = MagicMock()
    loaded_state.metadata = {"_plan_mode": True}
    backend = MagicMock()
    backend.set_session_id = MagicMock()
    backend.load_state = AsyncMock(return_value=loaded_state)

    handler = MagicMock()
    handler.cancel_pending = MagicMock()

    plan_mode._active.clear()
    await switch_session(agent, backend, handler, AsyncMock(), "new-id")

    assert plan_mode.is_active(agent) is True


async def test_switch_session_exits_plan_mode_when_metadata_false() -> None:
    agent = MagicMock()
    agent._collect_background_results = AsyncMock(return_value=False)
    agent._bg_manager.shutdown = AsyncMock()
    agent._bg_manager.get_all = MagicMock(return_value=[])
    agent._bg_manager._tasks = {}
    agent._sandbox.stop = AsyncMock()
    agent.apply_session_state = AsyncMock()
    agent.reset_session_state = AsyncMock()

    loaded_state = MagicMock()
    loaded_state.metadata = {"_plan_mode": False}
    backend = MagicMock()
    backend.set_session_id = MagicMock()
    backend.load_state = AsyncMock(return_value=loaded_state)

    handler = MagicMock()
    handler.cancel_pending = MagicMock()

    plan_mode._active.add(id(agent))
    await switch_session(agent, backend, handler, AsyncMock(), "new-id")

    assert plan_mode.is_active(agent) is False


async def test_switch_session_fresh_session_resets_plan_mode() -> None:
    agent = MagicMock()
    agent._collect_background_results = AsyncMock(return_value=False)
    agent._bg_manager.shutdown = AsyncMock()
    agent._bg_manager.get_all = MagicMock(return_value=[])
    agent._bg_manager._tasks = {}
    agent._sandbox.stop = AsyncMock()
    agent.reset_session_state = AsyncMock()

    backend = MagicMock()
    backend.set_session_id = MagicMock()
    backend.load_state = AsyncMock(return_value=None)

    handler = MagicMock()
    handler.cancel_pending = MagicMock()

    plan_mode._active.add(id(agent))
    await switch_session(agent, backend, handler, AsyncMock(), "new-id")

    assert plan_mode.is_active(agent) is False
    agent.reset_session_state.assert_awaited_once_with("new-id")
