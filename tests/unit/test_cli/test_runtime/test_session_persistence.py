"""Tests that make_save_session / switch_session persist plan_mode."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_cli.runtime import plan_mode
from agent_cli.runtime.session import make_save_session, switch_session


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
