import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_cli.commands.builtin.clear import CMD
from agent_cli.runtime.goal import mode as goal_mode

from ..conftest import render_output


async def test_clear_resets_all_runtime_state() -> None:
    agent = MagicMock()
    agent._bg_manager.shutdown = AsyncMock()
    agent._bg_manager._tasks = {"bg_001": object(), "bg_002": object()}
    agent.context.short_term_memory.clear = AsyncMock()
    agent.context.working_memory.clear = AsyncMock()

    tool_a = MagicMock(spec=["reset_state"])
    tool_b = MagicMock(spec=["reset_state"])
    agent.tool_registry.list_tools = MagicMock(return_value=[tool_a, tool_b])
    agent._reset_stateful_tools = lambda: [t.reset_state() for t in agent.tool_registry.list_tools()]

    agent._approval = MagicMock()
    agent._sandbox.stop = AsyncMock()
    agent.context.state = MagicMock()
    compressor = MagicMock()
    agent.context.short_term_memory.compressor = compressor

    save = AsyncMock()
    ctx = MagicMock(agent=agent, save_session=save)

    result = await CMD.handler(ctx, "")

    agent._bg_manager.shutdown.assert_awaited_once()
    assert agent._bg_manager._tasks == {}
    tool_a.reset_state.assert_called_once()
    tool_b.reset_state.assert_called_once()
    agent._approval.reset_session.assert_called_once()
    agent.context.state.reset.assert_called_once()
    compressor.restore_runtime_state.assert_called_once_with([])
    save.assert_awaited_once()
    assert "Context cleared" in render_output(result.output)


async def test_clear_without_compressor_does_not_crash() -> None:
    agent = MagicMock()
    agent._bg_manager.shutdown = AsyncMock()
    agent._bg_manager._tasks = {}
    agent.context.short_term_memory.clear = AsyncMock()
    agent.context.short_term_memory.compressor = None
    agent.context.working_memory.clear = AsyncMock()
    agent.tool_registry.list_tools = MagicMock(return_value=[])
    agent._approval = None
    agent._sandbox.stop = AsyncMock()
    agent.context.state = MagicMock()

    result = await CMD.handler(MagicMock(agent=agent, save_session=AsyncMock()), "")
    assert "Context cleared" in render_output(result.output)


async def test_clear_cancels_pending_approvals_after_bg_shutdown() -> None:
    """/clear must drain handler's pending approval queue so orphan approvals
    from cancelled bg tasks don't leak into the next REPL iteration."""
    agent = MagicMock()
    agent._bg_manager.shutdown = AsyncMock()
    agent._bg_manager._tasks = {}
    agent.context.short_term_memory.clear = AsyncMock()
    agent.context.short_term_memory.compressor = None
    agent.context.working_memory.clear = AsyncMock()
    agent.tool_registry.list_tools = MagicMock(return_value=[])
    agent._approval = None
    agent._sandbox.stop = AsyncMock()
    agent.context.state = MagicMock()

    approval_handler = MagicMock()
    ctx = MagicMock(
        agent=agent,
        save_session=AsyncMock(),
        approval_handler=approval_handler,
    )

    await CMD.handler(ctx, "")
    approval_handler.cancel_pending.assert_called_once()


async def test_clear_phase_a_cancel_leaves_state_untouched() -> None:
    """Ctrl+C during stop_sandbox (phase A) must not clear in-memory state."""
    agent = MagicMock()
    agent._bg_manager.shutdown = AsyncMock()
    agent._bg_manager._tasks = {"bg_001": object()}
    agent._sandbox.stop = AsyncMock(side_effect=asyncio.CancelledError())

    messages = ["m1", "m2"]
    agent.context.short_term_memory._messages = messages
    agent.context.short_term_memory.clear = AsyncMock()
    agent.context.working_memory.clear = AsyncMock()
    agent.context.state = MagicMock()

    save = AsyncMock()
    ctx = MagicMock(agent=agent, save_session=save)

    with pytest.raises(asyncio.CancelledError):
        await CMD.handler(ctx, "")

    # Phase B should not have run: no clears, no state reset, no save.
    agent.context.short_term_memory.clear.assert_not_awaited()
    agent.context.working_memory.clear.assert_not_awaited()
    agent.context.state.reset.assert_not_called()
    save.assert_not_awaited()
    # clear_tasks() never ran — bg_manager dict still populated.
    assert agent._bg_manager._tasks


async def test_clear_removes_goal_before_save() -> None:
    agent = MagicMock()
    agent._session_metadata_extras = {}
    agent.context.usage_meter.total.total_tokens = 0
    agent._bg_manager.shutdown = AsyncMock()
    agent._bg_manager._tasks = {}
    agent.context.short_term_memory.clear = AsyncMock()
    agent.context.short_term_memory.compressor = None
    agent.context.working_memory.clear = AsyncMock()
    agent.tool_registry.list_tools = MagicMock(return_value=[])
    agent._reset_stateful_tools = MagicMock()
    agent._approval = MagicMock()
    agent._sandbox.stop = AsyncMock()
    agent.context.state = MagicMock()
    goal_mode.begin(agent, "x")

    async def save() -> None:
        assert goal_mode.get_state(agent) is None
        assert "_goal" not in agent._session_metadata_extras

    ctx = MagicMock(
        agent=agent,
        save_session=AsyncMock(side_effect=save),
        approval_handler=MagicMock(),
    )
    await CMD.handler(ctx, "")
