from unittest.mock import AsyncMock, MagicMock

from agent_cli.runtime.background import cancel_all_with_note, cancel_with_note
from agent_harness.background import BackgroundTask
from agent_harness.core.message import Role


def _bg(task_id: str, status: str, *, tool: str = "terminal_tool", desc: str = "x") -> BackgroundTask:
    return BackgroundTask(
        task_id=task_id,
        tool_name=tool,
        description=desc,
        asyncio_task=None,
        status=status,
    )


def _make_agent(tasks: list[BackgroundTask]) -> MagicMock:
    agent = MagicMock()
    agent._bg_manager.get_all = MagicMock(return_value=tasks)

    def _get_task(tid: str) -> BackgroundTask | None:
        for t in tasks:
            if t.task_id == tid:
                return t
        return None

    def _cancel(tid: str) -> bool:
        task = _get_task(tid)
        if task is None or task.status != "running":
            return False
        task.status = "cancelled"
        return True

    agent._bg_manager.get_task = MagicMock(side_effect=_get_task)
    agent._bg_manager.cancel = MagicMock(side_effect=_cancel)
    agent.context.short_term_memory.add_message = AsyncMock()
    return agent


async def test_cancel_with_note_single_running_task_emits_system_message() -> None:
    t = _bg("bg_001", "running", tool="terminal_tool", desc="npm test")
    agent = _make_agent([t])
    cancelled = await cancel_with_note(agent, "bg_001")
    assert cancelled is True
    assert t.status == "cancelled"
    agent.context.short_term_memory.add_message.assert_awaited_once()
    msg = agent.context.short_term_memory.add_message.await_args.args[0]
    assert msg.role == Role.SYSTEM
    assert "[Background Task Cancelled]" in msg.content
    assert "bg_001" in msg.content
    assert "terminal_tool" in msg.content
    assert "npm test" in msg.content
    assert "**Cancelled By User**" in msg.content
    assert msg.metadata.get("is_background_result") is True


async def test_cancel_with_note_already_completed_returns_false() -> None:
    t = _bg("bg_001", "completed")
    agent = _make_agent([t])
    assert await cancel_with_note(agent, "bg_001") is False
    agent.context.short_term_memory.add_message.assert_not_awaited()


async def test_cancel_with_note_unknown_id_returns_false() -> None:
    agent = _make_agent([])
    assert await cancel_with_note(agent, "bg_999") is False
    agent.context.short_term_memory.add_message.assert_not_awaited()


async def test_cancel_all_with_note_picks_only_running() -> None:
    tasks = [
        _bg("bg_001", "running"),
        _bg("bg_002", "completed"),
        _bg("bg_003", "running"),
        _bg("bg_004", "failed"),
    ]
    agent = _make_agent(tasks)
    cancelled = await cancel_all_with_note(agent)
    assert set(cancelled) == {"bg_001", "bg_003"}
    assert agent.context.short_term_memory.add_message.await_count == 2


async def test_cancel_all_with_note_empty_when_nothing_running() -> None:
    tasks = [_bg("bg_001", "completed"), _bg("bg_002", "failed")]
    agent = _make_agent(tasks)
    cancelled = await cancel_all_with_note(agent)
    assert cancelled == []
    agent.context.short_term_memory.add_message.assert_not_awaited()
