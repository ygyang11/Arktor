from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from agent_cli.commands.builtin.tasks import CMD
from agent_harness.background import BackgroundTask

from .conftest import render_output


def _bg(
    task_id: str,
    status: str,
    *,
    tool: str = "read_file",
    desc: str = "",
    minutes_ago: int = 0,
) -> BackgroundTask:
    return BackgroundTask(
        task_id=task_id,
        tool_name=tool,
        description=desc,
        asyncio_task=None,
        status=status,
        created_at=datetime.now() - timedelta(minutes=minutes_ago),
    )


def _ctx(tasks: list[BackgroundTask], save: AsyncMock | None = None) -> MagicMock:
    agent = MagicMock()
    agent._bg_manager.get_all = MagicMock(return_value=tasks)
    return MagicMock(agent=agent, save_session=save or AsyncMock())


async def test_tasks_empty_returns_soft_message() -> None:
    result = await CMD.handler(_ctx([]), "")
    assert "No background tasks" in render_output(result.output)


async def test_tasks_renders_panel_with_title() -> None:
    result = await CMD.handler(_ctx([_bg("bg_001", "running", desc="npm test")]), "")
    out = render_output(result.output)
    assert "Background tasks" in out
    assert "npm test" in out
    assert "running" in out


async def test_tasks_sorted_running_first_then_completed_then_failed() -> None:
    tasks = [
        _bg("bg_004", "cancelled", minutes_ago=4),
        _bg("bg_003", "failed",    minutes_ago=3),
        _bg("bg_002", "completed", minutes_ago=2),
        _bg("bg_001", "running",   minutes_ago=1),
    ]
    out = render_output((await CMD.handler(_ctx(tasks), "")).output)
    idx_running   = out.index("running")
    idx_completed = out.index("completed")
    idx_failed    = out.index("failed")
    idx_cancelled = out.index("cancelled")
    assert idx_running < idx_completed < idx_failed < idx_cancelled


async def test_tasks_shows_truncated_id_and_tool_name() -> None:
    t = _bg("bg_abcdefghij", "running", tool="terminal_tool", desc="run tests")
    out = render_output((await CMD.handler(_ctx([t]), "")).output)
    assert "bg_abcde" in out
    assert "terminal_tool" in out


async def test_tasks_description_dash_when_empty() -> None:
    t = _bg("bg_001", "completed", desc="")
    out = render_output((await CMD.handler(_ctx([t]), "")).output)
    assert "—" in out


async def test_tasks_cancel_missing_arg_returns_err() -> None:
    result = await CMD.handler(_ctx([]), "cancel")
    assert "Missing task id" in render_output(result.output)


async def test_tasks_cancel_unknown_subcommand_returns_err() -> None:
    result = await CMD.handler(_ctx([]), "destroy")
    assert "Unknown subcommand" in render_output(result.output)


async def test_tasks_cancel_routes_to_cancel_with_note(monkeypatch) -> None:
    called_with: list[str] = []

    async def fake_cancel(agent, task_id):
        called_with.append(task_id)
        return True

    monkeypatch.setattr(
        "agent_cli.commands.builtin.tasks.background.cancel_with_note", fake_cancel,
    )

    save = AsyncMock()
    ctx = _ctx([], save=save)
    result = await CMD.handler(ctx, "cancel bg_001")
    assert called_with == ["bg_001"]
    assert "Cancelled" in render_output(result.output)
    save.assert_awaited_once()


async def test_tasks_cancel_unknown_id_returns_err(monkeypatch) -> None:
    async def fake_cancel(agent, task_id):
        return False

    monkeypatch.setattr(
        "agent_cli.commands.builtin.tasks.background.cancel_with_note", fake_cancel,
    )

    save = AsyncMock()
    ctx = _ctx([], save=save)
    result = await CMD.handler(ctx, "cancel bg_999")
    assert "No running task" in render_output(result.output)
    save.assert_not_awaited()


async def test_tasks_cancel_all_with_nothing_running_returns_soft(monkeypatch) -> None:
    async def fake_cancel_all(agent):
        return []

    monkeypatch.setattr(
        "agent_cli.commands.builtin.tasks.background.cancel_all_with_note",
        fake_cancel_all,
    )

    save = AsyncMock()
    ctx = _ctx([], save=save)
    result = await CMD.handler(ctx, "cancel all")
    assert "No running tasks to cancel" in render_output(result.output)
    save.assert_not_awaited()


async def test_tasks_cancel_all_iterates_and_saves(monkeypatch) -> None:
    async def fake_cancel_all(agent):
        return ["bg_001", "bg_002"]

    monkeypatch.setattr(
        "agent_cli.commands.builtin.tasks.background.cancel_all_with_note",
        fake_cancel_all,
    )

    save = AsyncMock()
    ctx = _ctx([], save=save)
    result = await CMD.handler(ctx, "cancel all")
    assert "Cancelled 2 tasks" in render_output(result.output)
    save.assert_awaited_once()
