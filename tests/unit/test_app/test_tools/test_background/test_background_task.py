"""Unit tests for BackgroundTaskTool."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_app.tools.background.background_task import BackgroundTaskTool
from agent_harness.background import BackgroundResult, BackgroundTask, BackgroundTaskManager
from agent_harness.core.errors import ToolValidationError


def _make_task(
    task_id: str = "bg_001",
    tool_name: str = "terminal",
    description: str = "pytest",
    status: str = "running",
    result: BackgroundResult | None = None,
    error: str | None = None,
) -> BackgroundTask:
    return BackgroundTask(
        task_id=task_id,
        tool_name=tool_name,
        description=description,
        asyncio_task=None,
        status=status,
        result=result,
        error=error,
    )


@pytest.fixture
def tool() -> BackgroundTaskTool:
    t = BackgroundTaskTool()
    manager = BackgroundTaskManager()
    agent = SimpleNamespace(_bg_manager=manager)
    t.bind_agent(agent)
    return t


@pytest.fixture
def tool_with_tasks(tool: BackgroundTaskTool) -> BackgroundTaskTool:
    mgr = tool._agent._bg_manager
    mgr._tasks["bg_001"] = _make_task("bg_001", status="running")
    mgr._tasks["bg_002"] = _make_task(
        "bg_002", status="completed",
        result=BackgroundResult(summary="8 passed", output_path="/tmp/bg_002.txt"),
    )
    mgr._tasks["bg_003"] = _make_task(
        "bg_003", status="failed", error="timeout",
    )
    mgr._tasks["bg_004"] = _make_task("bg_004", status="cancelled")
    return tool


# -- list --


class TestList:
    async def test_empty(self, tool: BackgroundTaskTool) -> None:
        result = await tool.execute(action="list")
        assert result == "No background tasks."

    async def test_mixed_statuses(self, tool_with_tasks: BackgroundTaskTool) -> None:
        result = await tool_with_tasks.execute(action="list")
        assert "4 background task(s)" in result
        assert "running" in result
        assert "completed" in result
        assert "failed" in result
        assert "cancelled" in result

    async def test_shows_elapsed(self, tool_with_tasks: BackgroundTaskTool) -> None:
        result = await tool_with_tasks.execute(action="list")
        assert "s]" in result


# -- status --


class TestStatus:
    async def test_completed_with_output(self, tool_with_tasks: BackgroundTaskTool) -> None:
        result = await tool_with_tasks.execute(action="status", task_id="bg_002")
        assert "completed" in result
        assert "8 passed" in result
        assert "/tmp/bg_002.txt" in result

    async def test_failed_with_error(self, tool_with_tasks: BackgroundTaskTool) -> None:
        result = await tool_with_tasks.execute(action="status", task_id="bg_003")
        assert "failed" in result
        assert "timeout" in result

    async def test_running(self, tool_with_tasks: BackgroundTaskTool) -> None:
        result = await tool_with_tasks.execute(action="status", task_id="bg_001")
        assert "running" in result

    async def test_not_found(self, tool: BackgroundTaskTool) -> None:
        result = await tool.execute(action="status", task_id="bg_999")
        assert "not found" in result

    async def test_missing_task_id(self, tool: BackgroundTaskTool) -> None:
        with pytest.raises(ToolValidationError, match="task_id is required"):
            await tool.execute(action="status")

    async def test_running_truncated_label(
        self, tool: BackgroundTaskTool, tmp_path: Path,
    ) -> None:
        mgr = tool._agent._bg_manager
        (tmp_path / "bg_050.txt").write_text("\n".join(str(i) for i in range(40)) + "\n")
        mgr._tasks["bg_050"] = BackgroundTask(
            task_id="bg_050", tool_name="terminal", description="t",
            asyncio_task=None, output_dir=str(tmp_path), streaming=True, status="running",
        )
        result = await tool.execute(action="status", task_id="bg_050")
        assert "Live streaming output (last 30 lines)" in result

    async def test_running_full_label(
        self, tool: BackgroundTaskTool, tmp_path: Path,
    ) -> None:
        mgr = tool._agent._bg_manager
        (tmp_path / "bg_051.txt").write_text("a\nb\nc\n")
        mgr._tasks["bg_051"] = BackgroundTask(
            task_id="bg_051", tool_name="terminal", description="t",
            asyncio_task=None, output_dir=str(tmp_path), streaming=True, status="running",
        )
        result = await tool.execute(action="status", task_id="bg_051")
        assert "Live streaming output:" in result
        assert "last 30 lines" not in result

    async def test_running_no_file_hint(
        self, tool: BackgroundTaskTool, tmp_path: Path,
    ) -> None:
        mgr = tool._agent._bg_manager
        mgr._tasks["bg_052"] = BackgroundTask(
            task_id="bg_052", tool_name="terminal", description="t",
            asyncio_task=None, output_dir=str(tmp_path), streaming=True, status="running",
        )
        result = await tool.execute(action="status", task_id="bg_052")
        assert "Still running" in result


# -- cancel --


class TestCancel:
    async def test_running_success(self, tool_with_tasks: BackgroundTaskTool) -> None:
        # Need a real asyncio task for cancel to work
        mgr = tool_with_tasks._agent._bg_manager

        async def _slow() -> tuple[str, str]:
            await asyncio.sleep(100)
            return "", ""

        mgr._tasks["bg_010"] = BackgroundTask(
            task_id="bg_010", tool_name="terminal", description="slow",
            asyncio_task=asyncio.create_task(_slow()),
            status="running",
        )
        result = await tool_with_tasks.execute(action="cancel", task_id="bg_010")
        assert "cancelled" in result

    async def test_already_done(self, tool_with_tasks: BackgroundTaskTool) -> None:
        result = await tool_with_tasks.execute(action="cancel", task_id="bg_002")
        assert "already completed" in result

    async def test_not_found(self, tool: BackgroundTaskTool) -> None:
        result = await tool.execute(action="cancel", task_id="bg_999")
        assert "not found" in result

    async def test_missing_task_id(self, tool: BackgroundTaskTool) -> None:
        with pytest.raises(ToolValidationError, match="task_id is required"):
            await tool.execute(action="cancel")


# -- unknown action --


class TestUnknown:
    async def test_unknown_action(self, tool: BackgroundTaskTool) -> None:
        with pytest.raises(ToolValidationError, match="Unknown action"):
            await tool.execute(action="explode")
