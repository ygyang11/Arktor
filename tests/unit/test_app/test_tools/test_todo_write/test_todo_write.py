"""Tests for TodoWriteTool."""
from __future__ import annotations

import pytest

from agent_app.tools.todo_write.todo_write import (
    MAX_CONTENT_LEN,
    MAX_TODOS,
    TodoItem,
    TodoStats,
    TodoWriteTool,
)
from agent_harness.core.errors import ToolValidationError
from agent_harness.hooks.base import DefaultHooks


# ── Data model ──


class TestTodoItem:
    def test_default_status(self) -> None:
        item = TodoItem(id="1", content="Task")
        assert item.status == "pending"

    def test_status_literal(self) -> None:
        for s in ("pending", "in_progress", "completed"):
            item = TodoItem(id="1", content="Task", status=s)  # type: ignore[arg-type]
            assert item.status == s


class TestTodoStats:
    def test_empty(self) -> None:
        s = TodoStats()
        assert s.total == 0 and s.pending == 0

    def test_counts(self) -> None:
        s = TodoStats(total=5, pending=2, in_progress=1, completed=2)
        assert s.total == 5


# ── Tool core ──


class TestTodoWriteTool:
    async def test_execute_basic(self) -> None:
        tool = TodoWriteTool()
        result = await tool.execute(
            todos=[
                {"id": "1", "content": "Explore", "status": "in_progress"},
                {"id": "2", "content": "Fix bug", "status": "pending"},
                {"id": "3", "content": "Test", "status": "pending"},
            ]
        )
        assert "[0/3]" in result
        assert "[>] Explore  <- current" in result
        assert "Todos updated successfully" in result

    async def test_execute_update_status(self) -> None:
        tool = TodoWriteTool()
        await tool.execute(
            todos=[
                {"id": "1", "content": "Step A", "status": "in_progress"},
                {"id": "2", "content": "Step B", "status": "pending"},
            ]
        )
        result = await tool.execute(
            todos=[
                {"id": "1", "content": "Step A", "status": "completed"},
                {"id": "2", "content": "Step B", "status": "in_progress"},
            ]
        )
        assert "[1/2]" in result
        assert "[>] Step B  <- current" in result

    async def test_execute_empty_list(self) -> None:
        tool = TodoWriteTool()
        await tool.execute(
            todos=[{"id": "1", "content": "X", "status": "pending"}]
        )
        result = await tool.execute(todos=[])
        assert result == "Task list cleared."
        assert tool.todos == []

    async def test_execute_all_completed(self) -> None:
        tool = TodoWriteTool()
        result = await tool.execute(
            todos=[
                {"id": "1", "content": "Done", "status": "completed"},
                {"id": "2", "content": "Also done", "status": "completed"},
            ]
        )
        assert "All tasks completed" in result
        assert "[2/2]" in result

    async def test_execute_invalid_not_list(self) -> None:
        tool = TodoWriteTool()
        with pytest.raises(ToolValidationError, match="must be a list"):
            await tool.execute(todos="not a list")

    async def test_execute_missing_id(self) -> None:
        tool = TodoWriteTool()
        with pytest.raises(ToolValidationError, match="'id' is required"):
            await tool.execute(
                todos=[{"content": "X", "status": "pending"}]
            )

    async def test_execute_missing_content(self) -> None:
        tool = TodoWriteTool()
        with pytest.raises(ToolValidationError, match="'content' is required"):
            await tool.execute(
                todos=[{"id": "1", "content": "", "status": "pending"}]
            )

    async def test_execute_duplicate_id(self) -> None:
        tool = TodoWriteTool()
        with pytest.raises(ToolValidationError, match="Duplicate id"):
            await tool.execute(
                todos=[
                    {"id": "1", "content": "A", "status": "pending"},
                    {"id": "1", "content": "B", "status": "pending"},
                ]
            )

    async def test_execute_invalid_status(self) -> None:
        tool = TodoWriteTool()
        with pytest.raises(ToolValidationError, match="invalid status"):
            await tool.execute(
                todos=[{"id": "1", "content": "A", "status": "done"}]
            )

    async def test_execute_multiple_in_progress(self) -> None:
        tool = TodoWriteTool()
        with pytest.raises(ToolValidationError, match="At most 1"):
            await tool.execute(
                todos=[
                    {"id": "1", "content": "A", "status": "in_progress"},
                    {"id": "2", "content": "B", "status": "in_progress"},
                ]
            )

    async def test_execute_over_max_todos(self) -> None:
        tool = TodoWriteTool()
        todos = [
            {"id": str(i), "content": f"Task {i}", "status": "pending"}
            for i in range(MAX_TODOS + 1)
        ]
        with pytest.raises(ToolValidationError, match=f"maximum is {MAX_TODOS}"):
            await tool.execute(todos=todos)

    async def test_execute_content_too_long(self) -> None:
        tool = TodoWriteTool()
        with pytest.raises(ToolValidationError, match=f"maximum is {MAX_CONTENT_LEN}"):
            await tool.execute(
                todos=[
                    {
                        "id": "1",
                        "content": "x" * (MAX_CONTENT_LEN + 1),
                        "status": "pending",
                    }
                ]
            )

    async def test_validate_collects_all_errors(self) -> None:
        tool = TodoWriteTool()
        with pytest.raises(ToolValidationError, match="3 validation error"):
            await tool.execute(
                todos=[
                    {"id": "", "content": "A", "status": "pending"},
                    {"id": "2", "content": "", "status": "pending"},
                    {"id": "3", "content": "C", "status": "bad"},
                ]
            )

    async def test_execute_coerces_int_id(self) -> None:
        tool = TodoWriteTool()
        await tool.execute(
            todos=[{"id": 1, "content": "X", "status": "pending"}]
        )
        assert tool.todos[0].id == "1"

    async def test_execute_coerces_int_content(self) -> None:
        tool = TodoWriteTool()
        await tool.execute(
            todos=[{"id": "1", "content": 123, "status": "pending"}]
        )
        assert tool.todos[0].content == "123"

    async def test_execute_non_dict_item(self) -> None:
        tool = TodoWriteTool()
        with pytest.raises(ToolValidationError, match="must be an object"):
            await tool.execute(todos=["just a string"])

    async def test_execute_none_item(self) -> None:
        tool = TodoWriteTool()
        with pytest.raises(ToolValidationError, match="must be an object"):
            await tool.execute(todos=[None])

    async def test_execute_strips_whitespace(self) -> None:
        tool = TodoWriteTool()
        await tool.execute(
            todos=[{"id": " 1 ", "content": "  hi  ", "status": "pending"}]
        )
        assert tool.todos[0].id == "1"
        assert tool.todos[0].content == "hi"

    async def test_todos_property_returns_copy(self) -> None:
        tool = TodoWriteTool()
        await tool.execute(
            todos=[{"id": "1", "content": "X", "status": "pending"}]
        )
        copy = tool.todos
        copy.clear()
        assert len(tool.todos) == 1

    async def test_stats_property(self) -> None:
        tool = TodoWriteTool()
        await tool.execute(
            todos=[
                {"id": "1", "content": "A", "status": "completed"},
                {"id": "2", "content": "B", "status": "in_progress"},
                {"id": "3", "content": "C", "status": "pending"},
            ]
        )
        s = tool.stats
        assert s.total == 3
        assert s.completed == 1
        assert s.in_progress == 1
        assert s.pending == 1

    def test_schema(self) -> None:
        tool = TodoWriteTool()
        schema = tool.get_schema()
        assert schema.name == "todo_write"
        props = schema.parameters["properties"]["todos"]["items"]["properties"]
        assert "id" in props
        assert "content" in props
        assert "status" in props
        assert "Short stable task identifier" in props["id"]["description"]
        assert "imperative form" in props["content"]["description"]


# ── Recap output ──


class TestRecap:
    async def test_recap_with_in_progress(self) -> None:
        tool = TodoWriteTool()
        result = await tool.execute(
            todos=[
                {"id": "1", "content": "Done", "status": "completed"},
                {"id": "2", "content": "Working", "status": "in_progress"},
                {"id": "3", "content": "Next", "status": "pending"},
            ]
        )
        assert "[>] Working  <- current" in result
        assert "[ ] Next" in result

    async def test_recap_lists_all_pending(self) -> None:
        tool = TodoWriteTool()
        result = await tool.execute(
            todos=[
                {"id": "1", "content": "A", "status": "in_progress"},
                {"id": "2", "content": "B", "status": "pending"},
                {"id": "3", "content": "C", "status": "pending"},
                {"id": "4", "content": "D", "status": "pending"},
                {"id": "5", "content": "E", "status": "pending"},
            ]
        )
        for name in ("A", "B", "C", "D", "E"):
            assert f"] {name}" in result

    async def test_recap_no_tasks(self) -> None:
        tool = TodoWriteTool()
        result = await tool.execute(todos=[])
        assert result == "Task list cleared."

    async def test_recap_all_done(self) -> None:
        tool = TodoWriteTool()
        result = await tool.execute(
            todos=[{"id": "1", "content": "X", "status": "completed"}]
        )
        assert "All tasks completed" in result

    async def test_recap_includes_guidance(self) -> None:
        tool = TodoWriteTool()
        result = await tool.execute(
            todos=[{"id": "1", "content": "X", "status": "in_progress"}]
        )
        assert "Proceed with the current task" in result


# ── Stateful protocol ──


class TestStatefulProtocol:
    async def test_get_state_empty(self) -> None:
        tool = TodoWriteTool()
        assert tool.get_state() is None

    async def test_get_state_with_todos(self) -> None:
        tool = TodoWriteTool()
        await tool.execute(
            todos=[{"id": "1", "content": "X", "status": "pending"}]
        )
        state = tool.get_state()
        assert state is not None
        assert len(state["todos"]) == 1
        assert state["todos"][0]["id"] == "1"

    async def test_restore_state(self) -> None:
        tool = TodoWriteTool()
        tool.restore_state(
            {"todos": [{"id": "1", "content": "Restored", "status": "in_progress"}]}
        )
        assert len(tool.todos) == 1
        assert tool.todos[0].content == "Restored"

    async def test_notify_state(self) -> None:
        tool = TodoWriteTool()
        await tool.execute(
            todos=[{"id": "1", "content": "X", "status": "pending"}]
        )

        called_with: dict[str, list[dict[str, str]] | dict[str, int]] = {}

        class TestHooks(DefaultHooks):
            async def on_todo_update(
                self,
                agent_name: str,
                todos: list[dict[str, str]],
                stats: dict[str, int],
            ) -> None:
                called_with["todos"] = todos
                called_with["stats"] = stats

        hooks = TestHooks()
        await tool.notify_state(hooks, "test-agent")
        assert "todos" in called_with
        assert len(called_with["todos"]) == 1  # type: ignore[arg-type]
