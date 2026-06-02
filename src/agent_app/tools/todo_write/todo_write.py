"""TodoWrite — declarative task list management for multi-step work."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from agent_harness.core.errors import ToolValidationError
from agent_harness.hooks.base import DefaultHooks
from agent_harness.tool.base import BaseTool, ToolSchema

MAX_TODOS = 20
MAX_CONTENT_LEN = 200


class TodoItem(BaseModel):
    id: str
    content: str
    status: Literal["pending", "in_progress", "completed"] = "pending"


class TodoStats(BaseModel):
    total: int = 0
    pending: int = 0
    in_progress: int = 0
    completed: int = 0


class TodoWriteTool(BaseTool):
    """Declarative task list management tool."""

    def __init__(self) -> None:
        super().__init__(
            name="todo_write",
            description=(
                "Create and manage a task list for the current session. "
                "Use proactively for multi-step work to track progress. "
                "Submit the complete list each time (declarative replacement). "
                "Keep exactly one task in_progress at all times while working."
            ),
        )
        self._todos: list[TodoItem] = []

    @property
    def todos(self) -> list[TodoItem]:
        """Current task list (read-only copy)."""
        return list(self._todos)

    @property
    def stats(self) -> TodoStats:
        counts: dict[str, int] = {"pending": 0, "in_progress": 0, "completed": 0}
        for t in self._todos:
            counts[t.status] += 1
        return TodoStats(total=len(self._todos), **counts)

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": (
                            "Complete task list (declarative replacement). "
                            "Submit ALL tasks every time, not just changes."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": (
                                        "Short stable task identifier "
                                        "(e.g. '1', 'Step1', 'Phase2')."
                                    ),
                                },
                                "content": {
                                    "type": "string",
                                    "description": (
                                        "Actionable task description in imperative form "
                                        "(e.g. 'Fix auth bug', 'Write unit tests')."
                                    ),
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "description": (
                                        "pending = not started, "
                                        "in_progress = actively working (max 1), "
                                        "completed = done."
                                    ),
                                },
                            },
                            "required": ["id", "content", "status"],
                        },
                    },
                },
                "required": ["todos"],
            },
        )

    async def execute(self, **kwargs: Any) -> str:
        raw_todos = kwargs.get("todos")
        if not isinstance(raw_todos, list):
            raise ToolValidationError("'todos' must be a list.")

        self._todos = self._validate(raw_todos)
        return self._build_recap()

    def _validate(self, todos: list[Any]) -> list[TodoItem]:
        errors: list[str] = []
        normalized: list[tuple[str, str, Any]] = []

        if len(todos) > MAX_TODOS:
            errors.append(
                f"{len(todos)} tasks submitted, maximum is {MAX_TODOS}. "
                "Reduce the list and resubmit."
            )

        in_progress_count = 0
        seen_ids: set[str] = set()

        for i, item in enumerate(todos):
            if not isinstance(item, dict):
                errors.append(f"Todo #{i + 1}: must be an object.")
                continue

            tid = str(item.get("id", "")).strip()
            if not tid:
                errors.append(f"Todo #{i + 1}: 'id' is required.")
                continue

            content = str(item.get("content", "")).strip()
            if not content:
                errors.append(f"Todo '{tid}': 'content' is required.")

            if content and len(content) > MAX_CONTENT_LEN:
                errors.append(
                    f"Task '{tid}' content is {len(content)} chars, "
                    f"maximum is {MAX_CONTENT_LEN}. Shorten and resubmit."
                )

            if tid in seen_ids:
                errors.append(f"Duplicate id: '{tid}'.")
            seen_ids.add(tid)

            status = item.get("status", "")
            if status not in ("pending", "in_progress", "completed"):
                errors.append(f"Todo '{tid}': invalid status '{status}'.")

            if status == "in_progress":
                in_progress_count += 1

            normalized.append((tid, content, status))

        if in_progress_count > 1:
            errors.append(
                f"At most 1 task can be in_progress, got {in_progress_count}."
            )

        if errors:
            raise ToolValidationError(
                f"{len(errors)} validation error(s): " + ";\n".join(errors)
            )

        return [TodoItem(id=t, content=c, status=s) for t, c, s in normalized]

    def _build_recap(self) -> str:
        s = self.stats
        if s.total == 0:
            return "Task list cleared."

        lines = [f"[{s.completed}/{s.total}]"]
        for t in self._todos:
            if t.status == "completed":
                lines.append(f"  - [x] {t.content}")
            elif t.status == "in_progress":
                lines.append(f"  - [>] {t.content}  <- current")
            else:
                lines.append(f"  - [ ] {t.content}")

        if s.completed == s.total:
            lines.append("\nAll tasks completed. Todos updated successfully.")
        else:
            lines.append(
                "\nTodos updated successfully. Ensure that you continue to use "
                "the todo list to track your progress. "
                "Proceed with the current task. "
                "Call todo_write again when it is done."
            )
        return "\n".join(lines)

    # ── Stateful tool protocol ──

    def get_state(self) -> dict[str, Any] | None:
        if not self._todos:
            return None
        return {"todos": [t.model_dump() for t in self._todos]}

    def restore_state(self, state: dict[str, Any]) -> None:
        self._todos = [TodoItem(**t) for t in state.get("todos", [])]

    def reset_state(self) -> None:
        self._todos.clear()

    async def notify_state(self, hooks: DefaultHooks, agent_name: str) -> None:
        await hooks.on_todo_update(
            agent_name,
            [t.model_dump() for t in self._todos],
            self.stats.model_dump(),
        )


todo_write = TodoWriteTool()
