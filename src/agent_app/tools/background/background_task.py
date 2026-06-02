"""Background task management tool — list, status, cancel."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from agent_harness.core.errors import ToolValidationError
from agent_harness.tool.base import BaseTool, ToolSchema

BACKGROUND_TASK_TOOL_DESCRIPTION = """\
View and manage background tasks. Results are automatically delivered \
when tasks complete — you do not need to poll or check proactively.

## Actions

- **list** (action only): Show all previously submitted tasks with task_id, \
status, tool, description, and elapsed time. Includes all statuses: \
running, completed, failed, cancelled.
- **status** (action + task_id): Detailed info for one task. When completed, \
includes result summary and output file path.
- **cancel** (action + task_id): Stop a running task. Only running tasks \
can be cancelled.\
"""


class BackgroundTaskTool(BaseTool):

    def __init__(self) -> None:
        super().__init__(
            name="background_task",
            description=BACKGROUND_TASK_TOOL_DESCRIPTION,
        )
        self._agent: Any = None

    def bind_agent(self, agent: Any) -> None:
        self._agent = agent

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "status", "cancel"],
                        "description": (
                            "'list' all tasks, 'status' for one task detail, "
                            "'cancel' a running task."
                        ),
                    },
                    "task_id": {
                        "type": "string",
                        "description": (
                            "The task id (e.g. 'bg_001') returned when the background task "
                            "was started. Required for 'status' and 'cancel' actions."
                        ),
                    },
                },
                "required": ["action"],
            },
        )

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")
        task_id = kwargs.get("task_id", "")
        manager = self._agent._bg_manager

        if action == "list":
            return self._list(manager)
        if action == "status":
            if not task_id:
                raise ToolValidationError("task_id is required for 'status'")
            return self._status(manager, task_id)
        if action == "cancel":
            if not task_id:
                raise ToolValidationError("task_id is required for 'cancel'")
            return self._cancel(manager, task_id)
        raise ToolValidationError(f"Unknown action: {action}")

    def _list(self, manager: Any) -> str:
        tasks = manager.get_all()
        if not tasks:
            return "No background tasks."
        lines = [f"{len(tasks)} background task(s):"]
        for t in tasks:
            elapsed = (datetime.now() - t.created_at).total_seconds()
            lines.append(
                f"  [{t.task_id}] {t.tool_name} — {t.status} — "
                f"{t.description} [{elapsed:.0f}s]"
            )
        return "\n".join(lines)

    def _status(self, manager: Any, task_id: str) -> str:
        task = manager.get_task(task_id)
        if not task:
            return f"Error: task {task_id} not found."
        lines = [
            f"Task: {task.task_id}",
            f"Tool: {task.tool_name}",
            f"Description: {task.description}",
            f"Status: {task.status}",
        ]
        if task.result and task.result.output_path:
            lines.append(f"Output: {task.result.output_path}")
        if task.result and task.result.summary:
            lines.append(f"Summary:\n{task.result.summary}")
        if task.error:
            lines.append(f"Error: {task.error}")
        return "\n".join(lines)

    def _cancel(self, manager: Any, task_id: str) -> str:
        if manager.cancel(task_id):
            return f"Task {task_id} cancelled successfully."
        task = manager.get_task(task_id)
        if not task:
            return f"Error: task {task_id} not found."
        if task.status == "running":
            return f"Error: task {task_id} could not be cancelled (internal error)."
        return f"Error: task {task_id} is already {task.status}, cannot cancel."


background_task = BackgroundTaskTool()
