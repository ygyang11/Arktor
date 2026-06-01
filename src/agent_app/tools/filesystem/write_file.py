"""Create new files only."""

from __future__ import annotations

import logging
from typing import Any

from agent_app.observability.file_freshness import mark_read
from agent_app.tools.filesystem._security import (
    normalize_path,
    relative_to_workspace,
)
from agent_harness.agent.base import BaseAgent
from agent_harness.core.errors import ToolValidationError
from agent_harness.tool.base import BaseTool, ToolSchema

logger = logging.getLogger(__name__)

WRITE_FILE_DESCRIPTION = (
    "Writes content to a new file in the filesystem.\n\n"
    "Usage:\n"
    "- This tool creates NEW files only. If the file already exists, "
    "it will return an error — to change an existing file (even a full rewrite), MUST use edit_file\n"
    "- Parent directories are created automatically if they don't exist\n"
    "- ALWAYS prefer editing existing files (with edit_file) over creating new ones "
    "when possible, as this prevents file bloat and builds on existing work"
)


class WriteFileTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="write_file",
            description=WRITE_FILE_DESCRIPTION,
            approval_resource_key="file_path",
        )
        self._agent: BaseAgent | None = None

    def bind_agent(self, agent: BaseAgent) -> None:
        self._agent = agent

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file (absolute or relative to workspace root).",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full content to write to the file.",
                    },
                },
                "required": ["file_path", "content"],
            },
        )

    async def execute(self, **kwargs: Any) -> str:
        file_path: str = kwargs.get("file_path", "")
        content: str = kwargs.get("content", "")

        if not file_path.strip():
            raise ToolValidationError("file_path cannot be empty")

        try:
            resolved = normalize_path(file_path)
        except ValueError as exc:
            return f"Error: {exc}"

        if resolved.is_dir():
            return f"Error: {file_path} is a directory."

        if resolved.exists():
            return f"Error: File already exists: {file_path}. Use edit_file to modify existing files."

        rel = relative_to_workspace(resolved)
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
        except PermissionError:
            return f"Error: Permission denied: {file_path}"
        except OSError as exc:
            return f"Error: {exc}"

        if self._agent is not None:
            mark_read(self._agent, resolved)

        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return f"Created {rel} ({line_count} lines)"


write_file = WriteFileTool()
