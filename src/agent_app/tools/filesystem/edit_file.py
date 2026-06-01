"""Exact string replacement preserving file format."""

from __future__ import annotations

import difflib
import logging
from typing import Any

from agent_app.observability.file_freshness import (
    Verdict,
    mark_read,
    stale_guard,
)
from agent_app.tools.filesystem._security import (
    detect_text_file,
    normalize_path,
    relative_to_workspace,
)
from agent_harness.agent.base import BaseAgent
from agent_harness.core.errors import ToolValidationError
from agent_harness.core.message import ToolOutput
from agent_harness.tool.base import BaseTool, ToolSchema

logger = logging.getLogger(__name__)
_MAX_DIFF_LINES = 100

EDIT_FILE_DESCRIPTION = (
    "Performs exact string replacements in files.\n\n"
    "Usage:\n"
    "- You must read the file before editing — understand existing content before making changes\n"
    "- When editing text from read_file output, preserve the exact indentation "
    "(tabs/spaces) as it appears in the file. Never include line number prefixes "
    "in old_string or new_string\n"
    "- The old_string must match exactly. If it appears more than once, "
    "include more surrounding context to make it unique, or set replace_all=True\n"
    "- MUST use it for ALL changes to an existing file, even a COMPLETE REWRITE\n"
    "- ALWAYS prefer editing existing files over creating new ones\n"
    "- The file's original line-ending style (LF or CRLF) and BOM are preserved automatically"
)


def _generate_diff(old: str, new: str, filename: str) -> str:
    """Generate unified diff between old and new content."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm="",
        )
    )
    if not diff:
        return "(no changes)"
    if len(diff) > _MAX_DIFF_LINES:
        return "\n".join(diff[:_MAX_DIFF_LINES]) + f"\n... ({len(diff) - _MAX_DIFF_LINES} more)"
    return "\n".join(diff)


class EditFileTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="edit_file",
            description=EDIT_FILE_DESCRIPTION,
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
                    "old_string": {
                        "type": "string",
                        "description": "The exact string to find and replace. Must be non-empty.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement string. Must differ from old_string.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "If True, replace all occurrences. Default False.",
                        "default": False,
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        )

    async def execute(self, **kwargs: Any) -> str | ToolOutput:
        file_path: str = kwargs.get("file_path", "")
        old_string: str = kwargs.get("old_string", "")
        new_string: str = kwargs.get("new_string", "")
        replace_all: bool = bool(kwargs.get("replace_all", False))

        if not file_path.strip():
            raise ToolValidationError("file_path cannot be empty")
        if not old_string:
            raise ToolValidationError(
                "old_string cannot be empty. Use write_file to create new files."
            )

        if old_string == new_string:
            raise ToolValidationError(
                "old_string and new_string are identical. No edit needed."
            )

        try:
            resolved = normalize_path(file_path, must_exist=False)
        except ValueError as exc:
            return f"Error: {exc}"

        agent = self._agent
        if agent is not None and stale_guard(agent, resolved) is Verdict.STALE:
            if not resolved.exists():
                return f"Error: {file_path} was deleted since you last accessed it."
            return (
                f"Error: {file_path} has changed since you last accessed it. "
                f"Re-read it with read_file before editing."
            )

        if not resolved.exists():
            return f"Error: Path does not exist: {file_path}"

        if resolved.is_dir():
            return f"Error: {file_path} is a directory."

        rel = relative_to_workspace(resolved)

        try:
            file_info = detect_text_file(resolved)
        except ValueError as exc:
            return f"Error: {exc}"

        content = file_info.content
        original_newline = file_info.newline

        work_content = content.replace("\r\n", "\n")
        work_old = old_string.replace("\r\n", "\n")
        work_new = new_string.replace("\r\n", "\n")

        count = work_content.count(work_old)

        if count == 0:
            return (
                f"Error: old_string not found in {rel}. "
                "Make sure the string matches exactly (including whitespace and indentation)."
            )

        if count > 1 and not replace_all:
            return (
                f"Error: old_string appears {count} times in {rel}. "
                "Provide more surrounding context to make it unique, "
                "or set replace_all=True to replace all occurrences."
            )

        if replace_all:
            new_content = work_content.replace(work_old, work_new)
        else:
            new_content = work_content.replace(work_old, work_new, 1)

        if original_newline == "\r\n":
            new_content = new_content.replace("\n", "\r\n")

        try:
            resolved.write_text(new_content, encoding=file_info.encoding)
        except OSError as exc:
            return f"Error: {exc}"

        if agent is not None:
            mark_read(agent, resolved)

        diff = _generate_diff(work_content, new_content.replace("\r\n", "\n"), rel)
        replaced = count if replace_all else 1
        header = f"Edited {rel} ({replaced} replacement{'s' if replaced > 1 else ''})"
        return ToolOutput(content=header, tool_metadata={"diff": diff})


edit_file = EditFileTool()
