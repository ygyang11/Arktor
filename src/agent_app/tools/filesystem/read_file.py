"""Paginated file reading with line numbers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent_app.observability.file_freshness import record_signature
from agent_app.tools.filesystem._security import (
    is_sensitive_path,
    normalize_path,
    relative_to_workspace,
)
from agent_harness.agent.base import BaseAgent
from agent_harness.tool.base import BaseTool, ToolSchema

logger = logging.getLogger(__name__)
_MAX_LINE_CHARS = 5_000

READ_FILE_DESCRIPTION = (
    "Reads a file from the filesystem with line numbers. "
    "Supports paginated reading for large files via offset and limit parameters.\n\n"
    "Usage:\n"
    "- Assume this tool is able to read all files. If the user provides a path, "
    "assume that path is valid. It is okay to read a file that does not exist; "
    "an error will be returned\n"
    "- By default, reads up to 200 lines from the beginning of the file\n"
    "- For large files, use pagination with offset and limit: "
    "read_file(path, offset=200, limit=200) to read the next section\n"
    "- Read only what you need — avoid reading entire large files at once\n"
    "- Results are returned in cat -n format (line_number + tab + content) "
    "with a header showing total lines and position\n"
    "- You should ALWAYS read a file before editing it\n"
    "- It is better to speculatively read multiple files as a batch when exploring a codebase\n"
    "- If you read a file that exists but has empty contents, "
    "a system message will indicate the file is empty"
)


def _count_lines(path: Path) -> int:
    """Count total lines by streaming (no full memory load)."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def _read_file_streaming(path: Path, offset: int, limit: int) -> str:
    """Core read logic using line-by-line streaming."""
    size = path.stat().st_size

    # Binary detection: read first 8KB only
    with open(path, "rb") as f:
        head = f.read(8192)
    if b"\x00" in head:
        return f"Binary file detected: {path.name} ({size:,} bytes)."

    encoding = "utf-8-sig" if head.startswith(b"\xef\xbb\xbf") else "utf-8"

    total_lines = _count_lines(path)

    if total_lines == 0:
        return "(empty file, 0 lines)"

    start = min(offset, total_lines)
    end = min(start + limit, total_lines)

    output_lines: list[str] = []
    try:
        with open(path, encoding=encoding, errors="replace") as f:
            for i, line in enumerate(f):
                if i < start:
                    continue
                if i >= end:
                    break
                line = line.rstrip("\n").rstrip("\r")
                if len(line) > _MAX_LINE_CHARS:
                    line = line[:_MAX_LINE_CHARS] + "... (truncated)"
                output_lines.append(f"{i + 1}\t{line}")
    except UnicodeDecodeError:
        with open(path, encoding="latin-1") as f:
            for i, line in enumerate(f):
                if i < start:
                    continue
                if i >= end:
                    break
                line = line.rstrip("\n").rstrip("\r")
                if len(line) > _MAX_LINE_CHARS:
                    line = line[:_MAX_LINE_CHARS] + "... (truncated)"
                output_lines.append(f"{i + 1}\t{line}")

    result = "\n".join(output_lines)

    rel = relative_to_workspace(path)
    header = f"[{rel}] lines {start + 1}-{end} of {total_lines}"
    parts: list[str] = []
    if start > 0:
        parts.append(f"{start} lines before")
    if end < total_lines:
        parts.append(f"{total_lines - end} lines after, use offset={end} to continue")
    if parts:
        header += f" ({'; '.join(parts)})"

    return f"{header}\n{result}"


class ReadFileTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="read_file",
            description=READ_FILE_DESCRIPTION,
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
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (0-based, default 0).",
                        "default": 0,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read (default 2000).",
                        "default": 2000,
                    },
                },
                "required": ["file_path"],
            },
        )

    async def execute(self, **kwargs: Any) -> str:
        file_path: str = kwargs.get("file_path", "")
        offset: int = int(kwargs.get("offset", 0))
        limit: int = int(kwargs.get("limit", 2000))

        if limit <= 0:
            return "Error: limit must be a positive integer."
        if offset < 0:
            return "Error: offset must be non-negative."

        try:
            resolved = normalize_path(file_path, must_exist=True)
        except ValueError as exc:
            return f"Error: {exc}"

        if resolved.is_dir():
            return f"Error: {file_path} is a directory. Use list_dir instead."

        if is_sensitive_path(resolved):
            logger.debug("Reading sensitive file: %s", resolved)

        try:
            content = _read_file_streaming(resolved, offset, limit)
        except PermissionError:
            return f"Error: Permission denied: {file_path}"
        except OSError as exc:
            return f"Error: {exc}"

        if self._agent is not None:
            record_signature(self._agent, resolved)
        return content


read_file = ReadFileTool()
