"""Paginated file reading with line numbers."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from agent_app.observability.file_freshness import mark_read
from agent_app.tools.filesystem._security import (
    normalize_path,
    relative_to_workspace,
)
from agent_harness.agent.base import BaseAgent
from agent_harness.core.errors import ToolValidationError
from agent_harness.core.message import ToolOutput
from agent_harness.tool.base import BaseTool, ToolSchema
from agent_harness.utils.blob import make_attachment
from agent_harness.utils.media import human_size, is_media_mime, is_pdf_mime

_MAX_LINE_CHARS = 5_000
_MEDIA_PREVIEW_MAX_BYTES = 30 * 1024 * 1024

READ_FILE_DESCRIPTION = (
    "Reads any file from the filesystem: text files paginated with line numbers, "
    "or PDF/image files inlined as media attachments for direct viewing. "
    "(files only — to see what a directory contains, use list_dir)"
    "Supports paginated reading for large text files via offset and limit parameters.\n\n"
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
    "- For PDF (.pdf) and image (png/jpg/jpeg/gif/webp) files, content is "
    "returned as a media attachment in the following message instead of "
    "paginated text — prefer this for viewing media directly\n"
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
        return f"Error: binary file (not readable as text or media): {path.name} ({size:,} bytes)"

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
                        "description": "Maximum number of lines to read (default 200).",
                        "default": 200,
                    },
                },
                "required": ["file_path"],
            },
        )

    async def execute(self, **kwargs: Any) -> str | ToolOutput:
        file_path_raw = kwargs.get("file_path", "")
        file_path: str = file_path_raw if isinstance(file_path_raw, str) else ""
        offset: int = int(kwargs.get("offset", 0))
        limit: int = int(kwargs.get("limit", 200))

        if not file_path.strip():
            raise ToolValidationError("file_path cannot be empty")
        if limit <= 0:
            raise ToolValidationError("limit must be a positive integer.")
        if offset < 0:
            raise ToolValidationError("offset must be non-negative.")

        try:
            resolved = normalize_path(file_path, must_exist=True)
        except ValueError as exc:
            return f"Error: {exc}"

        if resolved.is_dir():
            return f"Error: {file_path} is a directory. Use list_dir instead."

        mime, _ = mimetypes.guess_type(str(resolved))
        if mime and is_media_mime(mime):
            size = resolved.stat().st_size
            kind = "PDF" if is_pdf_mime(mime) else "image"
            if size > _MEDIA_PREVIEW_MAX_BYTES:
                return f"Error: {kind} too large to attach ({human_size(size)})"
            att = make_attachment(resolved.read_bytes(), mime, resolved.name)
            if self._agent is not None:
                mark_read(self._agent, resolved)
            return ToolOutput(
                content=(
                    f"Read {kind} from {file_path}; the {kind} is provided "
                    f"as an attachment in the following message."
                ),
                attachments=[att],
            )

        try:
            content = _read_file_streaming(resolved, offset, limit)
        except PermissionError:
            return f"Error: Permission denied: {file_path}"
        except OSError as exc:
            return f"Error: {exc}"

        if self._agent is not None:
            mark_read(self._agent, resolved)
        return content


read_file = ReadFileTool()
