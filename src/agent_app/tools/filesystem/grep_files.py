"""Regex content search across files."""

from __future__ import annotations

import re
from pathlib import Path

from agent_app.tools.filesystem._security import (
    normalize_path,
    relative_to_workspace,
    walk_files,
)
from agent_harness.core.errors import ToolValidationError
from agent_harness.tool.decorator import tool

_MAX_GREP_FILE_SIZE = 250_000_000  # 250MB — grep streams line-by-line
_MAX_LINE_CHARS = 5_000

GREP_FILES_DESCRIPTION = (
    "Search for a text pattern across files using regular expressions (Python re syntax).\n\n"
    "Searches file contents and returns matching lines with file paths and line numbers. "
    "Supports regex syntax — special characters like parentheses, brackets, pipes "
    "are treated as regex operators, not literals.\n\n"
    "The search runs recursively under `path` (default: '.', the workspace root). `include` "
    "selects which of those files to search: a filename glob matched at any depth, so "
    "'*.py' matches Python files anywhere under `path` (nested included). A '/' anchors "
    "the pattern relative to `path` — 'src/*.ts' = files directly in src/, "
    "'src/**' = the whole src/ subtree.\n\n"
    "Usage:\n"
    "- Search all files: grep_files(pattern='TODO')\n"
    "- Search Python files only: grep_files(pattern='import', include='*.py')\n"
    "- Search within a directory: grep_files(pattern='def', path='src')\n"
    "- Search a directory by file type: grep_files(pattern='import', path='src', include='*.ts')\n"
    "- Show context around matches: grep_files(pattern='error', context=2)\n"
    "- Case-insensitive: grep_files(pattern='todo', case_insensitive=True)\n\n"
    "If results are truncated, the output shows the next offset. "
    "Use offset to retrieve more if needed: grep_files(pattern='TODO', offset=250).\n\n"
    "Automatically skips binary files and common non-source directories "
    "(.git, node_modules, __pycache__, etc.); to search inside one, pass it explicitly as path."
)


def _search_file_streaming(
    file_path: Path,
    regex: re.Pattern[str],
    context: int,
) -> tuple[int, list[str]]:
    """Search a single file using line-by-line streaming.

    Returns (match_count, formatted_output_lines).
    """
    try:
        size = file_path.stat().st_size
        if size > _MAX_GREP_FILE_SIZE:
            rel = relative_to_workspace(file_path)
            limit_mb = _MAX_GREP_FILE_SIZE // 1_000_000
            return 0, [f"(skipped {rel}: {size:,} bytes, exceeds {limit_mb}MB grep limit)"]
    except OSError:
        return 0, []

    # Binary detection
    try:
        with open(file_path, "rb") as fb:
            head = fb.read(8192)
        if b"\x00" in head:
            return 0, []
    except OSError:
        return 0, []

    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            lines = [line.rstrip("\n").rstrip("\r") for line in f]
    except OSError:
        return 0, []

    matched_line_nums: set[int] = set()
    context_ranges: set[int] = set()

    for i, line in enumerate(lines):
        if regex.search(line):
            matched_line_nums.add(i)
            for j in range(max(0, i - context), min(len(lines), i + context + 1)):
                context_ranges.add(j)

    if not matched_line_nums:
        return 0, []

    rel = relative_to_workspace(file_path)
    result_lines: list[str] = []
    prev_idx = -2

    for idx in sorted(context_ranges):
        if idx > prev_idx + 1 and prev_idx >= 0:
            result_lines.append("--")
        line_text = lines[idx]
        if len(line_text) > _MAX_LINE_CHARS:
            line_text = line_text[:_MAX_LINE_CHARS] + "... (truncated)"
        marker = ":" if idx in matched_line_nums else "-"
        result_lines.append(f"{rel}{marker}{idx + 1}{marker}{line_text}")
        prev_idx = idx

    return len(matched_line_nums), result_lines


@tool(description=GREP_FILES_DESCRIPTION, approval_resource_key="path")
async def grep_files(
    pattern: str,
    path: str = ".",
    include: str = "",
    context: int = 0,
    case_insensitive: bool = False,
    max_results: int = 250,
    offset: int = 0,
) -> str:
    """Search file contents using a regular expression pattern.

    Args:
        pattern: Regex pattern (Python re syntax).
        path: Directory to search (absolute or relative to workspace root, default ".").
        include: Filename glob to filter which files are searched (default: "", no filter).
        context: Context lines before/after each match (default 0).
        case_insensitive: Case-insensitive search (default False).
        max_results: Maximum output lines to return (default 250).
        offset: Skip first N output lines for pagination (default 0).
    """
    if not pattern:
        raise ToolValidationError("pattern cannot be empty.")

    if max_results <= 0:
        raise ToolValidationError("max_results must be a positive integer.")
    if context < 0:
        raise ToolValidationError("context must be non-negative.")
    if offset < 0:
        raise ToolValidationError("offset must be non-negative.")

    flags = re.IGNORECASE if case_insensitive else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        raise ToolValidationError(f"invalid regex pattern: {exc}") from exc

    try:
        base = normalize_path(path, must_exist=True)
    except ValueError as exc:
        return f"Error: {exc}"

    if not base.is_dir():
        return f"Error: {path} is not a directory."

    excluded: set[str] = set()
    files = walk_files(base, include, workspace=base, skipped=excluded)
    files.sort()

    total_match_count = 0
    files_matched = 0
    all_output: list[str] = []

    for f in files:
        match_count, output_lines = _search_file_streaming(f, regex, context)
        if match_count > 0:
            total_match_count += match_count
            files_matched += 1
        # Always collect output (includes skip notices for oversized files)
        all_output.extend(output_lines)

    if total_match_count == 0:
        scope = f" in '{include}' files" if include else ""
        msg = f"No matches for '{pattern}'{scope} under {relative_to_workspace(base)}"
        if excluded:
            msg += f" (excluded: {', '.join(sorted(excluded))})"
        if all_output:
            msg += "\n" + "\n".join(all_output)
        return msg

    total_output_lines = len(all_output)
    paginated = all_output[offset : offset + max_results]

    header = f"{total_match_count} matches in {files_matched} files"
    if offset > 0:
        header += f" (offset {offset})"
    if offset + max_results < total_output_lines:
        next_offset = offset + max_results
        header += (
            f" (showing {len(paginated)} of {total_output_lines} output lines, "
            f"use offset={next_offset} for more)"
        )
    if excluded:
        header += f" (excluded: {', '.join(sorted(excluded))})"

    return header + "\n" + "\n".join(paginated)
