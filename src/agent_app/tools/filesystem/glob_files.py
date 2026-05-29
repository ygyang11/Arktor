"""Glob pattern file discovery, sorted by modification time."""

from __future__ import annotations

from agent_app.tools.filesystem._security import (
    _SKIP_DIRS,
    check_traversal,
    normalize_path,
    relative_to_workspace,
)
from agent_harness.core.errors import ToolValidationError
from agent_harness.tool.decorator import tool

_MAX_RESULTS = 100

GLOB_FILES_DESCRIPTION = (
    "Find files matching a glob pattern, sorted by modification time (newest first).\n\n"
    "Supports standard glob patterns: * (any characters), ** (any directories), "
    "? (single character), [abc] (character set).\n"
    "Returns a list of file paths relative to the workspace.\n\n"
    "Examples:\n"
    "- **/*.py — find all Python files\n"
    "- src/**/*.ts — find TypeScript files under src/\n"
    "- *.md — find markdown files in root\n"
    "- tests/**/test_*.py — find test files under tests/"
)


@tool(description=GLOB_FILES_DESCRIPTION, approval_resource_key="path")
async def glob_files(pattern: str, path: str = ".") -> str:
    """Find files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g. "**/*.py", "src/**/*.ts", "*.md").
        path: Base directory for the search (default: workspace root).
    """
    try:
        base = normalize_path(path, must_exist=True)
    except ValueError as exc:
        return f"Error: {exc}"

    if not base.is_dir():
        return f"Error: {path} is not a directory."

    try:
        raw_matches = list(base.glob(pattern))
    except (ValueError, NotImplementedError) as exc:
        raise ToolValidationError(f"invalid glob pattern: {exc}") from exc
    except OSError as exc:
        return f"Error: {exc}"

    files = [
        m
        for m in raw_matches
        if m.is_file()
        and check_traversal(m, workspace=base)
        and not (_SKIP_DIRS & set(m.relative_to(base).parts))
    ]

    if not files:
        return f"No files matching '{pattern}' in {relative_to_workspace(base)}"

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    total = len(files)
    truncated = total > _MAX_RESULTS
    files = files[:_MAX_RESULTS]
    rel_paths = [relative_to_workspace(f) for f in files]

    header = f"{total} files matching '{pattern}'"
    if truncated:
        header += f" (showing first {_MAX_RESULTS})"

    return header + "\n" + "\n".join(rel_paths)
