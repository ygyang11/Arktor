"""Glob pattern file discovery, sorted by modification time."""

from __future__ import annotations

from pathlib import Path

from agent_app.tools.filesystem._security import (
    _SKIP_DIRS,
    check_traversal,
    normalize_path,
    relative_to_workspace,
)
from agent_harness.core.errors import ToolValidationError
from agent_harness.tool.decorator import tool

GLOB_FILES_DESCRIPTION = (
    "Find files matching a glob pattern, sorted by modification time (newest first).\n\n"
    "Supports standard glob patterns: * (any characters), ** (any directories), "
    "? (single character), [abc] (character set).\n"
    "Patterns are matched under `path` and returns a list of file paths\n\n"
    "Examples:\n"
    "- **/*.py — find all Python files\n"
    "- src/**/*.ts — find TypeScript files under src/\n"
    "- *.md — find markdown files in root\n"
    "- tests/**/test_*.py — find test files under tests/\n\n"
    "If results are truncated, the output shows the next offset. "
    "Use offset to retrieve more if needed: glob_files(pattern='**/*.py', offset=200).\n\n"
    "Common non-source directories (.git, node_modules, __pycache__, etc.) are skipped; "
    "to search inside one, pass it explicitly as path."
)


@tool(description=GLOB_FILES_DESCRIPTION, approval_resource_key="path")
async def glob_files(
    pattern: str, path: str = ".", max_results: int = 200, offset: int = 0,
) -> str:
    """Find files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g. "**/*.py", "src/**/*.ts", "*.md").
        path: Directory for the search (absolute or relative to workspace root, default ".").
        max_results: Maximum number of paths to return (default 200).
        offset: Skip the first N matches for pagination (default 0).
    """
    if max_results <= 0:
        raise ToolValidationError("max_results must be a positive integer.")
    if offset < 0:
        raise ToolValidationError("offset must be non-negative.")
    if ".." in pattern.split("/"):
        raise ToolValidationError("glob pattern must not contain '..' path segments")
    if any("**" in seg and seg != "**" for seg in pattern.split("/")):
        raise ToolValidationError("invalid glob pattern: '**' must be a whole path segment")

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

    excluded: set[str] = set()
    files: list[Path] = []
    for m in raw_matches:
        if not m.is_file() or not check_traversal(m, workspace=base):
            continue
        hit = _SKIP_DIRS & set(m.relative_to(base).parts)
        if hit:
            excluded |= hit
            continue
        files.append(m)

    if not files:
        msg = f"No files matching '{pattern}' in {relative_to_workspace(base)}"
        if excluded:
            msg += f" (excluded: {', '.join(sorted(excluded))})"
        return msg

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    total = len(files)
    page = files[offset : offset + max_results]
    rel_paths = [relative_to_workspace(f) for f in page]

    header = f"{total} files matching '{pattern}'"
    if offset > 0:
        header += f" (offset {offset})"
    if offset + max_results < total:
        header += f" (showing {len(page)} of {total}, use offset={offset + max_results} for more)"
    if excluded:
        header += f" (excluded: {', '.join(sorted(excluded))})"

    return header + "\n" + "\n".join(rel_paths)
