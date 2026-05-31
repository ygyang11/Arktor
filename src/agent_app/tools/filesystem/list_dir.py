"""Non-recursive directory listing."""

from __future__ import annotations

from pathlib import Path

from agent_app.tools.filesystem._security import (
    normalize_path,
    relative_to_workspace,
)
from agent_harness.core.errors import ToolValidationError
from agent_harness.tool.decorator import tool
from agent_harness.utils.media import human_size as _format_size

LIST_DIR_DESCRIPTION = (
    "Lists all files and directories in a directory (non-recursive).\n\n"
    "This is useful for exploring the filesystem and understanding project structure. "
    "You should almost ALWAYS use this tool before using read_file or edit_file "
    "to find the right file to work with.\n\n"
    "Returns directories first (with trailing /), then files with sizes. "
    "Symlinks show their target when it points inside the listed directory, otherwise they "
    "are marked (external), or (unresolved symlink) if broken.\n\n"
    "If the listing is truncated, the output shows the next offset. "
    "Use offset to retrieve more if needed: list_dir(path='.', offset=200)."
)


def _list_dir_impl(resolved: Path, max_results: int, offset: int) -> str:
    """Core listing logic."""
    try:
        entries = sorted(resolved.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        return f"Error: Permission denied: {resolved}"

    if not entries:
        return f"{relative_to_workspace(resolved)}/  (empty directory)"

    total = len(entries)
    page = entries[offset : offset + max_results]

    lines: list[str] = []

    for entry in page:
        if entry.is_symlink():
            try:
                target = entry.resolve(strict=True)
            except (OSError, RuntimeError):
                lines.append(f"  {entry.name} -> (unresolved symlink)")
                continue
            try:
                target.relative_to(resolved)
                lines.append(f"  {entry.name} -> {relative_to_workspace(target)}")
            except ValueError:
                lines.append(f"  {entry.name} -> (external)")
        elif entry.is_dir():
            lines.append(f"  {entry.name}/")
        else:
            try:
                size = _format_size(entry.stat().st_size)
            except OSError:
                size = "?"
            lines.append(f"  {entry.name}  ({size})")

    header = f"{relative_to_workspace(resolved)}/  ({total} entries)"
    if offset > 0:
        header += f" (offset {offset})"
    if offset + max_results < total:
        header += f" — showing {len(page)} of {total}, use offset={offset + max_results} for more"

    return header + "\n" + "\n".join(lines)


@tool(description=LIST_DIR_DESCRIPTION, approval_resource_key="path")
async def list_dir(path: str = ".", max_results: int = 200, offset: int = 0) -> str:
    """List directory contents.

    Args:
        path: Directory path (absolute or relative to workspace root, default ".").
        max_results: Maximum number of entries to return (default 200).
        offset: Skip the first N entries for pagination (default 0).
    """
    if max_results <= 0:
        raise ToolValidationError("max_results must be a positive integer.")
    if offset < 0:
        raise ToolValidationError("offset must be non-negative.")

    try:
        resolved = normalize_path(path, must_exist=True)
    except ValueError as exc:
        return f"Error: {exc}"

    if not resolved.is_dir():
        return f"Error: {path} is not a directory. Use read_file for files."

    try:
        return _list_dir_impl(resolved, max_results, offset)
    except OSError as exc:
        return f"Error: {exc}"
