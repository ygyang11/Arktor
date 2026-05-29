"""Non-recursive directory listing."""

from __future__ import annotations

from pathlib import Path

from agent_app.tools.filesystem._security import (
    normalize_path,
    relative_to_workspace,
)
from agent_harness.tool.decorator import tool
from agent_harness.utils.media import human_size as _format_size

_MAX_ENTRIES = 200

LIST_DIR_DESCRIPTION = (
    "Lists all files and directories in a directory (non-recursive).\n\n"
    "This is useful for exploring the filesystem and understanding project structure. "
    "You should almost ALWAYS use this tool before using read_file or edit_file "
    "to find the right file to work with.\n\n"
    "Returns directories first (with trailing /), then files with sizes. "
    "Symlinks are shown with their targets."
)


def _list_dir_impl(resolved: Path) -> str:
    """Core listing logic."""
    try:
        entries = sorted(resolved.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        return f"Error: Permission denied: {resolved}"

    if not entries:
        return f"{relative_to_workspace(resolved)}/  (empty directory)"

    lines: list[str] = []
    truncated = len(entries) > _MAX_ENTRIES

    for entry in entries[:_MAX_ENTRIES]:
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

    header = f"{relative_to_workspace(resolved)}/  ({len(entries)} entries)"
    if truncated:
        header += f" — showing first {_MAX_ENTRIES}"

    return header + "\n" + "\n".join(lines)


@tool(description=LIST_DIR_DESCRIPTION, approval_resource_key="path")
async def list_dir(path: str = ".") -> str:
    """List directory contents.

    Args:
        path: Directory path (absolute or relative to workspace root, default ".").
    """
    try:
        resolved = normalize_path(path, must_exist=True)
    except ValueError as exc:
        return f"Error: {exc}"

    if not resolved.is_dir():
        return f"Error: {path} is not a directory. Use read_file for files."

    try:
        return _list_dir_impl(resolved)
    except OSError as exc:
        return f"Error: {exc}"
