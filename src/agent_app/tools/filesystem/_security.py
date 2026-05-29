"""Path security, encoding detection, and safe recursive traversal.

Shared infrastructure for all filesystem tools.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MAX_TEXT_FILE_SIZE: int = 50_000_000  # 50MB — used by detect_text_file (edit_file)

_SKIP_DIRS: frozenset[str] = frozenset(
    {
        # VCS internals
        ".git",
        ".svn",
        ".hg",
        # Python caches (zero source-code value)
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        # Virtual environments
        ".venv",
        "venv",
        # Test runners
        ".tox",
        ".nox",
        # JS dependencies (massive, not project source)
        "node_modules",
    }
)


def get_workspace_root() -> Path:
    """Return the resolved cwd as workspace root."""
    return Path.cwd().resolve()


def _is_within(path: Path, parent: Path) -> bool:
    """Check if path is within parent directory."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def relative_to_workspace(path: Path, workspace: Path | None = None) -> str:
    """Convert absolute path to workspace-relative string for tool output (saves tokens)."""
    ws = workspace or get_workspace_root()
    try:
        return str(path.relative_to(ws))
    except ValueError:
        return str(path)


def normalize_path(
    path_str: str,
    *,
    workspace: Path | None = None,
    must_exist: bool = False,
) -> Path:
    """Resolve a user-supplied path string to an absolute path.

    Expands ~ and resolves relative paths against the workspace. Authorization
    is delegated to ApprovalPolicy; no boundary rejection is performed here.

    Raises:
        ValueError: must_exist is set and the resolved path does not exist.
    """
    base = workspace or get_workspace_root()
    expanded = os.path.expanduser(path_str)
    raw = Path(expanded)
    resolved = (raw if raw.is_absolute() else base / raw).resolve()

    if must_exist and not resolved.exists():
        raise ValueError(f"Path does not exist: {path_str}")

    return resolved


def check_traversal(candidate: Path, workspace: Path | None = None) -> bool:
    """Check if a discovered path is safe to access during traversal.

    For symlinks: resolves realpath and checks it's within workspace.
    Returns False (skip silently) if the path is unsafe or broken.
    """
    ws = workspace or get_workspace_root()

    if candidate.is_symlink():
        try:
            real = Path(os.path.realpath(candidate))
        except OSError:
            return False
        if not _is_within(real, ws):
            return False

    try:
        resolved = candidate.resolve()
    except OSError:
        return False

    return _is_within(resolved, ws)


@dataclass(frozen=True)
class TextFileInfo:
    """Detected properties of a text file."""

    encoding: str
    newline: str
    has_bom: bool
    content: str


def detect_text_file(path: Path, max_size: int = MAX_TEXT_FILE_SIZE) -> TextFileInfo:
    """Read a text file and detect its encoding, BOM, and dominant line-ending style.

    Only supports UTF-8 family encodings. Non-UTF-8 is rejected.
    Binary files (containing NULL bytes) are rejected.

    Raises:
        ValueError: File too large, binary, or not UTF-8.
    """
    size = path.stat().st_size
    if size > max_size:
        raise ValueError(
            f"File too large ({size:,} bytes, limit {max_size:,}). "
            f"Use offset/limit for partial reads."
        )

    raw = path.read_bytes()

    # BOM/encoding detection MUST come before binary detection because
    # UTF-16 content contains NULL bytes that would trigger the binary check
    has_bom = False
    if raw.startswith(b"\xef\xbb\xbf"):
        encoding = "utf-8-sig"
        has_bom = True
    elif raw.startswith(b"\xff\xfe"):
        raise ValueError(
            f"UTF-16-LE file detected: {path.name}. Only UTF-8 files are supported for editing."
        )
    elif raw.startswith(b"\xfe\xff"):
        raise ValueError(
            f"UTF-16-BE file detected: {path.name}. Only UTF-8 files are supported for editing."
        )
    else:
        encoding = "utf-8"

    if b"\x00" in raw[:8192]:
        raise ValueError(f"Binary file detected: {path.name} ({size:,} bytes)")

    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"File is not valid UTF-8: {path.name}. Decoding failed at byte {exc.start}."
        ) from exc

    crlf_count = text.count("\r\n")
    lf_count = text.count("\n") - crlf_count
    newline = "\r\n" if crlf_count > lf_count else "\n"

    return TextFileInfo(
        encoding=encoding,
        newline=newline,
        has_bom=has_bom,
        content=text,
    )


def walk_files(
    base: Path,
    include: str,
    workspace: Path,
) -> list[Path]:
    """Recursively collect files with directory pruning and traversal safety.

    Uses os.walk(followlinks=False) to:
    1. Prune _SKIP_DIRS at the directory level (no wasted I/O)
    2. NOT follow any symlink directories (deliberate design choice)
    3. Check traversal safety on each discovered file

    include matching uses PurePath.match() (NOT fnmatch) for correct glob semantics:
    - "*.py"          — .py files at any depth
    - "src/*.py"      — .py files directly under src/ only
    - "src/**/*.py"   — .py files under src/ at any depth
    - ""              — no filter, return all files
    """
    files: list[Path] = []

    for dirpath_str, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]

        for fname in filenames:
            fpath = Path(dirpath_str) / fname

            if not check_traversal(fpath, workspace=workspace):
                continue

            if include:
                rel = fpath.relative_to(base)
                if not PurePosixPath(rel).match(include):
                    continue

            files.append(fpath)

    return files
