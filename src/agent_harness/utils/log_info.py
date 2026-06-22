"""Reading and summarizing captured-output / log files."""
from __future__ import annotations

from pathlib import Path


def tail_lines(path: Path, n: int, window: int = 65536) -> tuple[str, bool]:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            seeked = size > window
            if seeked:
                f.seek(size - window)
            data = f.read()
    except OSError:
        return "", False
    lines = data.decode("utf-8", errors="replace").splitlines()
    if seeked and lines:
        lines = lines[1:]
    return "\n".join(lines[-n:]), seeked or len(lines) > n


def summarize_log(
    path: Path, exit_code: int | None = None, n: int = 20, window: int = 65536
) -> str:
    tail, truncated = tail_lines(path, n, window)
    parts: list[str] = []
    if exit_code is not None:
        parts.append(f"Exit code: {exit_code}")
    if not tail:
        parts.append("Output: (none)")
    elif truncated:
        shown = tail.count("\n") + 1
        parts.append(f"Output (last {shown} lines):\n{tail}")
    else:
        parts.append(f"Output:\n{tail}")
    return "\n".join(parts)
