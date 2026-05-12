"""Shared git subprocess wrapper for CLI commands."""
from __future__ import annotations

import asyncio
import subprocess


def _run_sync(args: tuple[str, ...], timeout: float) -> tuple[int, str, str]:
    try:
        r = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode, r.stdout, r.stderr
    except (OSError, subprocess.SubprocessError) as e:
        return -1, "", str(e)


async def run(*args: str, timeout: float = 10.0) -> tuple[int, str, str]:
    return await asyncio.to_thread(_run_sync, args, timeout)
