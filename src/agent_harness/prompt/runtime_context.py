"""Runtime context provider — ephemeral environment info injected per LLM call."""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path

from agent_harness.core.message import Message

logger = logging.getLogger(__name__)


class RuntimeContextProvider:
    """Builds ephemeral runtime context as a system message.

    Injected into the LLM message list at call_llm() time but NOT persisted
    to short-term memory or sessions. Contains stable-per-session environment
    facts: date, platform, cwd.
    """

    def build_context_message(self) -> Message | None:
        """Build runtime context system message. Returns None if empty."""
        cwd = str(Path.cwd())
        parts: list[str] = ["# Environment"]

        parts.append(f"- Primary working directory: {cwd}")
        parts.append(f"  - Is a git repository: {self._is_git_repo(cwd)}")

        parts.append(f"- Platform: {platform.system().lower()}")
        shell = os.environ.get("SHELL", "")
        if shell:
            parts.append(f"- Shell: {shell.rsplit('/', 1)[-1]}")
        parts.append(f"- OS Version: {platform.platform()}")
        parts.append(f"- Current date: {datetime.now().strftime('%Y-%m-%d')}")

        return Message.system("\n".join(parts))

    @staticmethod
    def _is_git_repo(cwd: str) -> bool:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=3,
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False
