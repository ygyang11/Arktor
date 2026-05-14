"""CLI-only tail patch surfacing external filesystem drift to the model."""
from __future__ import annotations

import functools
from pathlib import Path

from agent_app.observability.file_freshness import Drift, poll_dirty
from agent_app.tools.filesystem._security import relative_to_workspace
from agent_harness.agent.base import BaseAgent
from agent_harness.core.message import Message
from agent_harness.prompt.patch import ContextPatch


def enable(agent: BaseAgent) -> None:
    patch = _patch_for(agent)
    if patch not in agent.context.context_patches:
        agent.context.context_patches.append(patch)


@functools.cache
def _patch_for(agent: BaseAgent) -> ContextPatch:
    def _build() -> Message | None:
        drifts = poll_dirty(agent)
        if not drifts:
            return None
        return Message.user(_format_notice(drifts))
    return ContextPatch(at="tail", build=_build)


def _format_notice(drifts: list[Drift]) -> str:
    lines = [
        "<system-reminder>",
        "Note: the following files were modified, either by the user or by "
        "a linter. These changes were intentional, so take them into account "
        "as you proceed (ie. don't revert them unless the user asks you to). "
        "If you plan to use or edit any of them again, re-read it with "
        "read_file first to refresh your view; otherwise this is "
        "informational. Don't tell the user this, since they are already "
        "aware.",
        "",
    ]
    for d in drifts:
        display = relative_to_workspace(Path(d.path))
        marker = "deleted" if d.current is None else "modified"
        lines.append(f"- {display} ({marker})")
    lines.append("</system-reminder>")
    return "\n".join(lines)
