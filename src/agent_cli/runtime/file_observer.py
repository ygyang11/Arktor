"""CLI-only: surface external filesystem drift by merging a reminder into the user turn."""
from __future__ import annotations

from pathlib import Path

from agent_app.observability.file_freshness import Drift, mark_seen, poll_drift
from agent_app.tools.filesystem._security import relative_to_workspace
from agent_harness.agent.base import BaseAgent
from agent_harness.core.message import Message


def annotate_drift(agent: BaseAgent, msg: Message) -> None:
    """Append a file-drift reminder to the user turn, like attachment reminders.

    Surfaces files changed since last read, marking them seen so the same
    drift is not repeated next turn.
    """
    drifts = poll_drift(agent)
    if not drifts:
        return
    notice = _format_notice(drifts)
    msg.content = f"{msg.content}\n\n{notice}" if msg.content else notice
    for d in drifts:
        mark_seen(agent, d.path)


def _format_notice(drifts: list[Drift]) -> str:
    lines = [
        "<system-reminder>",
        "Note: the following files changed on disk since you last read them — "
        "this may have been the user, your own terminal_tool command, or a tool "
        "such as a linter. Take their current contents into account as you "
        "proceed (don't revert the changes unless the user asks). If you plan "
        "to use or edit any of them again, re-read it with read_file first to "
        "refresh your view; otherwise this is informational. Don't mention "
        "this reminder to the user.",
        "",
    ]
    for d in drifts:
        display = relative_to_workspace(Path(d.path))
        marker = "deleted" if d.current is None else "modified"
        lines.append(f"- {display} ({marker})")
    lines.append("</system-reminder>")
    return "\n".join(lines)
