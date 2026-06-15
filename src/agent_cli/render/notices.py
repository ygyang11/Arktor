"""Inline notice formatters for adapter.print_inline (non-tool)."""

from __future__ import annotations

import json
import re

from rich.text import Text

from agent_cli.theme import TOOL_DONE
from agent_harness.core.message import ToolCall, ToolResult

_REMINDER_PAIR_RE = re.compile(
    r"\A<system-reminder>\nCalled the \S+ tool[^\n]*\n</system-reminder>(?=\n\n)"
    r"\n\n"
    r"<system-reminder>\nResult of calling the \S+ tool:\n.*?\n</system-reminder>(?=\n\n|\Z)",
    re.DOTALL,
)

_DRIFT_REMINDER_RE = re.compile(
    r"\n*<system-reminder>\nNote: the following files changed on disk\b"
    r".*?</system-reminder>\s*\Z",
    re.DOTALL,
)


def _emit_reminder(body: str) -> str:
    return f"<system-reminder>\n{body}\n</system-reminder>"


def format_warning(message: str) -> Text:
    t = Text()
    t.append(f"{TOOL_DONE}  ", style="error")
    t.append(message, style="muted")
    return t


def format_expired_notice(ids: list[int]) -> Text:
    ids_str = ", ".join(f"#{i}" for i in ids)
    return format_warning(f"Pasted text {ids_str} unavailable")


def format_attachment_reminders(tc: ToolCall, tr: ToolResult) -> str:
    """Reverse-pair with :func:`peel_attachment_reminders`.."""
    args_json = json.dumps(tc.arguments, ensure_ascii=False, separators=(",", ":"))
    return (
        _emit_reminder(f"Called the {tc.name} tool with the following input: {args_json}")
        + "\n\n"
        + _emit_reminder(f"Result of calling the {tc.name} tool:\n{tr.content}")
    )


def peel_attachment_reminders(content: str) -> str:
    """Reverse of :func:`format_attachment_reminders`. Greedily strips leading
    attachment-shaped pairs; skill envelopes / drift notices never match."""
    s = content
    while (m := _REMINDER_PAIR_RE.match(s)) is not None:
        s = s[m.end():].lstrip("\n")
    return s


def peel_drift_reminder(content: str) -> str:
    """Strip a trailing file-drift reminder merged into a user message."""
    return _DRIFT_REMINDER_RE.sub("", content).rstrip("\n")


def peel_reminders(content: str) -> str:
    """Strip all harness-injected reminders from a persisted user message."""
    return peel_drift_reminder(peel_attachment_reminders(content))
