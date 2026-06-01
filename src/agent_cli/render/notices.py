"""Inline notice formatters for adapter.print_inline (non-tool)."""

from __future__ import annotations

import json
import re

from rich.text import Text

from agent_cli.theme import TOOL_DONE
from agent_harness.core.message import ToolCall, ToolResult
from agent_harness.utils.token_counter import truncate_text_by_tokens

_SHELL_LANE_OUTPUT_TOKENS = 10_000
_SHELL_RUN_OPEN_TAG = "<user-shell-run>"
_SHELL_RUN_CLOSE_TAG = "</user-shell-run>"
_SHELL_RUN_CLOSE_TAG_ESCAPED = "</user-shell-run​>"

_SHELL_RUN_RE = re.compile(
    r"\A" + re.escape(_SHELL_RUN_OPEN_TAG) + r"\s*\n"
    r"```sh\n(?P<cmd>.*?)\n```\n"
    r"(?P<body>.*?)\n" + re.escape(_SHELL_RUN_CLOSE_TAG) + r"\s*\Z",
    re.DOTALL,
)

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


def _escape_envelope(s: str) -> str:
    return s.replace(_SHELL_RUN_CLOSE_TAG, _SHELL_RUN_CLOSE_TAG_ESCAPED)


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


def parse_shell_run_envelope(content: str) -> tuple[str, str] | None:
    """Reverse of :func:`format_shell_run`. Returns (cmd, body) or None."""
    m = _SHELL_RUN_RE.match(content)
    if m is None:
        return None
    return m.group("cmd"), m.group("body")


def format_shell_run(
    command: str,
    exit_code: int,
    output: str,
    post_notices: list[str] | None = None,
) -> str:
    """Format a `!`-lane shell run as the body of a ``Message.user``."""
    truncated = truncate_text_by_tokens(
        output,
        max_tokens=_SHELL_LANE_OUTPUT_TOKENS,
        suffix="\n... (truncated)",
    )
    has_output = bool(truncated.strip())

    if exit_code != 0 and has_output:
        body = f"[exit code {exit_code}]\n{truncated}"
    elif exit_code != 0:
        body = f"[exit code {exit_code}]\n(Completed with no output)"
    elif has_output:
        body = truncated
    else:
        body = "(Completed with no output)"

    if post_notices:
        body = body + "\n" + "\n".join(f"[Accident] {n}" for n in post_notices)

    safe_command = _escape_envelope(command)
    safe_body = _escape_envelope(body)
    return f"<user-shell-run>\n```sh\n{safe_command}\n```\n{safe_body}\n</user-shell-run>"


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