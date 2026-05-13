"""Static replay of a session's recent turns — no adapter, no Live.

User messages persisted to short-term memory may carry envelope/expansion
shapes that obscure what the user originally typed (shell-run wrap, skill
expansion, command prompt template). This module detects each known shape
and renders it back to the live-REPL look, then falls back to plain text.
"""
from __future__ import annotations

import re

from rich.console import Console
from rich.text import Text

from agent_cli.commands.builtin.init import _INIT_NEW, _INIT_UPDATE
from agent_cli.commands.builtin.review import _DEFAULT_REVIEW_TARGET, _REVIEW_PROMPT
from agent_cli.commands.ui import ok
from agent_cli.render.markdown_stream import render_markdown_block
from agent_cli.render.notices import parse_shell_run_envelope
from agent_cli.render.tool_display import (
    format_attachments,
    format_shell_run,
    render_completed_call,
)
from agent_cli.repl.mentions import is_attachment_turn
from agent_cli.runtime.session import get_messages
from agent_cli.theme import PROMPT, CliTheme
from agent_harness.agent.base import BaseAgent
from agent_harness.core.message import Message, Role, ToolResult


# ── recognisable user-message shapes ─────────────────────────────────

_FOCUS_PREFIX = "\n\nFocus: "
_INIT_NEW_BASE = _INIT_NEW.format(focus="")
_INIT_UPDATE_BASE = _INIT_UPDATE.format(target="AGENTS.md", focus="")
_REVIEW_HEAD = _REVIEW_PROMPT.split("{target}", 1)[0]

_SKILL_REMINDER_RE = re.compile(
    r"\A(?P<args>.*?)"
    r"(?:\n\n)?<system-reminder>The user has explicitly requested the "
    r"(?P<name>\S+) skill\..*?</system-reminder>\n\n"
    r"<skill-loaded name=\"[^\"]+\">.*?</skill-loaded>\s*\Z",
    re.DOTALL,
)

_EXIT_CODE_RE = re.compile(r"\A\[exit code (-?\d+)\]\n?(.*)", re.DOTALL)


def _match_init(content: str) -> str | None:
    """Return focus text (possibly empty) if content is an /init submission."""
    for base in (_INIT_NEW_BASE, _INIT_UPDATE_BASE):
        if content == base:
            return ""
        if content.startswith(base + _FOCUS_PREFIX):
            return content[len(base) + len(_FOCUS_PREFIX):]
    return None


def _match_review(content: str) -> str | None:
    """Return target text (empty for default) if content is a /review submission."""
    if not content.startswith(_REVIEW_HEAD):
        return None
    target = content[len(_REVIEW_HEAD):].rstrip("\n")
    return "" if target == _DEFAULT_REVIEW_TARGET else target


def _match_skill(content: str) -> tuple[str, str] | None:
    """Return (skill_name, args) if content is a /<skill> envelope."""
    m = _SKILL_REMINDER_RE.match(content)
    if m is None:
        return None
    return m.group("name"), m.group("args").strip()


def _split_exit_code(body: str) -> tuple[int, str]:
    m = _EXIT_CODE_RE.match(body)
    if m is None:
        return 0, body
    return int(m.group(1)), m.group(2)


# ── per-shape session preview ───────────────────────────────────

def peel_user_command(content: str) -> str | None:
    """Canonical "what the user typed" form, or None if content is plain text."""
    if (parsed := parse_shell_run_envelope(content)) is not None:
        cmd, _ = parsed
        return "! " + " ".join(cmd.split())
    if (focus := _match_init(content)) is not None:
        return f"/init {focus}".rstrip()
    if (target := _match_review(content)) is not None:
        return f"/review {target}".rstrip()
    if (parsed := _match_skill(content)) is not None:
        name, args = parsed
        return f"/{name} {args}".rstrip() if args else f"/{name}"
    return None


# ── per-shape user renderers ───────────────────────────────────

def _render_user_block(console: Console, body: str) -> None:
    """Print ``❯ <body>`` as a section-bg block padded to terminal width."""
    width = console.width
    for i, raw in enumerate(body.split("\n")):
        prefix = f"{PROMPT} " if i == 0 else "  "
        line = Text(prefix + raw)
        pad = max(0, width - line.cell_len)
        if pad:
            line.append(" " * pad)
        line.stylize("section")
        console.print(line)


def _render_user_shell_run(console: Console, content: str) -> bool:
    parsed = parse_shell_run_envelope(content)
    if parsed is None:
        return False
    cmd, body = parsed
    _render_user_block(console, f"!{cmd}")
    console.print()
    exit_code, output = _split_exit_code(body)
    for r in format_shell_run(cmd, exit_code, output):
        console.print(r)
    console.print()
    return True

def _render_attachment_block(
    console: Console,
    asst_msg: Message,
    results: dict[str, ToolResult],
) -> None:
    """Render an attachment turn via the shared live formatter."""
    items = [
        (tc, results[tc.id])
        for tc in asst_msg.tool_calls or []
        if tc.id in results
    ]
    if not items:
        return
    for r in format_attachments(items):
        console.print(r)
    console.print()


def _render_command_invocation(console: Console, content: str) -> bool:
    if (focus := _match_init(content)) is not None:
        return _emit_slash_command(console, "init", focus)
    if (target := _match_review(content)) is not None:
        return _emit_slash_command(console, "review", target)
    if (parsed := _match_skill(content)) is not None:
        name, args = parsed
        return _emit_slash_command(console, name, args)
    return False


def _emit_slash_command(console: Console, name: str, args: str) -> bool:
    body = f"/{name} {args}" if args else f"/{name}"
    _render_user_block(console, body)
    console.print()
    return True


def _render_plain_user(console: Console, content: str) -> None:
    _render_user_block(console, content)
    console.print()


# ── context compression ─────────────────────────

def _render_compaction(console: Console, messages: list[Message]) -> None:
    """Print ``── Earlier messages compacted [×N] ──`` (muted) at the top
    of replay if stm carries any compression summary; otherwise no-op."""
    for m in messages:
        if m.role != Role.SYSTEM:
            continue
        meta = getattr(m, "metadata", None) or {}
        if not meta.get("is_compression_summary"):
            continue
        round_n = int(meta.get("compression_round") or 1)
        label = "Earlier messages compacted"
        if round_n > 1:
            label = f"{label} ×{round_n}"
        console.print(Text(f"── {label} ──", style="muted"))
        console.print()
        return


# ── tool-result indexing for completed assistant calls ───────────────

def _index_results(messages: list[Message]) -> dict[str, ToolResult]:
    out: dict[str, ToolResult] = {}
    for m in messages:
        if m.role == Role.TOOL and m.tool_result is not None:
            out[m.tool_result.tool_call_id] = m.tool_result
    return out


# ── public API ───────────────────────────────────────────────────────

def _hard_clear(console: Console) -> None:
    # Rich Console.clear emits \x1b[2J\x1b[H but skips scrollback and is gated
    # on is_terminal; bypass both with a raw write so the user actually sees
    # a clean screen on /resume / /new.
    console.file.write("\x1b[2J\x1b[3J\x1b[H")
    console.file.flush()


def slice_last_turns(messages: list[Message], n: int) -> list[Message]:
    seen = 0
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].role == Role.USER:
            seen += 1
            if seen == n:
                return [m for m in messages[idx:] if m.role != Role.SYSTEM]
    return [m for m in messages if m.role != Role.SYSTEM]


def _render_user(console: Console, content: str) -> None:
    """Try each known user-message shape; first match wins."""
    if _render_user_shell_run(console, content):
        return
    if _render_command_invocation(console, content):
        return
    _render_plain_user(console, content)


def _render_assistant(
    console: Console,
    theme: CliTheme,
    msg: Message,
    results: dict[str, ToolResult],
) -> None:
    if msg.content:
        render_markdown_block(console, msg.content, theme)
        console.print()
    for tc in msg.tool_calls or ():
        console.print(render_completed_call(tc, results.get(tc.id)))
        console.print()


def replay(console: Console, theme: CliTheme, messages: list[Message]) -> None:
    """Print persisted turns in live-REPL style."""
    if not messages:
        return
    results = _index_results(messages)
    i = 0
    while i < len(messages):
        m = messages[i]
        step = 1
        if m.role == Role.USER and m.content:
            _render_user(console, m.content)
            nxt = messages[i + 1] if i + 1 < len(messages) else None
            if nxt is not None and is_attachment_turn(m, nxt):
                _render_attachment_block(console, nxt, results)
                step = 2
        elif m.role == Role.ASSISTANT:
            _render_assistant(console, theme, m, results)
        i += step


def render_post_switch(
    agent: BaseAgent,
    console: Console,
    theme: CliTheme,
    new_id: str,
) -> None:
    _hard_clear(console)
    msgs = get_messages(agent)
    if msgs:
        _render_compaction(console, msgs)
        replay(console, theme, slice_last_turns(msgs, 5))
        console.print(ok(("Resumed ", ""), (f"→ {new_id}", "muted")))
    else:
        console.print(ok(("New session ", ""), (f"→ {new_id}", "muted")))
    console.print()
