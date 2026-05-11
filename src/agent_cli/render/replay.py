"""Static replay of a session's recent turns — no adapter, no Live."""
from __future__ import annotations

from rich.console import Console
from rich.text import Text

from agent_cli.commands.ui import ok
from agent_cli.render.markdown_stream import render_markdown_block
from agent_cli.render.tool_display import render_completed_call
from agent_cli.runtime.session import get_messages
from agent_cli.theme import PROMPT, CliTheme
from agent_harness.agent.base import BaseAgent
from agent_harness.core.message import Message, Role, ToolResult


def slice_last_turns(messages: list[Message], n: int) -> list[Message]:
    seen = 0
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].role == Role.USER:
            seen += 1
            if seen == n:
                return [m for m in messages[idx:] if m.role != Role.SYSTEM]
    return [m for m in messages if m.role != Role.SYSTEM]


def _index_results(messages: list[Message]) -> dict[str, ToolResult]:
    out: dict[str, ToolResult] = {}
    for m in messages:
        if m.role == Role.TOOL and m.tool_result is not None:
            out[m.tool_result.tool_call_id] = m.tool_result
    return out


def replay(console: Console, theme: CliTheme, messages: list[Message]) -> None:
    if not messages:
        return
    results = _index_results(messages)
    for m in messages:
        if m.role == Role.USER:
            if m.content:
                line = Text(f"{PROMPT} ", style="primary")
                line.append(m.content)
                console.print(line)
                console.print()
            continue
        if m.role == Role.ASSISTANT:
            if m.content:
                render_markdown_block(console, m.content, theme)
                console.print()
            for tc in m.tool_calls or ():
                console.print(render_completed_call(tc, results.get(tc.id)))
                console.print()


def render_post_switch(
    agent: BaseAgent,
    console: Console,
    theme: CliTheme,
    new_id: str,
) -> None:
    console.clear()
    msgs = get_messages(agent)
    if msgs:
        replay(console, theme, slice_last_turns(msgs, 5))
        console.print(ok(("Resumed ", ""), (f"→ {new_id}", "muted")))
    else:
        console.print(ok(("New session ", ""), (f"→ {new_id}", "muted")))
    console.print()
