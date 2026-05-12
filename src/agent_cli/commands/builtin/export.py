"""/export — write the current session transcript to markdown."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import home_relative_path, ok
from agent_cli.render.tool_display import args_repr
from agent_harness.core.message import Message, Role


_ROLE_HEADERS: dict[Role, str] = {
    Role.USER: "## User",
    Role.ASSISTANT: "## Assistant",
    Role.SYSTEM: "## System",
}


def _format_message(m: Message) -> str:
    header = _ROLE_HEADERS.get(m.role, f"## {m.role.value}")
    parts: list[str] = []
    if m.content:
        parts.append(m.content)
    if m.tool_calls:
        tool_lines = "\n".join(
            f"- `{tc.name}({args_repr(tc.arguments)})`" for tc in m.tool_calls
        )
        parts.append(f"**Tool calls:**\n{tool_lines}")
    body = "\n\n".join(parts)
    return f"{header}\n\n{body}\n"


def _format_tool_group(
    prev_assistant: Message | None,
    tools: list[Message],
) -> str:
    name_by_id: dict[str, str] = {}
    if prev_assistant is not None and prev_assistant.tool_calls:
        name_by_id = {tc.id: tc.name for tc in prev_assistant.tool_calls}

    parts: list[str] = []
    for m in tools:
        tr = m.tool_result
        name = name_by_id.get(tr.tool_call_id) if tr else None
        err_suffix = " (error)" if tr is not None and tr.is_error else ""
        body = (tr.content if tr is not None else m.content) or ""
        if name:
            parts.append(f"**`{name}`**{err_suffix}:\n\n{body}")
        else:
            parts.append(body)
    return "## Tool\n\n" + "\n\n".join(parts) + "\n"


async def _handler(ctx: CommandContext, args: str) -> CommandResult:
    out_dir = Path.home() / ".agent-harness" / "sessions" / ctx.session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"export-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    msgs = await ctx.agent.context.short_term_memory.get_context_messages()

    blocks: list[str] = []
    prev_assistant: Message | None = None
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m.role == Role.TOOL:
            group: list[Message] = []
            while i < len(msgs) and msgs[i].role == Role.TOOL:
                group.append(msgs[i])
                i += 1
            blocks.append(_format_tool_group(prev_assistant, group))
            continue
        blocks.append(_format_message(m))
        if m.role == Role.ASSISTANT:
            prev_assistant = m
        i += 1

    fp.write_text("\n".join(blocks), encoding="utf-8")
    return CommandResult(output=ok(
        ("Exported → ", ""),
        (home_relative_path(fp), "primary"),
    ))


CMD = Command(
    name="/export",
    description="Export the current session transcript to markdown",
    handler=_handler,
)
