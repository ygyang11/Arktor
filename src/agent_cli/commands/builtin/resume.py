"""/resume — switch to another session by id, or list recent ones."""
from __future__ import annotations

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import err, render_session_list, soft


async def _handler(ctx: CommandContext, args: str) -> CommandResult:
    backend = ctx.session_backend
    target = args.strip()
    if target:
        if not await backend.has_session(target):
            return CommandResult(output=err(
                ("No session found: ", ""), (target, "warning"),
            ))
        return CommandResult(new_session_id=target)

    metas = [m for m in await backend.list_states() if m.session_id != ctx.session_id]
    if not metas:
        return CommandResult(output=soft(("No other sessions to resume", "")))
    return CommandResult(output=render_session_list(metas))


CMD = Command(
    name="/resume",
    description="Resume a session (use /resume <id> to switch)",
    handler=_handler,
)
