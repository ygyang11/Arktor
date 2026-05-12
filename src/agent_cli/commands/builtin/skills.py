"""/skills — list available skills as runnable commands."""
from __future__ import annotations

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import render_skill_list, soft


async def _handler(ctx: CommandContext, args: str) -> CommandResult:
    cmds = ctx.registry.list_skill_commands()
    if not cmds:
        return CommandResult(output=soft(("No skills available", "")))
    return CommandResult(output=render_skill_list(cmds))


CMD = Command(
    name="/skills",
    description="List available skills",
    handler=_handler,
)
