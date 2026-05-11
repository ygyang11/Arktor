"""/new — start a fresh session with a new id."""
from __future__ import annotations

from uuid import uuid4

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import ok


async def _handler(ctx: CommandContext, args: str) -> CommandResult:
    new_id = str(uuid4())
    return CommandResult(
        output=ok(("New session → ", ""), (new_id, "primary")),
        new_session_id=new_id,
    )


CMD = Command(name="/new", description="Start a fresh session", handler=_handler)
