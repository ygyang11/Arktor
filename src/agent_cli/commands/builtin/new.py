"""/new — start a fresh session with a new id."""
from __future__ import annotations

from uuid import uuid4

from agent_cli.commands.base import Command, CommandContext, CommandResult


async def _handler(ctx: CommandContext, args: str) -> CommandResult:
    return CommandResult(new_session_id=str(uuid4()))


CMD = Command(name="/new", description="Start a fresh session", handler=_handler)
