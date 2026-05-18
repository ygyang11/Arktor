"""/rename — rename the current session (changes its id and file)."""
from __future__ import annotations

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import err, ok, soft
from agent_cli.runtime import session as sess


async def _handler(ctx: CommandContext, args: str) -> CommandResult:
    backend = ctx.session_backend
    new_id = args.strip()

    if not new_id:
        return CommandResult(output=soft(
            ("Current session: ", ""), (backend.session_id, "warning")
        ))

    if new_id == backend.session_id:
        return CommandResult(output=soft(("Name is same to current session", "")))

    if await backend.has_session(new_id):
        return CommandResult(output=err(
            ("Session already exists: ", ""), (new_id, "warning"),
        ))

    old_id = backend.session_id
    try:
        await sess.rename_session(ctx.agent, backend, new_id)
    except ValueError:
        return CommandResult(output=err(
            ("Invalid name: ", ""), (new_id, "warning"),
            (" — use letters, digits, '-' or '_'", "muted"),
        ))

    await ctx.save_session()
    ctx.refresh_completer()
    return CommandResult(output=ok(
        "Renamed ", (old_id, "muted"), (" → ", ""), (new_id, "warning"),
    ))


CMD = Command(
    name="/rename",
    description="Rename the current session name",
    handler=_handler,
)
