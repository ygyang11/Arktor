"""/help — list registered commands."""
from __future__ import annotations

from rich.console import Group
from rich.table import Table
from rich.text import Text

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import info, short_desc


async def handle(ctx: CommandContext, args: str) -> CommandResult:
    tbl = Table(box=None, show_header=False, padding=(0, 2))
    tbl.add_column(style="primary", no_wrap=True)
    tbl.add_column(style="muted")
    for name, desc in ctx.registry.get_completions():
        tbl.add_row(name, short_desc(desc))
    return CommandResult(output=Group(info("Available commands"), Text(""), tbl))


CMD = Command(
    name="/help",
    description="List available commands",
    handler=handle,
)
