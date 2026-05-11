"""/tasks — list background tasks."""
from __future__ import annotations

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import err, ok, relative_time, soft
from agent_cli.runtime import background

_STATUS_PRIORITY: dict[str, int] = {
    "running": 0, "completed": 1, "failed": 2, "cancelled": 3,
}
_STATUS_STYLE: dict[str, str] = {
    "running": "warning", "completed": "success",
    "failed": "error", "cancelled": "muted",
}


async def _cancel(ctx: CommandContext, target: str) -> CommandResult:
    if not target:
        return CommandResult(output=err((
            "Missing task id. Use: /tasks cancel <id|all>", "",
        )))
    if target == "all":
        cancelled = await background.cancel_all_with_note(ctx.agent)
        if not cancelled:
            return CommandResult(output=soft(("No running tasks to cancel", "")))
        n = len(cancelled)
        success = ok((f"Cancelled {n} task{'s' if n != 1 else ''}", ""))
    else:
        if not await background.cancel_with_note(ctx.agent, target):
            return CommandResult(output=err(
                ("No running task: ", ""), (target, "warning"),
            ))
        success = ok(("Cancelled ", ""), (target, "primary"))

    await ctx.save_session()
    return CommandResult(output=success)


async def _handler(ctx: CommandContext, args: str) -> CommandResult:
    parts = args.strip().split(maxsplit=1)
    if parts and parts[0].lower() == "cancel":
        target = parts[1].strip() if len(parts) > 1 else ""
        return await _cancel(ctx, target)
    if parts:
        return CommandResult(output=err(
            ("Unknown subcommand: ", ""), (parts[0], "warning"),
            (". Use: /tasks or /tasks cancel <id|all>", ""),
        ))
    return _render_list(ctx)


def _render_list(ctx: CommandContext) -> CommandResult:
    tasks = background.get_all(ctx.agent)
    if not tasks:
        return CommandResult(output=soft(("No background tasks", "")))

    rows: list[RenderableType] = []
    ordered = sorted(
        tasks, key=lambda t: (_STATUS_PRIORITY.get(t.status, 9), t.created_at),
    )
    for t in ordered:
        line = Text("  ")
        line.append(t.task_id[:8], style="muted")
        line.append("  ")
        line.append(
            t.status.ljust(10),
            style=_STATUS_STYLE.get(t.status, "default"),
        )
        line.append("  ")
        line.append(relative_time(t.created_at).rjust(8), style="muted")
        line.append("  ")
        line.append(t.tool_name.ljust(16), style="muted")
        line.append("  ")
        line.append(t.description or "—")
        rows.append(line)

    return CommandResult(output=Panel(
        Group(*rows),
        title="Background tasks",
        title_align="left",
        border_style="muted",
        padding=(0, 1),
        expand=False,
    ))



CMD = Command(
    name="/tasks",
    description="List background tasks, (/tasks cancel <id|all> to cancel)",
    handler=_handler,
)
