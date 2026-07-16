"""/plan — toggle plan mode; an optional argument is submitted on entry."""
from __future__ import annotations

from rich.text import Text

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import MODE_INFO, err, ok
from agent_cli.runtime import plan_mode
from agent_cli.runtime import session as sess_rt
from agent_cli.runtime.goal import mode as goal_mode
from agent_harness.agent.base import BaseAgent


def _entered_output(agent: BaseAgent) -> Text:
    info = MODE_INFO[sess_rt.get_policy(agent).mode]
    return ok(
        ("Plan mode entered  ", ""),
        (f"{info.label} Approval ({info.short})", "muted"),
    )


async def _handler(ctx: CommandContext, args: str) -> CommandResult:
    if plan_mode.is_active(ctx.agent):
        plan_mode.exit(ctx.agent)
        await ctx.save_session()
        return CommandResult(output=ok(("Plan mode exited", "")))

    if goal_mode.is_active(ctx.agent):
        return CommandResult(output=err(
            "Can't enter plan mode while a goal is active",
            (" · /goal pause or /goal clear first", "muted"),
        ))

    plan_mode.enter(ctx.agent)
    await ctx.save_session()
    arg = args.strip()
    if arg:
        return CommandResult(output=_entered_output(ctx.agent), agent_input=arg)
    return CommandResult(output=_entered_output(ctx.agent))


CMD = Command(
    name="/plan",
    description="Toggle Plan mode, accepts an optional description to submit on entry",
    handler=_handler,
)
