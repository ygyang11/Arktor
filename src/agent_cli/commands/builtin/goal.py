"""/goal - set, view, or manage a persistent goal."""
from __future__ import annotations

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import err, ok, render_goal_panel
from agent_cli.runtime import plan_mode
from agent_cli.runtime.goal import mode as goal_mode
from agent_harness.utils.token_counter import count_tokens

_CONTROLS = {"pause", "resume", "clear"}


async def _handler(ctx: CommandContext, args: str) -> CommandResult:
    agent = ctx.agent
    arg = args.strip()

    if not arg:
        g = goal_mode.get_state(agent)
        tokens = (
            g.tokens_used(agent.context.usage_meter.total.total_tokens)
            if g is not None
            else None
        )
        return CommandResult(output=render_goal_panel(g, tokens=tokens))

    if arg in _CONTROLS:
        return await _control(ctx, arg)

    if goal_mode.has_live_goal(agent):
        return CommandResult(output=err(
            "A goal already exists",
            (" · /goal clear before setting another", "muted"),
        ))

    if plan_mode.is_active(agent):
        return CommandResult(output=err(
            "Can't set a goal in plan mode",
            (" · exit plan mode first", "muted"),
        ))

    objective_tokens = count_tokens(arg, model=agent.llm.model_name)
    if objective_tokens > goal_mode.MAX_OBJECTIVE_TOKENS:
        return CommandResult(output=err(
            "Goal objective is too large for a persistent goal",
            (
                f" · estimated {objective_tokens:,} tokens, "
                f"max {goal_mode.MAX_OBJECTIVE_TOKENS:,}",
                "muted",
            ),
            (" · put details in a file and reference it.", "muted"),
        ))

    goal_mode.begin(agent, arg)
    await ctx.save_session()
    return CommandResult(
        output=ok(("Goal set", ""), (" · working toward it across turns", "muted")),
        agent_input=goal_mode.make_start_input(arg),
    )


async def _control(ctx: CommandContext, action: str) -> CommandResult:
    agent = ctx.agent

    if action == "pause":
        g = goal_mode.pause(agent, reason="paused by user")
        if g is None:
            return CommandResult(output=err("No active goal to pause"))
        await ctx.save_session()
        return CommandResult(output=ok(("Goal paused", "")))

    if action == "resume":
        if plan_mode.is_active(agent):
            return CommandResult(output=err(
                "Can't resume a goal in plan mode",
                (" · exit plan mode first", "muted"),
            ))
        g = goal_mode.resume(agent)
        if g is None:
            return CommandResult(output=err("No paused goal to resume"))
        await ctx.save_session()
        return CommandResult(
            output=ok(("Goal resumed, continuing...", "")),
            agent_input=goal_mode.make_resume_message(g.objective),
        )

    goal_mode.clear(agent)
    await ctx.save_session()
    return CommandResult(output=ok(("Goal cleared", "")))


CMD = Command(
    name="/goal",
    description="Set, view or manage a persistent goal (pause|resume|clear)",
    handler=_handler,
)
