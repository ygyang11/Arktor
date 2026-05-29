"""/effort — switch LLM reasoning effort for the current process."""
from __future__ import annotations

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import info, ok


async def handle(ctx: CommandContext, args: str) -> CommandResult:
    agent = ctx.agent
    raw = args.strip()
    if not raw:
        current = agent.llm.config.reasoning_effort
        return CommandResult(output=info(
            "Current reasoning effort: ",
            (current or "default"),
        ))

    agent.llm.config.reasoning_effort = raw
    agent.context.config.llm.reasoning_effort = raw
    return CommandResult(output=ok(
        "Reasoning effort set to ",
        raw,
    ))


CMD = Command(
    name="/effort",
    description="Set LLM reasoning effort for this session",
    handler=handle,
)
