"""/effort — switch LLM reasoning effort for the current process."""
from __future__ import annotations

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import err, info, ok


async def handle(ctx: CommandContext, args: str) -> CommandResult:
    agent = ctx.agent
    raw = args.strip()
    if not raw:
        current = agent.llm.config.reasoning_effort
        return CommandResult(output=info(
            "Current reasoning effort: ",
            (current or "default"),
        ))

    parts = raw.split(maxsplit=1)
    if len(parts) == 2 and parts[0] == "sub":
        value = parts[1]
        sub_cfg = agent.context.config.llm.sub_model
        if (
            agent.sub_llm is agent.llm
            or sub_cfg is None
            or sub_cfg.model is None
        ):
            return CommandResult(output=err(
                "No separate sub model is configured",
            ))
        agent.sub_llm.config.reasoning_effort = value
        sub_cfg.reasoning_effort = value
        return CommandResult(output=ok(
            "Sub-model reasoning effort set to ",
            value,
        ))

    agent.llm.config.reasoning_effort = raw
    agent.context.config.llm.reasoning_effort = raw
    return CommandResult(output=ok(
        "Reasoning effort set to ",
        raw,
    ))


CMD = Command(
    name="/effort",
    description="Set LLM reasoning effort for this session, support [sub]",
    handler=handle,
)
