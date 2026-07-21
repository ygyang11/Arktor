"""/model — switch LLM model for the current process."""
from __future__ import annotations

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import err, info, ok
from agent_harness.llm import create_llm


async def handle(ctx: CommandContext, args: str) -> CommandResult:
    new_model = args.strip()
    agent = ctx.agent
    if not new_model:
        return CommandResult(output=info(
            "Current model: ",
            (agent.llm.model_name, "bold"),
        ))

    config = agent.context.config
    try:
        new_cfg = config.llm.model_copy(update={"model": new_model})
        new_llm = create_llm(new_cfg)
    except Exception as e:
        return CommandResult(output=err(f"Failed to switch model: {e}"))

    new_sub_llm = agent.sub_llm if agent.sub_llm is not agent.llm else new_llm
    config.llm = new_cfg
    agent.replace_llms(new_llm, new_sub_llm)
    return CommandResult(output=ok(
        "Model switched to ",
        (new_model, "bold"),
    ))


CMD = Command(
    name="/model",
    description="Switch LLM model for this session",
    handler=handle,
)
