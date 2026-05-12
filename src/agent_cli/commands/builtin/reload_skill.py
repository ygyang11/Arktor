"""/reload-skill — rescan skills directories and refresh command registration."""
from __future__ import annotations

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import err, ok


async def _handler(ctx: CommandContext, args: str) -> CommandResult:
    from agent_cli.commands.builtin import register_dynamic

    if not ctx.agent.tool_registry.has("skill_tool"):
        return CommandResult(output=err(("Skill not available", "")))
    tool = ctx.agent.tool_registry.get("skill_tool")
    reload = getattr(tool, "reload_skills", None)
    if not callable(reload):
        return CommandResult(output=err(("Skill not available", "")))
    reload()

    ctx.registry.unregister_skills()
    register_dynamic(ctx.registry, ctx.agent)
    ctx.refresh_completer()

    agent = ctx.agent
    agent.system_prompt = agent._prompt_builder.build(agent._make_builder_context())

    n = len(ctx.registry.list_skill_commands())
    return CommandResult(output=ok((f"Reloaded {n} skills", "")))


CMD = Command(
    name="/reload-skill",
    description="Re-scan skills directory",
    handler=_handler,
)
