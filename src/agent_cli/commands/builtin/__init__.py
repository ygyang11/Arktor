"""builtin command registration."""
from __future__ import annotations

import logging

from agent_app.tools.skill.skill_tool import SkillTool
from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.builtin import (
    clear,
    compact,
    context,
    copy,
    debug,
    diff,
    effort,
    exit,
    export,
    feedback,
    help,
    init,
    model,
    new,
    permissions,
    plan,
    reload_skill,
    rename,
    resume,
    review,
    skills,
    status,
    tasks,
    theme,
    usage,
)
from agent_cli.commands.registry import SLASH_CMD_RE, CommandRegistry
from agent_cli.commands.ui import err
from agent_harness.agent.base import BaseAgent

logger = logging.getLogger(__name__)


def register_builtin(registry: CommandRegistry) -> None:
    for module in (
        help, exit, clear, compact, context, copy, debug, diff, effort, export, feedback,
        init, model, permissions, plan, reload_skill, rename, resume, review,
        new, skills, status, tasks, theme, usage,
    ):
        registry.register_command(module.CMD)


def register_dynamic(registry: CommandRegistry, agent: BaseAgent) -> None:
    if not agent.tool_registry.has("skill_tool"):
        return
    raw_tool = agent.tool_registry.get("skill_tool")
    if not isinstance(raw_tool, SkillTool):
        return
    tool: SkillTool = raw_tool
    for name in tool.loader.list_names():
        key = f"/{name}"
        if not SLASH_CMD_RE.match(key):
            logger.warning(
                "Skill '%s' has an invalid slash-command name, skip", name,
            )
            continue
        if registry.has(key.lower()):
            logger.warning(
                "Skill '%s' shadowed by builtin command, skip", name,
            )
            continue
        registry.register_command(_make_skill_cmd(name, tool))


def _make_skill_cmd(name: str, tool: SkillTool) -> Command:
    desc = tool.loader._metadata[name].description

    async def handler(ctx: CommandContext, args: str) -> CommandResult:
        if name not in tool.loader.list_names():
            reload = getattr(tool, "reload_skills", None)
            if callable(reload):
                reload()
            if name not in tool.loader.list_names():
                return CommandResult(output=err(
                    ("Skill removed: ", ""), (name, "warning"),
                    (". Run /reload-skill to refresh.", ""),
                ))

        body = await tool.execute(skill_name=name, args=args)
        user_args = args.strip()
        bridge = (
            f"<system-reminder>The user has explicitly requested the {name} "
            "skill. Apply the skill instructions below to address their "
            "request.</system-reminder>"
        )
        prompt = (
            f"{user_args}\n\n{bridge}\n\n{body}" if user_args
            else f"{bridge}\n\n{body}"
        )
        return CommandResult(agent_input=prompt)

    return Command(
        name=f"/{name}", description=desc, handler=handler, is_skill=True,
    )
