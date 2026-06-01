from unittest.mock import AsyncMock, MagicMock

from agent_app.tools.skill.skill_tool import SkillTool
from agent_cli.commands.builtin.reload_skill import CMD
from agent_cli.commands.registry import CommandRegistry

from ..conftest import render_output


def _ctx(*, has_skill: bool, tool: SkillTool | None = None) -> MagicMock:
    agent = MagicMock()
    agent.tool_registry.has = MagicMock(return_value=has_skill)
    if has_skill:
        agent.tool_registry.get = MagicMock(return_value=tool)
    agent._prompt_builder = MagicMock()
    agent._prompt_builder.build = MagicMock(return_value="new prompt")
    agent._make_builder_context = MagicMock(return_value={})
    registry = CommandRegistry()
    return MagicMock(
        agent=agent,
        registry=registry,
        save_session=AsyncMock(),
        refresh_completer=MagicMock(),
    )


async def test_reload_skill_when_tool_missing_returns_err() -> None:
    result = await CMD.handler(_ctx(has_skill=False), "")
    assert "Skill not available" in render_output(result.output)


async def test_reload_skill_when_tool_not_skilltool_returns_err() -> None:
    other_tool = MagicMock(spec=[])
    result = await CMD.handler(_ctx(has_skill=True, tool=other_tool), "")
    assert "Skill not available" in render_output(result.output)


async def test_reload_skill_force_reloads_and_refreshes_completer() -> None:
    tool = MagicMock(spec=SkillTool)
    tool.reload_skills = MagicMock()
    tool.loader = MagicMock()
    tool.loader.list_names = MagicMock(return_value=[])
    ctx = _ctx(has_skill=True, tool=tool)

    result = await CMD.handler(ctx, "")
    tool.reload_skills.assert_called_once()
    ctx.refresh_completer.assert_called_once()
    ctx.agent._prompt_builder.build.assert_called_once()
    assert ctx.agent.system_prompt == "new prompt"
    assert "Reloaded 0 skills" in render_output(result.output)
