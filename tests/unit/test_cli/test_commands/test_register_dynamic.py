from collections import namedtuple
from unittest.mock import AsyncMock, MagicMock

from agent_app.tools.skill.skill_tool import SkillTool
from agent_cli.commands.base import Command, CommandResult
from agent_cli.commands.builtin import register_dynamic
from agent_cli.commands.registry import CommandRegistry


_FakeMeta = namedtuple("_FakeMeta", ["description"])


def _agent(*, has_skill: bool, tool: SkillTool | None) -> MagicMock:
    agent = MagicMock()
    agent.tool_registry.has = MagicMock(return_value=has_skill)
    if has_skill:
        agent.tool_registry.get = MagicMock(return_value=tool)
    return agent


def _skill_tool(names: list[str], descs: dict[str, str]) -> SkillTool:
    tool = MagicMock(spec=SkillTool)
    tool.loader = MagicMock()
    tool.loader.list_names = MagicMock(return_value=names)
    tool.loader._metadata = {n: _FakeMeta(description=descs[n]) for n in names}
    tool.execute = AsyncMock(return_value="<skill-loaded>body</skill-loaded>")
    tool.reload_skills = MagicMock()
    return tool


def test_register_dynamic_no_skill_tool_is_noop() -> None:
    registry = CommandRegistry()
    register_dynamic(registry, _agent(has_skill=False, tool=None))
    assert registry.list_names() == []


def test_register_dynamic_non_skilltool_is_noop() -> None:
    other = MagicMock(spec=[])
    registry = CommandRegistry()
    register_dynamic(registry, _agent(has_skill=True, tool=other))
    assert registry.list_names() == []


def test_register_dynamic_registers_one_command_per_skill() -> None:
    tool = _skill_tool(["web", "db"], {"web": "search", "db": "query"})
    registry = CommandRegistry()
    register_dynamic(registry, _agent(has_skill=True, tool=tool))
    assert registry.has("/web")
    assert registry.has("/db")
    cmd = registry.get("/web")
    assert cmd.is_skill is True
    assert cmd.description == "search"


def test_register_dynamic_skips_invalid_names(caplog) -> None:
    tool = _skill_tool(
        ["valid-skill", "has space", "2do", "a.b", "snake_case_ok"],
        {
            "valid-skill": "d1",
            "has space": "d2",
            "2do": "d3",
            "a.b": "d4",
            "snake_case_ok": "d5",
        },
    )
    registry = CommandRegistry()
    with caplog.at_level("WARNING"):
        register_dynamic(registry, _agent(has_skill=True, tool=tool))
    assert registry.has("/valid-skill")
    assert registry.has("/snake_case_ok")
    assert not registry.has("/has space")
    assert not registry.has("/2do")
    assert not registry.has("/a.b")
    invalid_warnings = [r.message for r in caplog.records if "invalid" in r.message]
    assert len(invalid_warnings) == 3


def test_register_dynamic_skips_when_name_shadows_builtin(caplog) -> None:
    async def _stub(ctx: object, args: str) -> CommandResult:
        return CommandResult()

    registry = CommandRegistry()
    registry.register_command(
        Command(name="/help", description="builtin", handler=_stub),
    )
    tool = _skill_tool(["help"], {"help": "skill help"})
    with caplog.at_level("WARNING"):
        register_dynamic(registry, _agent(has_skill=True, tool=tool))
    assert any("shadowed" in r.message for r in caplog.records)
    # `/help` still points to the builtin (not is_skill)
    assert registry.get("/help").is_skill is False


async def test_skill_command_handler_runs_tool_and_returns_agent_input() -> None:
    tool = _skill_tool(["web"], {"web": "search"})
    registry = CommandRegistry()
    register_dynamic(registry, _agent(has_skill=True, tool=tool))
    cmd = registry.get("/web")
    result = await cmd.handler(MagicMock(), "find docs")
    tool.execute.assert_awaited_once_with(skill_name="web", args="find docs")
    assert result.agent_input is not None
    assert "find docs" in result.agent_input
    assert "<system-reminder>" in result.agent_input
    assert "web" in result.agent_input
    assert "<skill-loaded>" in result.agent_input


async def test_skill_command_handler_no_args_omits_user_prefix() -> None:
    tool = _skill_tool(["web"], {"web": "search"})
    registry = CommandRegistry()
    register_dynamic(registry, _agent(has_skill=True, tool=tool))
    cmd = registry.get("/web")
    result = await cmd.handler(MagicMock(), "")
    assert result.agent_input is not None
    assert result.agent_input.startswith("<system-reminder>")


async def test_skill_command_handler_when_skill_removed_reloads_then_errs() -> None:
    tool = _skill_tool(["web"], {"web": "search"})
    registry = CommandRegistry()
    register_dynamic(registry, _agent(has_skill=True, tool=tool))
    cmd = registry.get("/web")
    tool.loader.list_names = MagicMock(return_value=[])
    result = await cmd.handler(MagicMock(), "")
    tool.reload_skills.assert_called_once()
    assert result.output is not None
    assert result.agent_input is None
