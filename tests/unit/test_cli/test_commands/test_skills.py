from unittest.mock import AsyncMock, MagicMock

from agent_cli.commands.base import Command, CommandResult
from agent_cli.commands.builtin.skills import CMD
from agent_cli.commands.registry import CommandRegistry

from .conftest import render_output


def _skill_cmd(name: str, desc: str = "do thing") -> Command:
    async def _h(ctx: object, args: str) -> CommandResult:
        return CommandResult()
    return Command(name=name, description=desc, handler=_h, is_skill=True)


def _ctx(registry: CommandRegistry) -> MagicMock:
    return MagicMock(registry=registry, save_session=AsyncMock())


async def test_skills_empty_returns_soft_message() -> None:
    r = CommandRegistry()
    result = await CMD.handler(_ctx(r), "")
    assert "No skills available" in render_output(result.output)


async def test_skills_renders_each_skill_command() -> None:
    r = CommandRegistry()
    r.register_command(_skill_cmd("/web", "search the web"))
    r.register_command(_skill_cmd("/db", "query the database"))
    result = await CMD.handler(_ctx(r), "")
    out = render_output(result.output)
    assert "/web" in out
    assert "search the web" in out
    assert "/db" in out
    assert "query the database" in out


async def test_skills_excludes_non_skill_commands() -> None:
    r = CommandRegistry()

    async def _h(ctx: object, args: str) -> CommandResult:
        return CommandResult()

    r.register_command(Command(name="/help", description="builtin", handler=_h))
    r.register_command(_skill_cmd("/web"))
    out = render_output((await CMD.handler(_ctx(r), "")).output)
    assert "/web" in out
    assert "/help" not in out


async def test_skills_description_truncated_when_long() -> None:
    r = CommandRegistry()
    long_desc = " ".join(["word"] * 200)
    r.register_command(_skill_cmd("/big", long_desc))
    out = render_output((await CMD.handler(_ctx(r), "")).output)
    assert "…" in out
    assert len(out) < len(long_desc) + 200
