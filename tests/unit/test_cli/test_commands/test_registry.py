from collections import namedtuple
from unittest.mock import AsyncMock, MagicMock

from agent_app.tools.skill.skill_tool import SkillTool
from agent_cli.commands.base import Command, CommandResult
from agent_cli.commands.builtin import register_dynamic
from agent_cli.commands.registry import CommandRegistry


def _cmd(
    name: str = "/x",
    aliases: tuple[str, ...] = (),
    hidden: bool = False,
    is_skill: bool = False,
) -> tuple[Command, AsyncMock]:
    h = AsyncMock(return_value=CommandResult(output="ok"))
    return (
        Command(
            name=name,
            description="desc",
            handler=h,
            aliases=aliases,
            hidden=hidden,
            is_skill=is_skill,
        ),
        h,
    )


async def test_dispatch_known_slash_command_with_args() -> None:
    r = CommandRegistry()
    cmd, h = _cmd("/foo")
    r.register_command(cmd)
    res = await r.dispatch("/foo bar baz", MagicMock())
    h.assert_awaited_once()
    assert res is not None
    assert res.output == "ok"


async def test_dispatch_unknown_slash_command_name_returns_message() -> None:
    r = CommandRegistry()
    res = await r.dispatch("/nope", MagicMock())
    assert res is not None
    assert "Unknown command" in (res.output or "")


async def test_dispatch_non_command_returns_none() -> None:
    r = CommandRegistry()
    res = await r.dispatch("hello world", MagicMock())
    assert res is None


async def test_slash_prefixed_path_like_input_falls_through() -> None:
    r = CommandRegistry()
    for line in ("/tmp/a.py", "/foo.py", "/some/deep/path", "/a.b.c"):
        assert await r.dispatch(line, MagicMock()) is None, line


async def test_shape_valid_unknown_slash_returns_message_even_with_args() -> None:
    r = CommandRegistry()
    res = await r.dispatch("/abc some question", MagicMock())
    assert res is not None
    assert "Unknown command: /abc" in (res.output or "")


async def test_slash_prefixed_existing_path_falls_through() -> None:
    r = CommandRegistry()
    assert await r.dispatch("/tmp", MagicMock()) is None


async def test_plain_alias_exact_match_dispatches() -> None:
    r = CommandRegistry()
    cmd, h = _cmd("/exit", aliases=("/quit", "/q", "exit", "quit"))
    r.register_command(cmd)
    for line in ("/q", "exit", "QUIT"):
        await r.dispatch(line, MagicMock())
    assert h.await_count == 3


async def test_plain_alias_with_args_falls_through() -> None:
    r = CommandRegistry()
    cmd, h = _cmd("/exit", aliases=("/quit", "/q", "exit", "quit"))
    r.register_command(cmd)
    for line in ("exit code 137", "quit vim 怎么保存"):
        res = await r.dispatch(line, MagicMock())
        assert res is None, line
    assert h.await_count == 0


async def test_slash_alias_with_args_dispatches() -> None:
    r = CommandRegistry()
    cmd, h = _cmd("/exit", aliases=("/quit", "/q", "exit", "quit"))
    r.register_command(cmd)
    await r.dispatch("/q --force", MagicMock())
    h.assert_awaited_once()
    _, forwarded_args = h.await_args.args
    assert forwarded_args == "--force"


async def test_leading_whitespace_disqualifies_slash() -> None:
    r = CommandRegistry()
    cmd, h = _cmd("/foo")
    r.register_command(cmd)
    for line in (" /foo", "\t/foo", "  /foo bar"):
        assert await r.dispatch(line, MagicMock()) is None, line
    h.assert_not_awaited()


async def test_leading_whitespace_disqualifies_plain_alias() -> None:
    r = CommandRegistry()
    cmd, h = _cmd("/exit", aliases=("exit",))
    r.register_command(cmd)
    for line in (" exit", "\texit"):
        assert await r.dispatch(line, MagicMock()) is None, line
    h.assert_not_awaited()


async def test_multiline_args_preserved() -> None:
    r = CommandRegistry()
    cmd, h = _cmd("/explain")
    r.register_command(cmd)
    await r.dispatch("/explain line1\nline2\nline3", MagicMock())
    h.assert_awaited_once()
    _, forwarded_args = h.await_args.args
    assert forwarded_args == "line1\nline2\nline3"


async def test_slash_without_space_before_newline_falls_through() -> None:
    # "/explain\ncode" has no space between name and content → name contains
    # "\n" → regex fails → treat as not-a-command, pass to agent verbatim.
    r = CommandRegistry()
    cmd, h = _cmd("/explain")
    r.register_command(cmd)
    res = await r.dispatch("/explain\ncode", MagicMock())
    assert res is None
    h.assert_not_awaited()


def test_completions_dedup_and_hide() -> None:
    r = CommandRegistry()
    c1, _ = _cmd("/exit", aliases=("/quit",))
    c2, _ = _cmd("/hidden", hidden=True)
    r.register_command(c1)
    r.register_command(c2)
    names = {n for n, _ in r.get_completions()}
    assert names == {"/exit"}


def test_unregister_skills_drops_only_skill_commands() -> None:
    r = CommandRegistry()
    builtin, _ = _cmd("/help")
    s1, _ = _cmd("/foo", is_skill=True)
    s2, _ = _cmd("/bar", is_skill=True)
    r.register_command(builtin)
    r.register_command(s1)
    r.register_command(s2)
    r.unregister_skills()
    assert r.has("/help")
    assert not r.has("/foo")
    assert not r.has("/bar")


def test_unregister_skills_no_skills_is_noop() -> None:
    r = CommandRegistry()
    builtin, _ = _cmd("/help")
    r.register_command(builtin)
    r.unregister_skills()
    assert r.has("/help")


def test_list_skill_commands_only_skills_deduped() -> None:
    r = CommandRegistry()
    builtin, _ = _cmd("/help")
    s1, _ = _cmd("/foo", aliases=("/foo-alias",), is_skill=True)
    s2, _ = _cmd("/bar", is_skill=True)
    r.register_command(builtin)
    r.register_command(s1)
    r.register_command(s2)
    names = [c.name for c in r.list_skill_commands()]
    assert sorted(names) == ["/bar", "/foo"]


# ── register_dynamic (skill → slash-command) ─────────────────────────

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
