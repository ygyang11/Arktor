from unittest.mock import AsyncMock, MagicMock

from agent_cli.commands.base import Command, CommandResult
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
