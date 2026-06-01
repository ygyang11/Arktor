from unittest.mock import MagicMock

from agent_cli.commands.builtin.exit import CMD


async def test_exit_returns_should_exit() -> None:
    ctx = MagicMock()
    result = await CMD.handler(ctx, "")
    assert result.should_exit is True
    assert result.output is None


def test_exit_command_registers_plain_and_slash_aliases() -> None:
    assert CMD.name == "/exit"
    assert set(CMD.aliases) == {"/quit", "/q", "exit", "quit"}
