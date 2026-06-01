from unittest.mock import MagicMock

from agent_cli.commands.builtin.help import CMD

from ..conftest import render_output


async def test_help_lists_registered_completions() -> None:
    registry = MagicMock()
    registry.get_completions = MagicMock(return_value=[
        ("/help", "List commands"),
        ("/exit", "Exit the REPL"),
    ])
    ctx = MagicMock(registry=registry)

    result = await CMD.handler(ctx, "")
    assert result.output is not None
    rendered = render_output(result.output)
    assert "/help" in rendered
    assert "List commands" in rendered
    assert "/exit" in rendered


def test_help_command_metadata() -> None:
    assert CMD.name == "/help"
    assert CMD.aliases == ()
