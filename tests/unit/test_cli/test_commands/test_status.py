from unittest.mock import MagicMock

from agent_cli.commands.builtin.status import CMD

from .conftest import render_output


async def test_status_prints_three_sections() -> None:
    console = MagicMock()
    agent = MagicMock()
    agent.llm.model_name = "gpt-4o"
    agent._approval.mode = "auto"
    stm = MagicMock()
    stm._messages = [1, 2, 3]
    stm.displayed_input_tokens = 12_345
    stm.max_tokens = 128_000
    agent.context.short_term_memory = stm

    tool = MagicMock()
    tool.name = "other_tool"
    agent.tools = [tool]
    agent.tool_registry.has = MagicMock(return_value=False)
    agent._bg_manager.get_all = MagicMock(return_value=[])

    ctx = MagicMock(agent=agent, session_id="sid-1")
    result = await CMD.handler(ctx, "")

    assert result.output is not None
    console.print.assert_not_called()
    rendered = render_output(result.output)
    assert "Identity" in rendered
    assert "Config" in rendered
    assert "Runtime" in rendered
    assert "sid-1" in rendered
    assert "gpt-4o" in rendered


def test_status_command_metadata() -> None:
    assert CMD.name == "/status"
    assert "status" in CMD.description.lower()
