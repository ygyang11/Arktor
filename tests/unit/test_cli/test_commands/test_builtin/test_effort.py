from unittest.mock import AsyncMock, MagicMock

from agent_cli.commands.builtin.effort import CMD

from ..conftest import render_output


async def test_set_effort_updates_llm_and_config_no_persistence() -> None:
    agent = MagicMock()
    save = AsyncMock()
    ctx = MagicMock(agent=agent, save_session=save)
    result = await CMD.handler(ctx, "xhigh")
    assert agent.llm.config.reasoning_effort == "xhigh"
    assert agent.context.config.llm.reasoning_effort == "xhigh"
    save.assert_not_called()
    assert "Reasoning effort set to xhigh" in render_output(result.output)


async def test_effort_lenient_passthrough_accepts_any_value() -> None:
    agent = MagicMock()
    await CMD.handler(MagicMock(agent=agent), "wibble")
    assert agent.llm.config.reasoning_effort == "wibble"


async def test_passthrough_does_not_special_case_any_word() -> None:
    for word in ("none", "off", "disabled", "xhigh"):
        agent = MagicMock()
        await CMD.handler(MagicMock(agent=agent), word)
        assert agent.llm.config.reasoning_effort == word
        assert agent.context.config.llm.reasoning_effort == word


async def test_no_arg_shows_current_effort() -> None:
    agent = MagicMock()
    agent.llm.config.reasoning_effort = "medium"
    result = await CMD.handler(MagicMock(agent=agent), "")
    assert "medium" in render_output(result.output)


async def test_no_arg_unset_shows_default() -> None:
    agent = MagicMock()
    agent.llm.config.reasoning_effort = None
    result = await CMD.handler(MagicMock(agent=agent), "")
    assert "default" in render_output(result.output)
