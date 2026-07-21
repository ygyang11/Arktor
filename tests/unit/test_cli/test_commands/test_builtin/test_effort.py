from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_cli.commands.builtin.effort import CMD
from agent_harness.core.config import LLMConfig, SubModelConfig

from ..conftest import render_output


def _agent(*, separate_sub: bool = False) -> MagicMock:
    agent = MagicMock()
    agent.llm = MagicMock()
    agent.llm.config.reasoning_effort = "medium"
    agent.sub_llm = MagicMock() if separate_sub else agent.llm
    agent.context.config.llm = LLMConfig(
        reasoning_effort="medium",
        sub_model=(
            SubModelConfig(model="gpt-4o-mini", reasoning_effort="low")
            if separate_sub
            else None
        ),
    )
    if separate_sub:
        agent.sub_llm.config.reasoning_effort = "low"
    return agent


async def test_set_effort_updates_main_no_persistence() -> None:
    agent = _agent()
    save = AsyncMock()
    result = await CMD.handler(MagicMock(agent=agent, save_session=save), "xhigh")
    assert agent.llm.config.reasoning_effort == "xhigh"
    assert agent.context.config.llm.reasoning_effort == "xhigh"
    save.assert_not_called()
    assert "Reasoning effort set to xhigh" in render_output(result.output)


@pytest.mark.parametrize(
    "value",
    ["wibble", "none", "off", "disabled", "xhigh", "sub", "main high"],
)
async def test_main_effort_lenient_passthrough(value: str) -> None:
    agent = _agent()
    await CMD.handler(MagicMock(agent=agent), value)
    assert agent.llm.config.reasoning_effort == value
    assert agent.context.config.llm.reasoning_effort == value


async def test_set_sub_effort_updates_runtime_and_nested_config() -> None:
    agent = _agent(separate_sub=True)
    result = await CMD.handler(MagicMock(agent=agent), "sub xhigh")
    assert agent.sub_llm.config.reasoning_effort == "xhigh"
    assert agent.context.config.llm.sub_model is not None
    assert agent.context.config.llm.sub_model.reasoning_effort == "xhigh"
    assert agent.llm.config.reasoning_effort == "medium"
    assert "Sub-model reasoning effort set to xhigh" in render_output(result.output)


async def test_set_sub_effort_requires_separate_sub() -> None:
    agent = _agent()
    result = await CMD.handler(MagicMock(agent=agent), "sub xhigh")
    assert agent.llm.config.reasoning_effort == "medium"
    assert agent.context.config.llm.reasoning_effort == "medium"
    assert "No separate sub model" in render_output(result.output)


async def test_sub_effort_lenient_passthrough() -> None:
    agent = _agent(separate_sub=True)
    result = await CMD.handler(MagicMock(agent=agent), "sub high extra")
    assert agent.llm.config.reasoning_effort == "medium"
    assert agent.sub_llm.config.reasoning_effort == "high extra"
    assert agent.context.config.llm.sub_model is not None
    assert agent.context.config.llm.sub_model.reasoning_effort == "high extra"
    assert "Sub-model reasoning effort set to high extra" in render_output(result.output)


async def test_no_arg_shows_main_effort_only() -> None:
    agent = _agent(separate_sub=True)
    result = await CMD.handler(MagicMock(agent=agent), "")
    rendered = render_output(result.output)
    assert "medium" in rendered
    assert "low" not in rendered


async def test_no_arg_unset_shows_default() -> None:
    agent = _agent()
    agent.llm.config.reasoning_effort = None
    result = await CMD.handler(MagicMock(agent=agent), "")
    assert "default" in render_output(result.output)


def test_effort_command_metadata_mentions_sub() -> None:
    assert "[sub]" in CMD.description
