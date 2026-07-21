from unittest.mock import AsyncMock, MagicMock, patch

from agent_cli.commands.builtin.model import CMD
from agent_harness.core.config import LLMConfig, SubModelConfig

from ..conftest import render_output


def _agent(*, distinct_sub: bool = False) -> MagicMock:
    agent = MagicMock()
    agent.llm = MagicMock(model_name="gpt-4o")
    agent.sub_llm = MagicMock(model_name="gpt-4o-mini") if distinct_sub else agent.llm
    agent.context.config.llm = LLMConfig(
        provider="openai",
        model="gpt-4o",
        temperature=0.3,
        max_tokens=8000,
        reasoning_effort="high",
        sub_model=(
            SubModelConfig(model="gpt-4o-mini", reasoning_effort="low")
            if distinct_sub
            else None
        ),
    )
    return agent


async def test_switch_model_updates_config_and_aliases_sub() -> None:
    agent = _agent()
    old_config = agent.context.config.llm
    save = AsyncMock()
    new_llm = MagicMock(model_name="gpt-5")

    with patch("agent_cli.commands.builtin.model.create_llm", return_value=new_llm) as create:
        result = await CMD.handler(MagicMock(agent=agent, save_session=save), "gpt-5")

    (cfg,), _ = create.call_args
    assert cfg.model == "gpt-5"
    assert cfg.temperature == 0.3
    assert cfg.max_tokens == 8000
    assert cfg.reasoning_effort == "high"
    assert agent.context.config.llm is cfg
    assert agent.context.config.llm is not old_config
    agent.replace_llms.assert_called_once_with(new_llm, new_llm)
    save.assert_not_called()
    assert "Model switched to gpt-5" in render_output(result.output)


async def test_switch_model_preserves_distinct_sub() -> None:
    agent = _agent(distinct_sub=True)
    old_sub = agent.sub_llm
    new_llm = MagicMock(model_name="gpt-5")

    with patch("agent_cli.commands.builtin.model.create_llm", return_value=new_llm):
        await CMD.handler(MagicMock(agent=agent), "gpt-5")

    agent.replace_llms.assert_called_once_with(new_llm, old_sub)
    assert agent.context.config.llm.sub_model == SubModelConfig(
        model="gpt-4o-mini",
        reasoning_effort="low",
    )


async def test_switch_model_failure_keeps_current_state() -> None:
    agent = _agent(distinct_sub=True)
    old_config = agent.context.config.llm

    with patch(
        "agent_cli.commands.builtin.model.create_llm",
        side_effect=ValueError("no client"),
    ):
        result = await CMD.handler(MagicMock(agent=agent), "gpt-5")

    assert agent.context.config.llm is old_config
    agent.replace_llms.assert_not_called()
    assert "Failed to switch model" in render_output(result.output)


async def test_no_arg_shows_current_model() -> None:
    agent = _agent()
    agent.llm.model_name = "claude-4"
    result = await CMD.handler(MagicMock(agent=agent), "")
    assert "claude-4" in render_output(result.output)
