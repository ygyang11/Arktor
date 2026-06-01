from unittest.mock import AsyncMock, MagicMock, patch

from agent_cli.commands.builtin.model import CMD

from ..conftest import render_output


async def test_switch_model_updates_all_sites_no_persistence() -> None:
    agent = MagicMock()
    agent.llm.model_name = "gpt-4o"
    compressor = MagicMock()
    agent.context.short_term_memory.compressor = compressor
    agent.context.config.memory.compression.summary_model = None
    save = AsyncMock()
    ctx = MagicMock(agent=agent, save_session=save)

    with patch("agent_cli.commands.builtin.model.create_llm") as create:
        new_llm = MagicMock()
        create.return_value = new_llm
        result = await CMD.handler(ctx, "gpt-5")

    assert agent.llm is new_llm
    assert agent.context.short_term_memory.model == "gpt-5"
    assert compressor._model == "gpt-5"
    assert compressor._llm is new_llm
    assert agent.context.config.llm.model == "gpt-5"
    save.assert_not_called()
    rendered = render_output(result.output)
    assert "Model switched to gpt-5" in rendered


async def test_switch_model_respects_separate_summary_model() -> None:
    agent = MagicMock()
    agent.llm.model_name = "gpt-4o"
    compressor = MagicMock()
    compressor._model = "gpt-4o-mini"
    compressor._llm = "separate_llm"
    agent.context.short_term_memory.compressor = compressor
    agent.context.config.memory.compression.summary_model = "gpt-4o-mini"

    with patch("agent_cli.commands.builtin.model.create_llm") as create:
        create.return_value = MagicMock()
        await CMD.handler(MagicMock(agent=agent, save_session=AsyncMock()), "gpt-5")

    assert compressor._model == "gpt-4o-mini"
    assert compressor._llm == "separate_llm"


async def test_no_arg_shows_current_model() -> None:
    agent = MagicMock()
    agent.llm.model_name = "claude-4"
    ctx = MagicMock(agent=agent)
    result = await CMD.handler(ctx, "")
    assert "claude-4" in render_output(result.output)


async def test_switch_model_preserves_inference_params() -> None:
    from agent_harness.core.config import LLMConfig

    agent = MagicMock()
    agent.llm.model_name = "gpt-4o"
    agent.context.config.llm = LLMConfig(
        provider="openai",
        model="gpt-4o",
        temperature=0.3,
        max_tokens=8000,
        reasoning_effort="high",
    )
    agent.context.short_term_memory.compressor = None
    agent.context.config.memory.compression.summary_model = None

    with patch("agent_cli.commands.builtin.model.create_llm") as create:
        create.return_value = MagicMock()
        await CMD.handler(MagicMock(agent=agent, save_session=AsyncMock()), "gpt-5")

    create.assert_called_once()
    (cfg_arg,), _ = create.call_args
    assert cfg_arg.model == "gpt-5"
    assert cfg_arg.temperature == 0.3
    assert cfg_arg.max_tokens == 8000
    assert cfg_arg.reasoning_effort == "high"
