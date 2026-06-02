from unittest.mock import MagicMock, patch

import pytest

from agent_cli.commands.builtin.provider import CMD, _active_profile
from agent_harness.core.config import LLMConfig

from ..conftest import render_output


def _agent(provider: str = "openai", model: str = "gpt-4o", base_url: str | None = None) -> MagicMock:
    agent = MagicMock()
    agent.context.config.llm = LLMConfig(
        provider=provider, model=model, base_url=base_url, api_key="x",
    )
    agent.context.short_term_memory.compressor = MagicMock()
    agent.context.config.memory.compression.summary_model = None
    return agent


async def test_no_arg_empty_shows_guidance() -> None:
    ctx = MagicMock(agent=_agent())
    with patch("agent_cli.commands.builtin.provider.read_prefs", return_value={}):
        result = await CMD.handler(ctx, "")
    rendered = render_output(result.output)
    assert "No provider profiles yet" in rendered
    assert "llm_profiles" in rendered
    assert "cli-prefs.json" in rendered


async def test_no_arg_lists_profiles_and_marks_active() -> None:
    ctx = MagicMock(agent=_agent("openai", "gpt-5.4", "https://x/v1"))
    prefs = {"llm_profiles": {
        "codex": {"provider": "openai", "model": "gpt-5.4", "base_url": "https://x/v1"},
        "claude": {"provider": "anthropic", "model": "claude-opus-4-8", "base_url": "https://y"},
    }}
    with patch("agent_cli.commands.builtin.provider.read_prefs", return_value=prefs):
        result = await CMD.handler(ctx, "")
    rendered = render_output(result.output)
    assert "codex" in rendered
    assert "claude" in rendered
    assert "▌" in rendered


async def test_switch_profile_swaps_all_sites_from_fresh_config() -> None:
    agent = _agent()
    compressor = agent.context.short_term_memory.compressor
    ctx = MagicMock(agent=agent)
    prefs = {"llm_profiles": {
        "codex": {"provider": "openai", "model": "gpt-5.4",
                  "base_url": "https://x/v1", "api_key": "sk-x"},
    }}
    with patch("agent_cli.commands.builtin.provider.read_prefs", return_value=prefs), \
         patch("agent_cli.commands.builtin.provider.create_llm") as create:
        new_llm = MagicMock()
        create.return_value = new_llm
        result = await CMD.handler(ctx, "codex")

    assert agent.llm is new_llm
    assert agent.context.short_term_memory.model == "gpt-5.4"
    assert compressor._model == "gpt-5.4"
    assert compressor._llm is new_llm
    assert agent.context.config.llm.provider == "openai"
    assert agent.context.config.llm.base_url == "https://x/v1"
    (cfg_arg,), _ = create.call_args
    assert cfg_arg.provider == "openai"
    assert cfg_arg.base_url == "https://x/v1"
    assert cfg_arg.api_key == "sk-x"
    assert "Switched to codex" in render_output(result.output)


def _rich_agent() -> MagicMock:
    agent = MagicMock()
    agent.context.config.llm = LLMConfig(
        provider="openai", model="gpt-4o", api_key="old",
        temperature=0.3, max_tokens=8000, timeout=200.0,
        max_retries=7, retry_delay=2.0, reasoning_effort="high",
    )
    agent.context.short_term_memory.compressor = MagicMock()
    agent.context.config.memory.compression.summary_model = None
    return agent


async def test_switch_inherits_runtime_params_from_session() -> None:
    ctx = MagicMock(agent=_rich_agent())
    prefs = {"llm_profiles": {
        "codex": {"provider": "openai", "model": "gpt-5.4",
                  "base_url": "https://x/v1", "api_key": "sk-x"},
    }}
    with patch("agent_cli.commands.builtin.provider.read_prefs", return_value=prefs), \
         patch("agent_cli.commands.builtin.provider.create_llm") as create:
        create.return_value = MagicMock()
        await CMD.handler(ctx, "codex")
    (cfg,), _ = create.call_args
    assert cfg.temperature == 0.3
    assert cfg.max_tokens == 8000
    assert cfg.timeout == 200.0
    assert cfg.max_retries == 7
    assert cfg.retry_delay == 2.0
    assert cfg.reasoning_effort == "high"
    assert cfg.provider == "openai" and cfg.model == "gpt-5.4"
    assert cfg.base_url == "https://x/v1" and cfg.api_key == "sk-x"


async def test_profile_overrides_inherited_runtime_param() -> None:
    ctx = MagicMock(agent=_rich_agent())
    prefs = {"llm_profiles": {
        "cold": {"provider": "anthropic", "model": "claude-opus-4-8",
                 "base_url": "https://y", "api_key": "ak-y", "temperature": 1.0},
    }}
    with patch("agent_cli.commands.builtin.provider.read_prefs", return_value=prefs), \
         patch("agent_cli.commands.builtin.provider.create_llm") as create:
        create.return_value = MagicMock()
        await CMD.handler(ctx, "cold")
    (cfg,), _ = create.call_args
    assert cfg.temperature == 1.0
    assert cfg.reasoning_effort == "high"


async def test_omitted_key_falls_back_to_env_not_old_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    ctx = MagicMock(agent=_rich_agent())
    prefs = {"llm_profiles": {
        "codex": {"provider": "openai", "model": "gpt-5.4", "base_url": "https://x/v1"},
    }}
    with patch("agent_cli.commands.builtin.provider.read_prefs", return_value=prefs), \
         patch("agent_cli.commands.builtin.provider.create_llm") as create:
        create.return_value = MagicMock()
        await CMD.handler(ctx, "codex")
    (cfg,), _ = create.call_args
    assert cfg.api_key == "env-key"


async def test_unknown_profile_lists_available() -> None:
    ctx = MagicMock(agent=_agent())
    prefs = {"llm_profiles": {"codex": {"provider": "openai", "model": "gpt-5.4"}}}
    with patch("agent_cli.commands.builtin.provider.read_prefs", return_value=prefs):
        result = await CMD.handler(ctx, "nope")
    rendered = render_output(result.output)
    assert "Unknown provider: nope" in rendered
    assert "codex" in rendered


async def test_bad_profile_reports_error() -> None:
    agent = _agent()
    ctx = MagicMock(agent=agent)
    prefs = {"llm_profiles": {"broken": {"provider": "openai", "model": "gpt-5.4"}}}
    with patch("agent_cli.commands.builtin.provider.read_prefs", return_value=prefs), \
         patch("agent_cli.commands.builtin.provider.create_llm",
               side_effect=ValueError("no client")):
        result = await CMD.handler(ctx, "broken")
    assert agent.llm is not None
    assert "Failed to switch provider" in render_output(result.output)


async def test_malformed_entry_is_ignored_not_crash() -> None:
    ctx = MagicMock(agent=_agent("openai", "gpt-5.4", "https://x/v1"))
    prefs = {"llm_profiles": {
        "good": {"provider": "openai", "model": "gpt-5.4", "base_url": "https://x/v1"},
        "bad": "oops",
    }}
    with patch("agent_cli.commands.builtin.provider.read_prefs", return_value=prefs):
        result = await CMD.handler(ctx, "")
    rendered = render_output(result.output)
    assert "good" in rendered
    assert "bad" not in rendered


async def test_unknown_field_blocks_switch() -> None:
    ctx = MagicMock(agent=_rich_agent())
    prefs = {"llm_profiles": {
        "codex": {"provider": "openai", "mdel": "gpt-5.4", "base_url": "https://x/v1"},
    }}
    with patch("agent_cli.commands.builtin.provider.read_prefs", return_value=prefs), \
         patch("agent_cli.commands.builtin.provider.create_llm") as create:
        result = await CMD.handler(ctx, "codex")
    create.assert_not_called()
    rendered = render_output(result.output)
    assert "unknown field" in rendered
    assert "mdel" in rendered


def test_active_profile_matches_on_base_url_only() -> None:
    profiles = {
        "codex": {"provider": "openai", "model": "gpt-5.4", "base_url": "https://x/v1"},
        "claude": {"provider": "anthropic", "model": "claude-opus", "base_url": "https://y"},
    }
    drifted = LLMConfig(
        provider="openai", model="gpt-9-other", base_url="https://x/v1", api_key="x",
    )
    assert _active_profile(profiles, drifted) == "codex"
    elsewhere = LLMConfig(
        provider="openai", model="gpt-5.4", base_url="https://z", api_key="x",
    )
    assert _active_profile(profiles, elsewhere) is None
    no_url_profiles = {"a": {"provider": "openai", "model": "x"}}
    no_url_current = LLMConfig(provider="openai", model="x", base_url=None, api_key="x")
    assert _active_profile(no_url_profiles, no_url_current) is None


def test_provider_command_metadata() -> None:
    assert CMD.name == "/provider"
    assert "provider" in CMD.description.lower()
