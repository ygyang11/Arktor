from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_cli.commands.builtin.provider import CMD, _active_profile
from agent_cli.commands.ui import _PROFILE_EXAMPLE
from agent_harness.core.config import LLMConfig, SubModelConfig

from ..conftest import render_output


def _agent(
    provider: str = "openai",
    model: str = "gpt-4o",
    base_url: str | None = None,
    *,
    sub_model: str | None = None,
    sub_effort: str | None = None,
) -> MagicMock:
    agent = MagicMock()
    agent.llm = MagicMock(model_name=model)
    agent.sub_llm = (
        MagicMock(model_name=sub_model)
        if sub_model is not None
        else agent.llm
    )
    agent.context.config.llm = LLMConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key="x",
        temperature=0.3,
        max_tokens=8000,
        timeout=200.0,
        max_retries=7,
        retry_delay=2.0,
        reasoning_effort="high",
        sub_model=(
            SubModelConfig(model=sub_model, reasoning_effort=sub_effort)
            if sub_model is not None
            else None
        ),
    )
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


async def test_switch_profile_builds_main_and_sub_then_commits() -> None:
    agent = _agent(sub_model="old-sub", sub_effort="xhigh")
    old_config = agent.context.config.llm
    prefs = {"llm_profiles": {
        "codex": {
            "provider": "openai",
            "model": "gpt-5.4",
            "sub_model": "gpt-5.4-mini",
            "base_url": "https://x/v1",
            "api_key": "sk-x",
        },
    }}
    new_main = MagicMock(model_name="gpt-5.4")
    new_sub = MagicMock(model_name="gpt-5.4-mini")

    with (
        patch("agent_cli.commands.builtin.provider.read_prefs", return_value=prefs),
        patch(
            "agent_cli.commands.builtin.provider.create_llm",
            return_value=new_main,
        ) as create_main,
        patch(
            "agent_cli.commands.builtin.provider.create_sub_llm",
            return_value=new_sub,
        ) as create_sub,
    ):
        result = await CMD.handler(MagicMock(agent=agent), "codex")

    (main_cfg,), _ = create_main.call_args
    (sub_cfg,), _ = create_sub.call_args
    assert main_cfg is sub_cfg
    assert main_cfg.provider == "openai"
    assert main_cfg.model == "gpt-5.4"
    assert main_cfg.base_url == "https://x/v1"
    assert main_cfg.api_key == "sk-x"
    assert main_cfg.sub_model == SubModelConfig(
        model="gpt-5.4-mini",
        reasoning_effort="xhigh",
    )
    assert agent.context.config.llm is main_cfg
    assert agent.context.config.llm is not old_config
    agent.replace_llms.assert_called_once_with(new_main, new_sub)
    assert "Switched to codex" in render_output(result.output)


async def test_profile_without_sub_aliases_main_and_drops_old_sub() -> None:
    agent = _agent(sub_model="old-sub", sub_effort="xhigh")
    prefs = {"llm_profiles": {
        "codex": {"provider": "openai", "model": "gpt-5.4"},
    }}
    new_main = MagicMock(model_name="gpt-5.4")

    with (
        patch("agent_cli.commands.builtin.provider.read_prefs", return_value=prefs),
        patch("agent_cli.commands.builtin.provider.create_llm", return_value=new_main),
        patch(
            "agent_cli.commands.builtin.provider.create_sub_llm",
            return_value=None,
        ),
    ):
        await CMD.handler(MagicMock(agent=agent), "codex")

    assert agent.context.config.llm.sub_model is None
    agent.replace_llms.assert_called_once_with(new_main, new_main)


@pytest.mark.parametrize("profile_sub", ["", "   "])
async def test_empty_profile_sub_aliases_main_despite_current_effort(
    profile_sub: str,
) -> None:
    agent = _agent(sub_model="old-sub", sub_effort="xhigh")
    prefs = {"llm_profiles": {
        "codex": {
            "provider": "openai",
            "model": "gpt-5.4",
            "sub_model": profile_sub,
        },
    }}
    new_main = MagicMock(model_name="gpt-5.4")

    with (
        patch("agent_cli.commands.builtin.provider.read_prefs", return_value=prefs),
        patch("agent_cli.commands.builtin.provider.create_llm", return_value=new_main),
        patch(
            "agent_cli.commands.builtin.provider.create_sub_llm",
            return_value=None,
        ),
    ):
        result = await CMD.handler(MagicMock(agent=agent), "codex")

    assert "Switched to codex" in render_output(result.output)
    assert agent.context.config.llm.sub_model is None
    agent.replace_llms.assert_called_once_with(new_main, new_main)


async def test_switch_inherits_runtime_params_from_session() -> None:
    agent = _agent()
    prefs = {"llm_profiles": {
        "codex": {
            "provider": "openai",
            "model": "gpt-5.4",
            "base_url": "https://x/v1",
            "api_key": "sk-x",
        },
    }}

    with (
        patch("agent_cli.commands.builtin.provider.read_prefs", return_value=prefs),
        patch("agent_cli.commands.builtin.provider.create_llm") as create,
        patch("agent_cli.commands.builtin.provider.create_sub_llm", return_value=None),
    ):
        create.return_value = MagicMock()
        await CMD.handler(MagicMock(agent=agent), "codex")

    (cfg,), _ = create.call_args
    assert cfg.temperature == 0.3
    assert cfg.max_tokens == 8000
    assert cfg.timeout == 200.0
    assert cfg.max_retries == 7
    assert cfg.retry_delay == 2.0
    assert cfg.reasoning_effort == "high"


async def test_profile_overrides_inherited_runtime_param() -> None:
    agent = _agent()
    prefs = {"llm_profiles": {
        "cold": {
            "provider": "anthropic",
            "model": "claude-opus-4-8",
            "base_url": "https://y",
            "api_key": "ak-y",
            "temperature": 1.0,
        },
    }}

    with (
        patch("agent_cli.commands.builtin.provider.read_prefs", return_value=prefs),
        patch("agent_cli.commands.builtin.provider.create_llm") as create,
        patch("agent_cli.commands.builtin.provider.create_sub_llm", return_value=None),
    ):
        create.return_value = MagicMock()
        await CMD.handler(MagicMock(agent=agent), "cold")

    (cfg,), _ = create.call_args
    assert cfg.temperature == 1.0
    assert cfg.reasoning_effort == "high"


async def test_omitted_key_falls_back_to_env_not_old_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    agent = _agent()
    prefs = {"llm_profiles": {
        "codex": {
            "provider": "openai",
            "model": "gpt-5.4",
            "base_url": "https://x/v1",
        },
    }}

    with (
        patch("agent_cli.commands.builtin.provider.read_prefs", return_value=prefs),
        patch("agent_cli.commands.builtin.provider.create_llm") as create,
        patch("agent_cli.commands.builtin.provider.create_sub_llm", return_value=None),
    ):
        create.return_value = MagicMock()
        await CMD.handler(MagicMock(agent=agent), "codex")

    (cfg,), _ = create.call_args
    assert cfg.api_key == "env-key"


async def test_main_construction_failure_commits_nothing() -> None:
    agent = _agent(sub_model="old-sub", sub_effort="low")
    old_config = agent.context.config.llm
    prefs = {"llm_profiles": {
        "broken": {"provider": "openai", "model": "gpt-5.4"},
    }}

    with (
        patch("agent_cli.commands.builtin.provider.read_prefs", return_value=prefs),
        patch(
            "agent_cli.commands.builtin.provider.create_llm",
            side_effect=ValueError("no client"),
        ),
        patch("agent_cli.commands.builtin.provider.create_sub_llm") as create_sub,
    ):
        result = await CMD.handler(MagicMock(agent=agent), "broken")

    assert agent.context.config.llm is old_config
    agent.replace_llms.assert_not_called()
    create_sub.assert_not_called()
    assert "Failed to switch provider" in render_output(result.output)


async def test_sub_construction_failure_closes_new_main_and_commits_nothing() -> None:
    agent = _agent(sub_model="old-sub", sub_effort="low")
    old_config = agent.context.config.llm
    new_main = MagicMock()
    new_main.aclose = AsyncMock()
    prefs = {"llm_profiles": {
        "broken": {
            "provider": "openai",
            "model": "gpt-5.4",
            "sub_model": "gpt-5.4-mini",
        },
    }}

    with (
        patch("agent_cli.commands.builtin.provider.read_prefs", return_value=prefs),
        patch(
            "agent_cli.commands.builtin.provider.create_llm",
            return_value=new_main,
        ),
        patch(
            "agent_cli.commands.builtin.provider.create_sub_llm",
            side_effect=ValueError("no sub client"),
        ),
    ):
        result = await CMD.handler(MagicMock(agent=agent), "broken")

    new_main.aclose.assert_awaited_once()
    assert agent.context.config.llm is old_config
    agent.replace_llms.assert_not_called()
    assert "Failed to switch provider" in render_output(result.output)


async def test_unknown_profile_lists_available() -> None:
    ctx = MagicMock(agent=_agent())
    prefs = {"llm_profiles": {"codex": {"provider": "openai", "model": "gpt-5.4"}}}
    with patch("agent_cli.commands.builtin.provider.read_prefs", return_value=prefs):
        result = await CMD.handler(ctx, "nope")
    rendered = render_output(result.output)
    assert "Unknown provider: nope" in rendered
    assert "codex" in rendered


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
    ctx = MagicMock(agent=_agent())
    prefs = {"llm_profiles": {
        "codex": {"provider": "openai", "mdel": "gpt-5.4"},
    }}
    with (
        patch("agent_cli.commands.builtin.provider.read_prefs", return_value=prefs),
        patch("agent_cli.commands.builtin.provider.create_llm") as create,
    ):
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
        provider="openai",
        model="gpt-9-other",
        base_url="https://x/v1",
        api_key="x",
    )
    assert _active_profile(profiles, drifted) == "codex"
    elsewhere = LLMConfig(
        provider="openai",
        model="gpt-5.4",
        base_url="https://z",
        api_key="x",
    )
    assert _active_profile(profiles, elsewhere) is None
    no_url_profiles = {"a": {"provider": "openai", "model": "x"}}
    no_url_current = LLMConfig(
        provider="openai",
        model="x",
        base_url=None,
        api_key="x",
    )
    assert _active_profile(no_url_profiles, no_url_current) is None


def test_profile_example_uses_flat_sub_model() -> None:
    assert '"sub_model": ".."' in _PROFILE_EXAMPLE
    assert "reasoning_effort" not in _PROFILE_EXAMPLE


def test_provider_command_metadata() -> None:
    assert CMD.name == "/provider"
    assert "provider" in CMD.description.lower()
