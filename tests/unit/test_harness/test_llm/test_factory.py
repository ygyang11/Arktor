"""Tests for LLM factory entrypoints."""
from __future__ import annotations

import pytest

from agent_harness.core.config import HarnessConfig, LLMConfig
from agent_harness.llm import LLM, create_llm, create_sub_llm
from agent_harness.llm.anthropic_provider import AnthropicProvider
from agent_harness.llm.openai_provider import OpenAIProvider


class TestCreateLLM:
    def test_create_llm_openai_from_harness_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        cfg = HarnessConfig(llm=LLMConfig(provider="openai", model="gpt-4o"))

        llm = create_llm(cfg)

        assert isinstance(llm, OpenAIProvider)
        assert llm.config.model == "gpt-4o"

    def test_create_llm_anthropic_from_harness_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-test")
        cfg = HarnessConfig(llm=LLMConfig(provider="anthropic", model="claude-3-5-sonnet-latest"))

        llm = create_llm(cfg)

        assert isinstance(llm, AnthropicProvider)
        assert llm.config.model == "claude-3-5-sonnet-latest"

    def test_create_llm_unsupported_provider_raises(self) -> None:
        cfg = HarnessConfig(llm=LLMConfig(provider="mock", model="mock-model"))

        with pytest.raises(ValueError, match="Unsupported llm.provider"):
            create_llm(cfg)

    def test_create_sub_llm_returns_none_when_unconfigured(self) -> None:
        cfg = HarnessConfig(llm=LLMConfig(model="gpt-main", api_key="sk-test"))
        assert create_sub_llm(cfg) is None

    def test_create_sub_llm_uses_effective_sub_config(self) -> None:
        cfg = HarnessConfig(
            llm=LLMConfig(
                provider="openai",
                model="gpt-main",
                temperature=0.3,
                max_tokens=8192,
                api_key="sk-test",
                reasoning_effort="high",
                sub_model={"model": "gpt-sub", "reasoning_effort": "low"},
            )
        )

        llm = create_sub_llm(cfg)

        assert isinstance(llm, OpenAIProvider)
        assert llm.config.model == "gpt-sub"
        assert llm.config.reasoning_effort == "low"
        assert llm.config.temperature == 0.3
        assert llm.config.max_tokens == 8192
        assert llm.config.sub_model is None


class TestLLMFacade:
    def test_llm_facade_matches_create_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        cfg = HarnessConfig(llm=LLMConfig(provider="openai", model="gpt-4o"))

        llm = LLM(cfg)

        assert isinstance(llm, OpenAIProvider)
