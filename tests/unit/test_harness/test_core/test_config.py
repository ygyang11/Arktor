"""Tests for agent_harness.core.config — HarnessConfig defaults, from_env, merge."""
from __future__ import annotations

import pytest

from agent_harness.core.config import (
    ApprovalConfig,
    HarnessConfig,
    LLMConfig,
    MemoryConfig,
    PaperConfig,
    SandboxConfig,
    SearchConfig,
    ToolConfig,
    TracingConfig,
    resolve_llm_config,
    resolve_paper_config,
    resolve_search_config,
    resolve_tool_config,
)


class TestLLMConfig:
    def test_defaults(self) -> None:
        cfg = LLMConfig()
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 4096
        assert cfg.timeout == 120.0
        assert cfg.base_url is None
        assert cfg.max_retries == 3
        assert cfg.retry_delay == 1.0

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        cfg = LLMConfig()
        assert cfg.api_key == "sk-test"

    def test_api_key_explicit_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        cfg = LLMConfig(api_key="sk-explicit")
        assert cfg.api_key == "sk-explicit"

    def test_anthropic_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-test")
        cfg = LLMConfig(provider="anthropic")
        assert cfg.api_key == "ak-test"

    def test_blank_fields_fall_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("ARKTOR_LLM_BASE_URL", "https://example.com/v1")
        cfg = LLMConfig(api_key="", base_url="")
        assert cfg.api_key == "sk-test"
        assert cfg.base_url == "https://example.com/v1"


class TestToolConfig:
    def test_defaults(self) -> None:
        cfg = ToolConfig()
        assert cfg.max_concurrency == 5
        assert cfg.default_timeout == 30.0
        assert cfg.sandbox.enabled is False


class TestSandboxConfig:
    def test_defaults(self) -> None:
        cfg = SandboxConfig()
        assert cfg.enabled is False
        assert cfg.docker.image == "python:3.11-slim"
        assert cfg.docker.network == "none"
        assert cfg.docker.setup == ""
        assert cfg.docker.setup_timeout == 300
        assert cfg.docker.volumes == []

    def test_nested_in_tool_config(self) -> None:
        cfg = ToolConfig()
        assert isinstance(cfg.sandbox, SandboxConfig)
        assert cfg.sandbox.enabled is False

    def test_from_dict(self) -> None:
        data = {
            "enabled": True,
            "docker": {
                "image": "ubuntu:22.04",
                "network": "bridge",
                "memory": "1g",
                "cpus": 2.0,
                "volumes": ["/data:/data:ro"],
            },
        }
        cfg = SandboxConfig.model_validate(data)
        assert cfg.enabled is True
        assert cfg.docker.image == "ubuntu:22.04"
        assert cfg.docker.cpus == 2.0
        assert cfg.docker.volumes == ["/data:/data:ro"]


class TestMemoryConfig:
    def test_defaults(self) -> None:
        cfg = MemoryConfig()
        assert cfg.max_tokens == 100000
        assert cfg.strategy == "summarize"
        assert cfg.forget_threshold == 0.3
        assert cfg.compression.threshold == 0.75
        assert cfg.compression.retain_count == 6


class TestTracingConfig:
    def test_defaults(self) -> None:
        cfg = TracingConfig()
        assert cfg.enabled is True
        assert cfg.exporter == "both"
        assert cfg.export_path == "./traces"


class TestHarnessConfig:
    def test_defaults(self) -> None:
        cfg = HarnessConfig()
        assert cfg.verbose is False
        assert cfg.llm.provider == "openai"
        assert cfg.tool.max_concurrency == 5
        assert cfg.memory.max_tokens == 100000
        assert cfg.tracing.enabled is True

    def test_from_env_llm_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARKTOR_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ARKTOR_LLM_MODEL", "claude-3")
        monkeypatch.setenv("ARKTOR_LLM_TEMPERATURE", "0.5")
        monkeypatch.setenv("ARKTOR_LLM_MAX_TOKENS", "2048")

        cfg = HarnessConfig.from_env()
        assert cfg.llm.provider == "anthropic"
        assert cfg.llm.model == "claude-3"
        assert cfg.llm.temperature == 0.5
        assert cfg.llm.max_tokens == 2048

    def test_from_env_verbose(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARKTOR_VERBOSE", "true")
        cfg = HarnessConfig.from_env()
        assert cfg.verbose is True

    def test_from_env_verbose_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARKTOR_VERBOSE", "no")
        cfg = HarnessConfig.from_env()
        assert cfg.verbose is False

    def test_from_env_tracing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARKTOR_TRACING_ENABLED", "0")
        cfg = HarnessConfig.from_env()
        assert cfg.tracing.enabled is False

    def test_from_env_no_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARKTOR_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("ARKTOR_LLM_MODEL", raising=False)
        monkeypatch.delenv("ARKTOR_LLM_TEMPERATURE", raising=False)
        monkeypatch.delenv("ARKTOR_LLM_MAX_TOKENS", raising=False)
        monkeypatch.delenv("ARKTOR_VERBOSE", raising=False)
        monkeypatch.delenv("ARKTOR_TRACING_ENABLED", raising=False)
        cfg = HarnessConfig.from_env()
        assert cfg.llm.provider == "openai"

    def test_merge_other_overrides(self) -> None:
        base = HarnessConfig()
        other = HarnessConfig(llm=LLMConfig(model="gpt-3.5"), verbose=True)
        merged = base.merge(other)
        assert merged.llm.model == "gpt-3.5"
        assert merged.verbose is True
        assert merged.llm.provider == "openai"

    def test_merge_preserves_base_when_other_default(self) -> None:
        base = HarnessConfig(llm=LLMConfig(temperature=0.2))
        other = HarnessConfig()
        merged = base.merge(other)
        assert merged.llm.temperature == 0.2

    def test_merge_deep_nested(self) -> None:
        base = HarnessConfig(memory=MemoryConfig(max_tokens=200000))
        other = HarnessConfig(memory=MemoryConfig(forget_threshold=0.5))
        merged = base.merge(other)
        assert merged.memory.max_tokens == 200000
        assert merged.memory.forget_threshold == 0.5

    def test_from_yaml_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            HarnessConfig.from_yaml("/nonexistent/arktor.yaml")

    def test_search_config_in_harness(self) -> None:
        cfg = HarnessConfig()
        assert cfg.search.provider == "tavily"
        assert cfg.search.tavily_api_key is None or isinstance(cfg.search.tavily_api_key, str)

    def test_load_merges_yaml_and_env(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "arktor.yaml"
        path.write_text(
            "llm:\n"
            "  provider: openai\n"
            "  model: yaml-model\n"
            "tracing:\n"
            "  enabled: false\n"
        )
        monkeypatch.setenv("ARKTOR_LLM_MODEL", "env-model")

        cfg = HarnessConfig.load(path, env_override=True)

        assert cfg.llm.model == "env-model"
        assert cfg.tracing.enabled is False
        assert HarnessConfig.get() is cfg

    def test_load_without_env_override_keeps_yaml(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "arktor.yaml"
        path.write_text(
            "llm:\n"
            "  provider: openai\n"
            "  model: yaml-model\n"
        )
        monkeypatch.setenv("ARKTOR_LLM_MODEL", "env-model")

        cfg = HarnessConfig.load(path, env_override=False)

        assert cfg.llm.model == "yaml-model"
        assert HarnessConfig.get() is cfg


class TestSearchConfig:
    def test_defaults(self) -> None:
        cfg = SearchConfig()
        assert cfg.provider == "tavily"

    def test_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        monkeypatch.setenv("SERPAPI_API_KEY", "serp-test")
        cfg = SearchConfig()
        assert cfg.tavily_api_key == "tvly-test"
        assert cfg.serpapi_api_key == "serp-test"

    def test_blank_fields_fall_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        monkeypatch.setenv("SERPAPI_API_KEY", "serp-test")
        cfg = SearchConfig(tavily_api_key="", serpapi_api_key="")
        assert cfg.tavily_api_key == "tvly-test"
        assert cfg.serpapi_api_key == "serp-test"


class TestResolveConfigHelpers:
    def test_resolve_llm_config(self) -> None:
        cfg = HarnessConfig(llm=LLMConfig(model="m1"))
        assert resolve_llm_config(cfg) is cfg.llm
        assert resolve_llm_config(cfg.llm) is cfg.llm

    def test_resolve_tool_config(self) -> None:
        cfg = HarnessConfig(tool=ToolConfig(max_concurrency=9))
        assert resolve_tool_config(cfg) is cfg.tool
        assert resolve_tool_config(cfg.tool) is cfg.tool

    def test_resolve_search_config(self) -> None:
        cfg = HarnessConfig(search=SearchConfig(provider="serpapi"))
        assert resolve_search_config(cfg) is cfg.search
        assert resolve_search_config(cfg.search) is cfg.search

    def test_resolve_paper_config(self) -> None:
        cfg = HarnessConfig(paper=PaperConfig(semantic_scholar_api_key="k"))
        assert resolve_paper_config(cfg) is cfg.paper
        assert resolve_paper_config(cfg.paper) is cfg.paper


class TestPaperConfig:
    def test_defaults(self) -> None:
        cfg = PaperConfig()
        assert cfg.semantic_scholar_api_key is None

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-key")
        cfg = PaperConfig()
        assert cfg.semantic_scholar_api_key == "test-key"

    def test_blank_to_none(self) -> None:
        cfg = PaperConfig(semantic_scholar_api_key="  ")
        assert cfg.semantic_scholar_api_key is None

    def test_placeholder_to_none_allows_env_fallback(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "env-key")
        cfg = PaperConfig(semantic_scholar_api_key="__SET_API_KEY__")
        assert cfg.semantic_scholar_api_key == "env-key"

    def test_real_value_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "env-key")
        cfg = PaperConfig(semantic_scholar_api_key="real-key")
        assert cfg.semantic_scholar_api_key == "real-key"

    def test_in_harness(self) -> None:
        cfg = HarnessConfig()
        assert isinstance(cfg.paper, PaperConfig)


class TestApprovalConfigValidation:
    def test_valid_tool_level(self) -> None:
        cfg = ApprovalConfig(always_allow=["read_file", "list_dir"])
        assert len(cfg.always_allow) == 2

    def test_valid_resource_level(self) -> None:
        cfg = ApprovalConfig(
            always_allow=["terminal_tool(git *)", "web_fetch(domain:github.com)"],
            always_deny=["write_file(**/.env*)"],
        )
        assert len(cfg.always_allow) == 2
        assert len(cfg.always_deny) == 1

    def test_invalid_rule_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            ApprovalConfig(always_allow=["not valid!"])

    def test_empty_rule_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty"):
            ApprovalConfig(always_deny=[""])
