"""LLM module: model provider abstractions."""
from agent_harness.core.config import (
    HarnessConfig,
    LLMConfig,
    resolve_llm_config,
    resolve_sub_llm_config,
)
from agent_harness.llm.anthropic_provider import AnthropicProvider
from agent_harness.llm.base import BaseLLM, FallbackChain, RateLimiter
from agent_harness.llm.openai_provider import OpenAIProvider
from agent_harness.llm.types import FinishReason, LLMResponse, StreamDelta, Usage

_PROVIDER_MAP: dict[str, type[BaseLLM]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def create_llm(
    config: HarnessConfig | LLMConfig | None = None,
) -> BaseLLM:
    llm_cfg = resolve_llm_config(config)
    provider = _PROVIDER_MAP.get(llm_cfg.provider.lower())
    if provider is None:
        supported = ", ".join(sorted(_PROVIDER_MAP))
        raise ValueError(
            f"Unsupported llm.provider: {llm_cfg.provider!r}. Supported: {supported}"
        )
    return provider(llm_cfg)


def create_sub_llm(
    config: HarnessConfig | LLMConfig | None = None,
) -> BaseLLM | None:
    sub_cfg = resolve_sub_llm_config(config)
    return create_llm(sub_cfg) if sub_cfg is not None else None


def LLM(config: HarnessConfig | LLMConfig | None = None) -> BaseLLM:
    return create_llm(config)


__all__ = [
    "BaseLLM",
    "LLMResponse", "Usage", "FinishReason", "StreamDelta",
    "OpenAIProvider", "AnthropicProvider",
    "FallbackChain", "RateLimiter",
    "create_llm", "create_sub_llm", "LLM",
]
