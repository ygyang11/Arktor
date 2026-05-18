"""Tests for OpenAIProvider._parse_usage cache-semantics normalization."""
from __future__ import annotations

from types import SimpleNamespace

from agent_harness.llm.openai_provider import OpenAIProvider


def _make_provider() -> OpenAIProvider:
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.config = SimpleNamespace(model="m", base_url="http://x", temperature=0)
    provider._client = None
    provider._rate_limiter = None
    provider._additive_semantics = False
    return provider


def _usage(prompt: int, cached: int, completion: int, total: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total if total is not None else prompt + completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
    )


class TestInclusiveSemanticsUnchanged:
    def test_openai_standard_cached_subset_passes_through(self) -> None:
        p = _make_provider()
        u = p._parse_usage(_usage(prompt=1000, cached=600, completion=50))
        assert u.prompt_tokens == 1000
        assert u.completion_tokens == 50
        assert u.total_tokens == 1050
        assert u.cache_read_tokens == 600
        assert p._additive_semantics is False

    def test_zero_cached_no_detection(self) -> None:
        p = _make_provider()
        u = p._parse_usage(_usage(prompt=500, cached=0, completion=20))
        assert u.prompt_tokens == 500
        assert p._additive_semantics is False

    def test_cached_equal_prompt_no_detection(self) -> None:
        p = _make_provider()
        u = p._parse_usage(_usage(prompt=400, cached=400, completion=10))
        assert u.prompt_tokens == 400
        assert p._additive_semantics is False


class TestAdditiveSemanticsDetection:
    def test_cached_exceeds_prompt_triggers_normalization(self) -> None:
        p = _make_provider()
        u = p._parse_usage(_usage(prompt=40, cached=10752, completion=205))
        assert p._additive_semantics is True
        assert u.prompt_tokens == 40 + 10752
        assert u.total_tokens == 40 + 10752 + 205
        assert u.cache_read_tokens == 10752

    def test_sticky_applies_to_subsequent_cache_miss_heavy_call(self) -> None:
        p = _make_provider()
        p._parse_usage(_usage(prompt=40, cached=10752, completion=205))
        assert p._additive_semantics is True
        u = p._parse_usage(_usage(prompt=5119, cached=11104, completion=293))
        assert u.prompt_tokens == 5119 + 11104
        assert u.total_tokens == 5119 + 11104 + 293
        assert u.cache_read_tokens == 11104

    def test_sticky_persists_after_detection(self) -> None:
        p = _make_provider()
        p._additive_semantics = True
        u = p._parse_usage(_usage(prompt=100, cached=50, completion=10))
        assert u.prompt_tokens == 150
        assert u.total_tokens == 160

    def test_empty_usage_returns_default(self) -> None:
        p = _make_provider()
        u = p._parse_usage(None)
        assert u.prompt_tokens == 0
        assert u.cache_read_tokens == 0
        assert p._additive_semantics is False
