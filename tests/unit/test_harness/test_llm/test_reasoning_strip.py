"""Sticky-gated reasoning_details strip/flatten for strict-input relays."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import openai
import pytest

from agent_harness.core.errors import LLMContextLengthError, LLMError
from agent_harness.llm.openai_provider import (
    OpenAIProvider,
    _is_reasoning_details_rejection,
    _normalize_reasoning_for_strict_input,
)


class _FakeBadRequest(openai.BadRequestError):
    def __init__(self, msg: str) -> None:
        Exception.__init__(self, msg)


def _provider() -> OpenAIProvider:
    p = OpenAIProvider.__new__(OpenAIProvider)
    p._additive_semantics = False
    p._strip_reasoning_details = False
    return p


_REJECTION = (
    "Error code: 400 - Extra inputs are not permitted, field: "
    "'messages[4].reasoning_details', value: [{'type':'reasoning.text'}]"
)


class TestRejectionDetection:
    def test_rejection_matches(self) -> None:
        assert _is_reasoning_details_rejection(
            _FakeBadRequest(_REJECTION)
        )

    def test_context_length_not_a_rejection(self) -> None:
        assert not _is_reasoning_details_rejection(
            _FakeBadRequest("maximum context length exceeded")
        )

    def test_unrelated_400_not_a_rejection(self) -> None:
        assert not _is_reasoning_details_rejection(
            _FakeBadRequest("some other 400")
        )


class TestNormalizeReasoning:
    def test_flatten_text_blocks_when_content_absent(self) -> None:
        req: dict[str, Any] = {"messages": [{
            "role": "assistant",
            "reasoning_details": [
                {"type": "reasoning.text", "text": "step 1. "},
                {"type": "reasoning.text", "text": "step 2."},
            ],
        }]}
        _normalize_reasoning_for_strict_input(req)
        m = req["messages"][0]
        assert "reasoning_details" not in m
        assert m["reasoning_content"] == "step 1. step 2."

    def test_existing_reasoning_content_kept_details_dropped(self) -> None:
        req: dict[str, Any] = {"messages": [{
            "role": "assistant",
            "reasoning_content": "real thinking",
            "reasoning_details": [{"type": "reasoning.text", "text": "x"}],
        }]}
        _normalize_reasoning_for_strict_input(req)
        m = req["messages"][0]
        assert m["reasoning_content"] == "real thinking"
        assert "reasoning_details" not in m

    def test_summary_fallback_and_encrypted_skipped(self) -> None:
        req: dict[str, Any] = {"messages": [{
            "role": "assistant",
            "reasoning_details": [
                {"type": "reasoning.summary", "summary": "S"},
                {"type": "reasoning.encrypted", "data": "ZW5j"},
                "not-a-dict",
            ],
        }]}
        _normalize_reasoning_for_strict_input(req)
        m = req["messages"][0]
        assert m["reasoning_content"] == "S"
        assert "reasoning_details" not in m

    def test_no_reasoning_details_untouched(self) -> None:
        req: dict[str, Any] = {"messages": [{"role": "user", "content": "hi"}]}
        _normalize_reasoning_for_strict_input(req)
        assert req["messages"][0] == {"role": "user", "content": "hi"}


@pytest.mark.asyncio
class TestGenerateStickyRetry:
    def _wire(self, p: OpenAIProvider, side: Any) -> list[dict[str, Any]]:
        seen: list[dict[str, Any]] = []

        async def fake_create(**req: Any) -> Any:
            seen.append({k: v for k, v in req.items()})
            return side(req)

        p._build_request = lambda *a, **k: {  # type: ignore[method-assign]
            "messages": [{
                "role": "assistant",
                "reasoning_details": [{"type": "reasoning.text", "text": "T"}],
            }],
        }
        p._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
        )
        p._parse_response = lambda r: r  # type: ignore[method-assign]
        return seen

    async def test_matching_rejection_flips_sticky_and_retries(self) -> None:
        p = _provider()
        calls = {"n": 0}

        def side(_req: dict[str, Any]) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                raise _FakeBadRequest(_REJECTION)
            return "OK"

        seen = self._wire(p, side)
        out = await p.generate([])

        assert out == "OK"
        assert p._strip_reasoning_details is True
        assert calls["n"] == 2
        retried = seen[1]["messages"][0]
        assert "reasoning_details" not in retried
        assert retried["reasoning_content"] == "T"

    async def test_sticky_prestrips_on_subsequent_calls(self) -> None:
        p = _provider()
        p._strip_reasoning_details = True
        seen = self._wire(p, lambda _req: "OK")

        await p.generate([])

        assert len(seen) == 1  # no retry
        assert "reasoning_details" not in seen[0]["messages"][0]
        assert seen[0]["messages"][0]["reasoning_content"] == "T"

    async def test_context_length_400_raised_not_sticky(self) -> None:
        p = _provider()

        def side(_req: dict[str, Any]) -> Any:
            raise _FakeBadRequest("maximum context length exceeded")

        self._wire(p, side)
        with pytest.raises(LLMContextLengthError):
            await p.generate([])
        assert p._strip_reasoning_details is False

    async def test_unrelated_400_maps_to_llmerror_not_sticky(self) -> None:
        p = _provider()

        def side(_req: dict[str, Any]) -> Any:
            raise _FakeBadRequest("some other 400")

        self._wire(p, side)
        with pytest.raises(LLMError):
            await p.generate([])
        assert p._strip_reasoning_details is False


def test_normalize_backfills_empty_string_reasoning_content() -> None:
    # sticky path intentionally treats "" as needing backfill — Moonshot
    # treats an empty reasoning_content as missing.
    req: dict[str, Any] = {"messages": [{
        "role": "assistant",
        "reasoning_content": "",
        "reasoning_details": [{"type": "reasoning.text", "text": "deep"}],
    }]}
    _normalize_reasoning_for_strict_input(req)
    m = req["messages"][0]
    assert m["reasoning_content"] == "deep"
    assert "reasoning_details" not in m


@pytest.mark.asyncio
async def test_flag_flipped_by_concurrent_request_still_retries() -> None:
    # Reproduces the shared-provider race: this request built its payload
    # before the sticky flag flipped (no pre-strip), then a concurrent
    # request flips the instance flag while this one's create is in flight.
    # The per-call retry guard must still let this request normalize+retry.
    p = _provider()
    calls = {"n": 0}

    async def fake_create(**req: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            p._strip_reasoning_details = True  # concurrent peer flipped it
            raise _FakeBadRequest(_REJECTION)
        return "OK"

    p._build_request = lambda *a, **k: {  # type: ignore[method-assign]
        "messages": [{
            "role": "assistant",
            "reasoning_details": [{"type": "reasoning.text", "text": "T"}],
        }],
    }
    p._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    p._parse_response = lambda r: r  # type: ignore[method-assign]

    out = await p.generate([])

    assert out == "OK"
    assert calls["n"] == 2  # retried despite instance flag already True


class _StreamIter:
    def __init__(self) -> None:
        self._done = False

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration


@pytest.mark.asyncio
async def test_stream_matching_rejection_flips_sticky_and_retries() -> None:
    p = _provider()
    seen: list[dict[str, Any]] = []

    async def fake_create(**req: Any) -> Any:
        snap = {k: v for k, v in req.items()}
        seen.append(snap)
        if any("reasoning_details" in m for m in snap["messages"]):
            raise _FakeBadRequest(_REJECTION)
        return _StreamIter()

    p._build_request = lambda *a, **k: {  # type: ignore[method-assign]
        "messages": [{
            "role": "assistant",
            "reasoning_details": [{"type": "reasoning.text", "text": "T"}],
        }],
    }
    p._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )

    async for _ in p.stream([]):
        pass

    assert p._strip_reasoning_details is True
    assert len(seen) == 2
    assert "reasoning_details" not in seen[1]["messages"][0]
    assert seen[1]["messages"][0]["reasoning_content"] == "T"
