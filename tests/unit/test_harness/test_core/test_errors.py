"""Tests for error scrubbing of base64 payloads."""
from __future__ import annotations

from agent_harness.core.errors import (
    LLMContextLengthError,
    LLMError,
    LLMRateLimitError,
)
from agent_harness.utils.blob import scrub_base64


def test_scrub_strips_image_data_uri() -> None:
    raw = "Bad request: data:image/png;base64," + "A" * 200 + " end"
    assert scrub_base64(raw) == "Bad request: data:<base64 elided> end"


def test_scrub_strips_pdf_data_uri() -> None:
    raw = "data:application/pdf;base64," + "Z" * 100
    assert scrub_base64(raw) == "data:<base64 elided>"


def test_scrub_strips_multiple_payloads() -> None:
    raw = (
        "two: data:image/png;base64," + "A" * 50
        + " then data:application/pdf;base64," + "B" * 60 + " done"
    )
    out = scrub_base64(raw)
    assert "data:image/png" not in out
    assert "data:application/pdf" not in out
    assert out.count("<base64 elided>") == 2


def test_scrub_short_base64_left_alone() -> None:
    raw = "short: data:image/png;base64,ABC"
    assert scrub_base64(raw) == raw


def test_scrub_non_data_uri_untouched() -> None:
    raw = "regular error: 400 Bad Request for /v1/messages"
    assert scrub_base64(raw) == raw


def test_llm_error_scrubs_on_construction() -> None:
    e = LLMError("Extra inputs: data:application/pdf;base64," + "X" * 80)
    assert "data:<base64 elided>" in str(e)
    assert "XXXXXX" not in str(e)


def test_llm_subclass_inherits_scrubbing() -> None:
    e = LLMContextLengthError("payload data:image/png;base64," + "Q" * 80)
    assert "<base64 elided>" in str(e)


def test_llm_rate_limit_preserves_kwarg() -> None:
    e = LLMRateLimitError(
        "rate data:image/png;base64," + "R" * 80, retry_after=2.5,
    )
    assert "<base64 elided>" in str(e)
    assert e.retry_after == 2.5
