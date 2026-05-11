"""Token counting utilities for agent_harness.

Wraps tiktoken for local token estimation. NOT the source of authoritative
token totals — those come from provider post-call usage (see Usage type and
ShortTermMemory.last_call). This module is used as:
  - proportion estimator for /context section breakdown (calibrated to
    provider truth before display)
  - fallback when no LLM call has happened yet (e.g. compressor.should_compress
    on first call before any snapshot exists)
  - trace truncation in tracing.py
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from agent_harness.core.message import Message

logger = logging.getLogger(__name__)


class _Encoding(Protocol):
    """Protocol for tokenizer encode/decode methods."""

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, tokens: list[int]) -> str:
        ...

# Model to encoding mapping
_MODEL_ENCODING_MAP: dict[str, str] = {
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-5": "o200k_base",
    "gpt-5-mini": "o200k_base",
    # Anthropic models: use cl100k_base as approximation
    "claude-3-opus": "cl100k_base",
    "claude-3-sonnet": "cl100k_base",
    "claude-3-haiku": "cl100k_base",
    "claude-3.5-sonnet": "cl100k_base",
    "claude-4-sonnet": "cl100k_base",
    "claude-4": "cl100k_base",
    "claude-4.5": "cl100k_base",
    "claude-5": "cl100k_base",
    "claude-6": "cl100k_base",
}

# Prefix-based fallback: if exact match fails, try prefix
_PREFIX_ENCODING_MAP: list[tuple[str, str]] = [
    ("gpt-5", "o200k_base"),
    ("gpt-4o", "o200k_base"),
    ("gpt-4", "cl100k_base"),
    ("gpt-3.5", "cl100k_base"),
    ("claude-", "cl100k_base"),
]

DEFAULT_ENCODING = "cl100k_base"


def _resolve_encoding_name(model: str) -> str:
    """Resolve tokenizer encoding name for a model."""
    encoding_name = _MODEL_ENCODING_MAP.get(model)
    if encoding_name is not None:
        return encoding_name

    for prefix, enc in _PREFIX_ENCODING_MAP:
        if model.startswith(prefix):
            return enc
    return DEFAULT_ENCODING


@lru_cache(maxsize=8)
def _get_encoding(encoding_name: str) -> _Encoding | None:
    """Get a tiktoken encoding, cached."""
    try:
        import tiktoken
        return tiktoken.get_encoding(encoding_name)
    except ImportError:
        logger.warning("tiktoken not installed, token counting will use approximation")
        return None


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in a text string for a given model.
    
    Falls back to word-based approximation if tiktoken is unavailable.
    """
    encoding_name = _resolve_encoding_name(model)
    encoding = _get_encoding(encoding_name)

    if encoding is not None:
        return len(encoding.encode(text))

    # Fallback: rough approximation (1 token ≈ 4 chars for English, 2 chars for CJK)
    return len(text) // 3


def _truncate_by_approx_chars(text: str, max_tokens: int, suffix: str) -> str:
    """Fallback truncation when tokenizer is unavailable."""
    approx_max_chars = max_tokens * 3
    if len(text) <= approx_max_chars:
        return text

    if not suffix:
        return text[:approx_max_chars]

    if approx_max_chars <= len(suffix):
        return suffix[:approx_max_chars]

    return text[: approx_max_chars - len(suffix)] + suffix


def truncate_text_by_tokens(
    text: str,
    max_tokens: int,
    model: str = "gpt-4o",
    suffix: str = "...",
) -> str:
    """Truncate text to a maximum token budget.
    
    Appends ``suffix`` only when truncation happens.
    """
    if max_tokens <= 0:
        return ""

    encoding = _get_encoding(_resolve_encoding_name(model))
    if encoding is None:
        return _truncate_by_approx_chars(text, max_tokens, suffix)

    token_ids: list[int] = encoding.encode(text)
    if len(token_ids) <= max_tokens:
        return text

    if not suffix:
        return encoding.decode(token_ids[:max_tokens])

    suffix_tokens: list[int] = encoding.encode(suffix)
    if len(suffix_tokens) >= max_tokens:
        return encoding.decode(token_ids[:max_tokens])

    kept = token_ids[: max_tokens - len(suffix_tokens)]
    return encoding.decode(kept) + suffix


def count_messages_tokens(messages: list[Message], model: str = "gpt-4o") -> int:
    """Count total tokens for a list of messages.

    Accounts for message structure overhead (role, separators).
    """
    total = 0
    tokens_per_message = 4  # overhead per message in ChatML format

    for msg in messages:
        total += tokens_per_message
        if msg.content:
            total += count_tokens(msg.content, model)
        if msg.tool_calls:
            for tc in msg.tool_calls:
                total += count_tokens(tc.name, model)
                total += count_tokens(str(tc.arguments), model)
        if msg.tool_result:
            total += count_tokens(msg.tool_result.content, model)
        if msg.provider_metadata:
            for ns_fields in msg.provider_metadata.values():
                for v in ns_fields.values():
                    if isinstance(v, str):
                        total += count_tokens(v, model)
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                for fval in item.values():
                                    if isinstance(fval, str):
                                        total += count_tokens(fval, model)

    total += 3  # assistant reply priming
    return total
