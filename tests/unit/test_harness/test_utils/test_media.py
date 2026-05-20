"""Tests for media mime classification."""
from __future__ import annotations

import pytest

from agent_harness.utils.media import (
    MEDIA_REJECTION_PHRASES,
    is_image_mime,
    is_media_mime,
    is_media_rejection,
    is_pdf_mime,
)


@pytest.mark.parametrize(
    "mime",
    ["image/png", "image/jpeg", "image/gif", "image/webp"],
)
def test_is_image_mime_whitelist_hits(mime: str) -> None:
    assert is_image_mime(mime) is True


@pytest.mark.parametrize(
    "mime",
    [
        "image/svg+xml",
        "image/heic",
        "image/tiff",
        "image/bmp",
        "image/avif",
        "image/vnd.fastbidsheet",
        "application/pdf",
        "text/plain",
        "",
    ],
)
def test_is_image_mime_non_whitelist_misses(mime: str) -> None:
    assert is_image_mime(mime) is False


def test_is_pdf_mime_exact_match() -> None:
    assert is_pdf_mime("application/pdf") is True


@pytest.mark.parametrize(
    "mime",
    ["application/x-pdf", "PDF", "image/png", "", "application/pdf "],
)
def test_is_pdf_mime_strict_rejects(mime: str) -> None:
    assert is_pdf_mime(mime) is False


def test_is_media_mime_union() -> None:
    assert is_media_mime("image/png") is True
    assert is_media_mime("application/pdf") is True
    assert is_media_mime("image/heic") is False
    assert is_media_mime("text/plain") is False


# -- is_media_rejection: real provider samples --

@pytest.mark.parametrize(
    "err",
    [
        # kimi / moonshot
        "Invalid request: the message at position 5 with role 'user' "
        "contains an invalid part type: file",
        # deepseek (via openai-compat)
        "Failed to deserialize the JSON body into the target type: "
        "messages[3]: unknown variant `image_url`, expected `text`",
        "messages[3]: unknown variant `image`, expected `text`",
        "messages[3]: unknown variant `file`, expected `text`",
        # minimax
        "Error from provider: This model does not support image inputs",
        # azure openai
        "Invalid Value: 'file'. This model does not support file content types.",
        # anthropic
        "Could not process image",
        "Image does not match the provided media type",
        "Image media type mismatch with base64 data",
        "Image dimensions exceed max allowed size (8000 pixels)",
        # openai
        "Invalid file 'image': unsupported mimetype ('application/octet-stream')",
        "Expected file type to be a supported format: .jpeg, .jpg, .png, .gif, .webp",
        # gemini
        "Unable to submit request because it has an empty inlineData parameter",
        # capitalization stress
        "INVALID PART TYPE: file",
        "DOES NOT SUPPORT IMAGE",
    ],
)
def test_is_media_rejection_real_provider_samples(err: str) -> None:
    assert is_media_rejection(err) is True


@pytest.mark.parametrize(
    "err",
    [
        "Rate limit exceeded",
        "Context length exceeded: 200000 tokens > 128000 max",
        "401 Unauthorized",
        "500 Internal Server Error",
        # Phrase-like but non-media context
        "Maximum context length is 16384 tokens",
        "Invalid argument: top_p must be between 0 and 1",
        "",
    ],
)
def test_is_media_rejection_non_media_negatives(err: str) -> None:
    assert is_media_rejection(err) is False


def test_phrase_list_is_lowercase() -> None:
    """All phrases must already be lowercase since matcher lowercases input."""
    for p in MEDIA_REJECTION_PHRASES:
        assert p == p.lower()
