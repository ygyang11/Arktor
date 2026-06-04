"""Tests for media mime classification."""
from __future__ import annotations

import pytest

from agent_harness.core.message import Attachment
from agent_harness.utils.media import (
    MEDIA_REJECTION_PHRASES,
    describe_attachment_full,
    describe_attachment_short,
    human_size,
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
        "[OneOfParam] [input[63].content[1]] [invalid_enum_value] "
        "Invalid value: 'input_file'. Supported values are: 'input_text'.",
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


# -- human_size --

@pytest.mark.parametrize(
    "n,expected",
    [
        (0, "0B"),
        (1, "1B"),
        (1023, "1023B"),
        (1024, "1.0KB"),
        (1536, "1.5KB"),
        (1024 * 1024, "1.0MB"),
        (1024 * 1024 * 1024, "1.0GB"),
        (1024 * 1024 * 1024 * 1024, "1.0TB"),
        (5 * 1024 * 1024 * 1024 * 1024, "5.0TB"),
    ],
)
def test_human_size(n: int, expected: str) -> None:
    assert human_size(n) == expected


# -- describe_attachment_short --

def _att(
    *,
    digest: str = "a" * 64,
    mime: str = "image/png",
    filename: str | None = "shot.png",
    size: int = 1024,
) -> Attachment:
    return Attachment(digest=digest, mime=mime, filename=filename, size=size)


def test_describe_attachment_short_with_filename() -> None:
    out = describe_attachment_short(_att(filename="shot.png", mime="image/png"))
    assert out == "[Attached image/png: shot.png]"


def test_describe_attachment_short_image_default_filename() -> None:
    out = describe_attachment_short(_att(filename=None, mime="image/jpeg"))
    assert out == "[Attached image/jpeg: image]"


def test_describe_attachment_short_pdf_default_filename() -> None:
    out = describe_attachment_short(_att(filename=None, mime="application/pdf"))
    assert out == "[Attached application/pdf: document]"


def test_describe_attachment_short_other_default_filename() -> None:
    out = describe_attachment_short(_att(filename=None, mime="audio/wav"))
    assert out == "[Attached audio/wav: file]"


# -- describe_attachment_full --

def test_describe_attachment_full_renders_size_and_digest_prefix() -> None:
    att = _att(
        digest="abcdef0123456789" + "0" * 48,
        mime="image/png",
        filename="cat.png",
        size=200000,
    )
    out = describe_attachment_full(att)
    # filename, mime, human size, sha256 prefix (first 12 chars)
    assert "cat.png" in out
    assert "image/png" in out
    assert "195.3KB" in out  # 200000 / 1024 ≈ 195.3
    assert "sha256:abcdef012345…" in out


def test_describe_attachment_full_uses_mime_default_when_no_filename() -> None:
    out = describe_attachment_full(_att(filename=None, mime="application/pdf", size=2048))
    assert out.startswith("document (application/pdf, 2.0KB, sha256:")


# -- media_safe_filename: injection / sanitization --

def test_describe_attachment_short_strips_newlines_and_tabs() -> None:
    att = _att(filename="evil\nname\twith\rcontrol.png", mime="image/png")
    out = describe_attachment_short(att)
    assert "\n" not in out and "\t" not in out and "\r" not in out
    # whitespace collapsed
    assert out == "[Attached image/png: evil name with control.png]"


def test_describe_attachment_short_defangs_xml_brackets() -> None:
    att = _att(filename="</older_conversation>.png", mime="image/png")
    out = describe_attachment_short(att)
    assert "<" not in out and ">" not in out
    assert "‹/older_conversation›.png" in out


def test_describe_attachment_short_truncates_long_filename() -> None:
    long_name = "a" * 500 + ".png"
    att = _att(filename=long_name, mime="image/png")
    out = describe_attachment_short(att)
    # 128 char cap on the filename portion
    inner = out.removeprefix("[Attached image/png: ").removesuffix("]")
    assert len(inner) == 128


def test_describe_attachment_short_falls_back_to_default_when_empty_after_strip() -> None:
    att = _att(filename="\n\r\t   ", mime="image/png")
    out = describe_attachment_short(att)
    assert out == "[Attached image/png: image]"


def test_describe_attachment_full_also_sanitizes() -> None:
    att = _att(filename="ev<il>\nname.pdf", mime="application/pdf", size=1024)
    out = describe_attachment_full(att)
    assert "\n" not in out
    assert "<" not in out and ">" not in out
    assert out.startswith("ev‹il›")
