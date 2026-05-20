"""Tests for media mime classification."""
from __future__ import annotations

import pytest

from agent_harness.utils.media import is_image_mime, is_media_mime, is_pdf_mime


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
