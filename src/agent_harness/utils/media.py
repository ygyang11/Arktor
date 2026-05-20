"""Media MIME classification — image/PDF derived from mime, no kind enum."""
from __future__ import annotations

_IMAGE_MIMES = frozenset(
    {"image/png", "image/jpeg", "image/gif", "image/webp"}
)
_PDF = "application/pdf"


def is_image_mime(mime: str) -> bool:
    return mime in _IMAGE_MIMES


def is_pdf_mime(mime: str) -> bool:
    return mime == _PDF


def is_media_mime(mime: str) -> bool:
    return is_image_mime(mime) or is_pdf_mime(mime)
