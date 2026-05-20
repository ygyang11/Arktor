"""Media MIME classification — image/PDF derived from mime, no kind enum."""
from __future__ import annotations

_IMAGE_MIMES = frozenset(
    {"image/png", "image/jpeg", "image/gif", "image/webp"}
)
_PDF = "application/pdf"

MEDIA_REJECTION_PHRASES: tuple[str, ...] = (
    "invalid part type",
    "unknown variant `image_url`",
    "unknown variant `image`",
    "unknown variant `file`",
    "does not support image",
    "does not support file",
    "does not support video",
    "does not support audio",
    "does not support multimodal",
    "does not support vision",
    "file content types",
    "could not process image",
    "could not process file",
    "image does not match",
    "media type mismatch",
    "unsupported mimetype",
    "unsupported image",
    "image dimensions exceed",
    "expected file type",
    "inlinedata parameter",
)

def is_image_mime(mime: str) -> bool:
    return mime in _IMAGE_MIMES


def is_pdf_mime(mime: str) -> bool:
    return mime == _PDF


def is_media_mime(mime: str) -> bool:
    return is_image_mime(mime) or is_pdf_mime(mime)


def is_media_rejection(err_msg: str) -> bool:
    """Substring-match an LLM provider error message against known
    media-rejection phrases."""
    low = err_msg.lower()
    return any(p in low for p in MEDIA_REJECTION_PHRASES)
