"""Media MIME classification — image/PDF derived from mime, no kind enum."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_harness.core.message import Attachment

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
    """Substring-match an LLM provider error message against known media-rejection phrases."""
    low = err_msg.lower()
    return any(p in low for p in MEDIA_REJECTION_PHRASES)


def media_safe_filename(name: str | None, mime: str) -> str:
    """Defang ``Attachment.filename`` before embedding into prompts/markdown."""
    default = (
        "image" if is_image_mime(mime)
        else "document" if is_pdf_mime(mime)
        else "file"
    )
    if not name:
        return default
    cleaned = "".join(c if c.isprintable() else " " for c in name)
    cleaned = " ".join(cleaned.split())
    cleaned = cleaned.replace("<", "‹").replace(">", "›")[:128]
    return cleaned or default


def describe_attachment_short(att: Attachment) -> str:
    """Terse, LLM-friendly."""
    return f"[Attached {att.mime}: {media_safe_filename(att.filename, att.mime)}]"


def describe_attachment_full(att: Attachment) -> str:
    """Detailed, human-facing"""
    name = media_safe_filename(att.filename, att.mime)
    return f"{name} ({att.mime}, {human_size(att.size)}, sha256:{att.digest[:12]}…)"


def human_size(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024:
            return f"{int(f)}B" if unit == "B" else f"{f:.1f}{unit}"
        f /= 1024
    return f"{f:.1f}TB"