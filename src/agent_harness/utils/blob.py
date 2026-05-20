"""Content-addressed blob store for media payloads."""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path

from agent_harness.core.message import Attachment

_BLOB_DIR = Path.home() / ".agent-harness" / "blobs"

_BASE64_DATA_URI_RE = re.compile(
    r"data:[\w.+-]+/[\w.+-]+;base64,[A-Za-z0-9+/=]{40,}"
)


def _path(digest: str) -> Path:
    return _BLOB_DIR / digest


def put_bytes(data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    dest = _path(digest)
    if dest.exists():
        return digest
    _BLOB_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=_BLOB_DIR, suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        tmp_path.write_bytes(data)
        tmp_path.replace(dest)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return digest


def get_bytes(digest: str) -> bytes:
    return _path(digest).read_bytes()


def exists(digest: str) -> bool:
    return _path(digest).exists()


def make_attachment(
    data: bytes, mime: str, filename: str | None = None,
) -> Attachment:
    return Attachment(
        digest=put_bytes(data),
        mime=mime,
        size=len(data),
        filename=filename,
    )


def scrub_base64(message: str) -> str:
    """Replace long base64 data URIs in a string with a short marker."""
    return _BASE64_DATA_URI_RE.sub("data:<base64 elided>", message)
