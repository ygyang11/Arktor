"""Bracketed-paste placeholder store, regex, highlight processor, clipboard image."""
from __future__ import annotations

import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
from base64 import b64decode
from dataclasses import dataclass, field
from pathlib import Path

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout.processors import (
    Processor,
    Transformation,
    TransformationInput,
)

from agent_harness.core.message import Attachment
from agent_harness.utils.blob import make_attachment
from agent_harness.utils.media import is_media_mime

CHAR_THRESHOLD = 700
LINE_THRESHOLD = 4
_UNAVAILABLE = "[Pasted text unavailable]"
_PLACEHOLDER_RE = re.compile(
    r"\[Pasted (?:text|Image|File) #(\d+)(?: \+\d+ lines)?\]"
)
_TRAILING_PLACEHOLDER_RE = re.compile(
    r"\[Pasted (?:text|Image|File) #(\d+)(?: \+\d+ lines)?\]\Z",
)
_STYLE_CLASS = "class:paste-placeholder"


@dataclass
class PasteStore:
    """REPL-process-scoped paste registry: id -> original text or Attachment."""

    _contents: dict[int, str | Attachment] = field(default_factory=dict)
    _next_id: int = 1

    def register(self, text: str) -> str | None:
        if not text:
            return None
        line_count = text.count("\n") + (0 if text.endswith("\n") else 1)
        if len(text) <= CHAR_THRESHOLD and line_count < LINE_THRESHOLD:
            return None

        pid = self._next_id
        self._next_id += 1
        self._contents[pid] = text
        nl = text.count("\n")
        return (
            f"[Pasted text #{pid} +{nl} lines]" if nl else f"[Pasted text #{pid}]"
        )

    def register_media(self, att: Attachment) -> str:
        pid = self._next_id
        self._next_id += 1
        self._contents[pid] = att
        kind = "File" if att.mime == "application/pdf" else "Image"
        return f"[Pasted {kind} #{pid}]"

    def resolve(self, text: str) -> tuple[str, list[int], list[Attachment]]:
        matches = list(_PLACEHOLDER_RE.finditer(text))
        missing: list[int] = []
        attachments: list[Attachment] = []
        seen: set[int] = set()
        for m in matches:
            pid = int(m.group(1))
            if pid not in self._contents and pid not in seen:
                seen.add(pid)
                missing.append(pid)
        for m in reversed(matches):
            pid = int(m.group(1))
            item = self._contents.get(pid)
            if item is None:
                text = text[: m.start()] + _UNAVAILABLE + text[m.end():]
            elif isinstance(item, Attachment):
                attachments.insert(0, item)
            else:
                text = text[: m.start()] + item + text[m.end():]
        return text, missing, attachments


class PastePlaceholderProcessor(Processor):
    """Highlight [Pasted text|Image|File #N ...] tokens with class:paste-placeholder."""

    def apply_transformation(
        self, transformation_input: TransformationInput,
    ) -> Transformation:
        try:
            fragments: StyleAndTextTuples = list(transformation_input.fragments)
            plain = "".join(text for _, text, *_ in fragments)
            spans = [
                (m.start(), m.end()) for m in _PLACEHOLDER_RE.finditer(plain)
            ]
            if not spans:
                return Transformation(fragments)
            return Transformation(_restyle_fragments(fragments, spans))
        except Exception:
            try:
                return Transformation(list(transformation_input.fragments))
            except Exception:
                return Transformation([])


def _restyle_fragments(
    fragments: StyleAndTextTuples,
    spans: list[tuple[int, int]],
) -> StyleAndTextTuples:
    out: StyleAndTextTuples = []
    pos = 0
    span_iter = iter(spans)
    cur = next(span_iter, None)

    for fragment in fragments:
        style, text = fragment[0], fragment[1]
        if not text:
            out.append(fragment)
            continue
        for ch in text:
            in_span = cur is not None and cur[0] <= pos < cur[1]
            target_style = f"{style} {_STYLE_CLASS}".strip() if in_span else style
            if out and out[-1][0] == target_style and len(out[-1]) == 2:
                prev_style, prev_text = out[-1][0], out[-1][1]
                out[-1] = (prev_style, prev_text + ch)
            else:
                out.append((target_style, ch))
            pos += 1
            if cur is not None and pos >= cur[1]:
                cur = next(span_iter, None)
    return out


def trailing_placeholder_length(text_before_cursor: str) -> int | None:
    m = _TRAILING_PLACEHOLDER_RE.search(text_before_cursor)
    if not m:
        return None
    return len(m.group(0))


def path_to_attachment(text: str) -> Attachment | None:
    text = text.strip().strip('"').strip("'")
    if not text or "\n" in text:
        return None
    try:
        p = Path(text)
        if not p.is_file():
            return None
        mime, _ = mimetypes.guess_type(p.name)
        if not mime or not is_media_mime(mime):
            return None
        data = p.read_bytes()
    except OSError:
        return None
    return make_attachment(data, mime, p.name)


_OSASCRIPT_WRITE_PNG = (
    'set f to (POSIX file "{path}")\n'
    "set ph to (the clipboard as «class PNGf»)\n"
    "set fh to open for access f with write permission\n"
    "write ph to fh\nclose access fh"
)

_POWERSHELL_GET_IMAGE_PNG_B64 = (
    "Add-Type -AssemblyName System.Windows.Forms; "
    "$img = [System.Windows.Forms.Clipboard]::GetImage(); "
    "if ($img) { "
    "$ms = New-Object System.IO.MemoryStream; "
    "$img.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png); "
    "[System.Convert]::ToBase64String($ms.ToArray()) "
    "}"
)


def read_clipboard_image() -> tuple[str, bytes] | None:
    if sys.platform == "darwin":
        return _macos_clipboard_image()
    if sys.platform == "win32":
        return _windows_clipboard_image()
    if sys.platform.startswith("linux"):
        return _linux_clipboard_image() or _windows_clipboard_image()
    return None


def _macos_clipboard_image() -> tuple[str, bytes] | None:
    if shutil.which("pngpaste"):
        r = subprocess.run(["pngpaste", "-"], capture_output=True)
        if r.returncode == 0 and r.stdout:
            return "image/png", r.stdout
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        r = subprocess.run(
            ["osascript", "-e", _OSASCRIPT_WRITE_PNG.format(path=tmp)],
            capture_output=True,
        )
        if r.returncode != 0:
            return None
        data = Path(tmp).read_bytes()
        return ("image/png", data) if data else None
    finally:
        Path(tmp).unlink(missing_ok=True)


def _linux_clipboard_image() -> tuple[str, bytes] | None:
    for cmd in (
        ["wl-paste", "-t", "image/png"],
        ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"],
    ):
        if not shutil.which(cmd[0]):
            continue
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0 and r.stdout:
            return "image/png", r.stdout
    return None


def _windows_clipboard_image() -> tuple[str, bytes] | None:
    if not shutil.which("powershell.exe"):
        return None
    r = subprocess.run(
        ["powershell.exe", "-NonInteractive", "-NoProfile", "-command",
         _POWERSHELL_GET_IMAGE_PNG_B64],
        capture_output=True,
    )
    if r.returncode != 0:
        return None
    b64 = r.stdout.decode("utf-8", errors="replace").strip()
    if not b64:
        return None
    try:
        return "image/png", b64decode(b64)
    except (ValueError, TypeError):
        return None
