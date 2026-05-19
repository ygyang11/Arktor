"""Bracketed-paste placeholder store, regex, highlight processor."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout.processors import (
    Processor,
    Transformation,
    TransformationInput,
)

CHAR_THRESHOLD = 700
LINE_THRESHOLD = 4
_UNAVAILABLE = "[Pasted text unavailable]"
_PLACEHOLDER_RE = re.compile(r"\[Pasted text #(\d+)(?: \+\d+ lines)?\]")
_TRAILING_PLACEHOLDER_RE = re.compile(
    r"\[Pasted text #(\d+)(?: \+\d+ lines)?\]\Z",
)
_STYLE_CLASS = "class:paste-placeholder"


@dataclass
class PasteStore:
    """REPL-process-scoped paste registry: id -> original text."""

    _contents: dict[int, str] = field(default_factory=dict)
    _next_id: int = 1

    def register(self, text: str) -> str | None:
        """Return placeholder if text exceeds threshold, else None."""
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

    def expand(self, text: str) -> tuple[str, list[int]]:
        """Splice originals back; rewrite unknown ids to _UNAVAILABLE."""
        matches = list(_PLACEHOLDER_RE.finditer(text))
        missing: list[int] = []
        seen: set[int] = set()
        for m in matches:
            pid = int(m.group(1))
            if pid not in self._contents and pid not in seen:
                seen.add(pid)
                missing.append(pid)
        for m in reversed(matches):
            pid = int(m.group(1))
            replacement = self._contents.get(pid, _UNAVAILABLE)
            text = text[: m.start()] + replacement + text[m.end() :]
        return text, missing


class PastePlaceholderProcessor(Processor):
    """Highlight [Pasted text #N ...] tokens with class:paste-placeholder."""

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


def trailing_placeholder_length(text_before_cursor: str) -> int | None:
    """Length of trailing paste placeholder in text, or None if not present."""
    m = _TRAILING_PLACEHOLDER_RE.search(text_before_cursor)
    if not m:
        return None
    return len(m.group(0))


def _restyle_fragments(
    fragments: StyleAndTextTuples,
    spans: list[tuple[int, int]],
) -> StyleAndTextTuples:
    """Walk fragments char-by-char; chars inside spans get _STYLE_CLASS."""
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
            # Layer (don't replace): preserves built-in selection / search
            # styling that prompt_toolkit applies before user processors.
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
