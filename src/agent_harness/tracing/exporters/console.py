"""Console exporter for trace spans.

Prints spans to the terminal with indentation and optional color.
Useful for development and debugging.
"""
from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import TextIO

from agent_harness.tracing.tracer import Span
from agent_harness.utils.token_counter import truncate_text_by_tokens

_CONSOLE_ATTR_MAX_TOKENS = 100

# ANSI color codes
_COLORS = {
    "agent": "\033[1;34m",         # Bold Blue
    "llm": "\033[1;32m",           # Bold Green
    "tool": "\033[1;33m",          # Bold Yellow
    "memory": "\033[1;35m",        # Bold Magenta
    "orchestration": "\033[1;36m", # Bold Cyan
    "subagent": "\033[1;36m",      # Bold Cyan (same as orchestration)
    "internal": "\033[0;37m",      # Gray
    "error": "\033[1;31m",         # Bold Red
    "reset": "\033[0m",
}


class ConsoleExporter:
    """Export spans to the console with colored, indented output."""

    def __init__(self, stream: TextIO = sys.stderr, color: bool = True) -> None:
        self._stream = stream
        self._color = color and hasattr(stream, "isatty") and stream.isatty()

    def write_inline(self, text: str) -> None:
        """Write text inline without newline (for streaming output)."""
        self._stream.write(text)
        self._stream.flush()

    def export(self, spans: list[Span]) -> None:
        """Export a list of spans to the console."""
        depth_map = self._build_depth_map(spans)
        for span in spans:
            self._print_span(span, indent=depth_map.get(span.span_id, 0))

    def export_one(self, span: Span, *, indent: int) -> None:
        """Export a single span with an explicit indent."""
        self._print_span(span, indent=indent)

    def print_start(
        self,
        *,
        kind: str,
        name: str,
        indent: int = 0,
        attributes: Mapping[str, object] | None = None,
    ) -> None:
        """Print an immediate start marker for an in-flight span."""
        prefix = "  " * indent
        kind_color = _COLORS.get(kind, _COLORS["internal"]) if self._color else ""
        reset = _COLORS["reset"] if self._color else ""

        self._stream.write(f"{prefix}▶ {kind_color}[{kind}]{reset} {name} (start)\n")
        if attributes:
            for key, value in attributes.items():
                self._stream.write(f"{prefix}  {key}: {self._format_attr_value(value)}\n")

    def _build_depth_map(self, spans: list[Span]) -> dict[str, int]:
        """Build a map of span_id -> depth based on parent_span_id relationships."""
        depth_map: dict[str, int] = {}
        span_lookup = {s.span_id: s for s in spans}

        def get_depth(span_id: str) -> int:
            if span_id in depth_map:
                return depth_map[span_id]
            span = span_lookup.get(span_id)
            if span is None or span.parent_span_id is None:
                depth_map[span_id] = 0
                return 0
            parent_depth = get_depth(span.parent_span_id)
            depth_map[span_id] = parent_depth + 1
            return parent_depth + 1

        for span in spans:
            get_depth(span.span_id)
        return depth_map

    def _print_span(self, span: Span, indent: int = 0) -> None:
        prefix = "  " * indent
        kind_color = _COLORS.get(span.kind, _COLORS["internal"]) if self._color else ""
        reset = _COLORS["reset"] if self._color else ""
        error_color = _COLORS["error"] if self._color else ""

        duration = f"{span.duration_ms:.1f}ms" if span.duration_ms is not None else "..."
        status_indicator = "✓" if span.status == "ok" else f"{error_color}✗{reset}"

        line = (
            f"{prefix}{status_indicator} {kind_color}[{span.kind}]{reset} "
            f"{span.name} ({duration})"
        )
        self._stream.write(line + "\n")

        if span.attributes:
            for key, value in span.attributes.items():
                self._stream.write(f"{prefix}  {key}: {self._format_attr_value(value)}\n")

        if span.error_message:
            self._stream.write(f"{prefix}  {error_color}error: {span.error_message}{reset}\n")

        for event in span.events:
            self._stream.write(f"{prefix}  • {event.name}")
            if event.attributes:
                self._stream.write(f" {self._format_event_attributes(event.attributes)}")
            self._stream.write("\n")

    def _format_event_attributes(self, attributes: Mapping[str, object]) -> str:
        pairs: list[str] = []
        for key, value in attributes.items():
            pairs.append(f"{key}={self._format_attr_value(value)}")
        return "{" + ", ".join(pairs) + "}"

    @staticmethod
    def _format_attr_value(value: object) -> str:
        text = str(value)
        return truncate_text_by_tokens(
            text,
            max_tokens=_CONSOLE_ATTR_MAX_TOKENS,
            suffix="...",
        )
