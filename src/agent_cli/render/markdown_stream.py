"""MarkdownStream — split-stream pattern"""
from __future__ import annotations

import io
import time

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from agent_cli.theme import TOOL_DONE, CliTheme

_LIVE_WINDOW = 6
_MIN_DELAY_FLOOR = 1.0 / 20
_MIN_DELAY_CEIL = 2.0
_INDENT_COLS = 2
_CONT_LINE_PREFIX = "\x1b[0m  "


class MarkdownStream:
    def __init__(self, console: Console, theme: CliTheme) -> None:
        self._console = console
        buf = io.StringIO()
        offline = Console(
            file=buf,
            force_terminal=True,
            color_system="truecolor",
            theme=theme.rich,
            width=10,
        )
        offline.print(Text(f"{TOOL_DONE} ", style="primary"), end="")
        self._first_line_prefix = "\x1b[0m" + buf.getvalue()
        self._buffer: str = ""
        self._live: Live | None = None
        self._printed: list[str] = []
        self._last_update_ts: float = 0.0
        self._min_delay: float = _MIN_DELAY_FLOOR

    def update(self, delta: str) -> None:
        if not delta:
            return
        self._buffer += delta
        self._render(final=False)

    def pause(self) -> None:
        self._render(final=True)
        self._reset_after_final()

    def finalize(self) -> None:
        self._render(final=True)
        self._reset_after_final()

    def abort(self) -> None:
        if self._live is not None:
            self._live.update(Text(""), refresh=True)
            self._live.stop()
            self._live = None
        self._reset_after_final()

    def _reset_after_final(self) -> None:
        self._buffer = ""
        self._printed = []
        self._last_update_ts = 0.0
        self._min_delay = _MIN_DELAY_FLOOR

    def _render(self, final: bool) -> None:
        if self._live is None:
            if final and not self._buffer:
                return
            self._live = Live(
                Text(""),
                console=self._console,
                auto_refresh=False,
                transient=False,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self._live.start()

        now = time.monotonic()
        if not final and now - self._last_update_ts < self._min_delay:
            return
        self._last_update_ts = now

        start = time.monotonic()
        lines = self._render_to_ansi_lines(self._buffer)
        render_time = time.monotonic() - start
        self._min_delay = min(max(render_time * 10, _MIN_DELAY_FLOOR), _MIN_DELAY_CEIL)

        num_lines = len(lines)
        if not final:
            num_lines = max(0, num_lines - _LIVE_WINDOW)

        if final or num_lines > 0:
            num_printed = len(self._printed)
            if num_lines > num_printed:
                promote = "".join(lines[num_printed:num_lines])
                self._live.console.print(Text.from_ansi(promote), end="")
                self._printed = lines[:num_lines]

        if final:
            self._live.update(Text(""), refresh=True)
            self._live.stop()
            self._live = None
            return

        tail = "".join(lines[num_lines:])
        self._live.update(Text.from_ansi(tail), refresh=True)

    def _render_to_ansi_lines(self, text: str) -> list[str]:
        buf = io.StringIO()
        offline = Console(
            file=buf,
            force_terminal=True,
            color_system="truecolor",
            width=max(self._console.width - _INDENT_COLS, 20),
        )
        offline.print(Markdown(text))
        raw = buf.getvalue().splitlines(keepends=True)
        return [
            (self._first_line_prefix if i == 0 else _CONT_LINE_PREFIX) + line
            for i, line in enumerate(raw)
        ]


def render_markdown_block(
    console: Console, text: str, theme: CliTheme,
) -> None:
    if not text:
        return
    prefix_buf = io.StringIO()
    Console(
        file=prefix_buf,
        force_terminal=True,
        color_system="truecolor",
        theme=theme.rich,
        width=10,
    ).print(Text(f"{TOOL_DONE} ", style="primary"), end="")
    first_prefix = "\x1b[0m" + prefix_buf.getvalue()

    body_buf = io.StringIO()
    Console(
        file=body_buf,
        force_terminal=True,
        color_system="truecolor",
        width=max(console.width - _INDENT_COLS, 20),
    ).print(Markdown(text))
    raw = body_buf.getvalue().splitlines(keepends=True)
    out = "".join(
        (first_prefix if i == 0 else _CONT_LINE_PREFIX) + line
        for i, line in enumerate(raw)
    )
    console.print(Text.from_ansi(out), end="")
