"""Tests for the shared status-line writer and factory."""
from __future__ import annotations

import asyncio
import io

import pytest
from rich.console import Console

from agent_cli.render.status_lines import (
    _ansi_color,
    make_command_status_line,
    status_line_write,
)
from agent_cli.theme import DEFAULT_THEME, ELLIPSIS_FRAMES, SPINNER_FRAMES


def test_status_line_write_first_frame_uses_first_spinner_glyph() -> None:
    out = status_line_write(0, accent="", label="Compacting")
    assert out.startswith(SPINNER_FRAMES[0])
    assert "Compacting" in out


def test_status_line_write_rotates_spinner_by_frame() -> None:
    glyphs_seen = {status_line_write(i, accent="", label="X")[0] for i in range(len(SPINNER_FRAMES))}
    assert glyphs_seen == set(SPINNER_FRAMES)


def test_status_line_write_animates_ellipsis_dots() -> None:
    """Dots advance via `_ELLIPSIS_TICKS`-scaled frame index, cycling through
    each of `ELLIPSIS_FRAMES`."""
    from agent_cli.render.status_lines import _ELLIPSIS_TICKS

    samples = [
        status_line_write(i * _ELLIPSIS_TICKS, accent="", label="X")
        for i in range(len(ELLIPSIS_FRAMES))
    ]
    for dots, rendered in zip(ELLIPSIS_FRAMES, samples):
        assert rendered.endswith(f"{dots}\x1b[0m")


def test_status_line_write_wraps_with_accent_and_reset() -> None:
    accent = "\x1b[38;2;255;0;0m"
    out = status_line_write(0, accent=accent, label="X")
    assert out.startswith(accent)
    assert out.endswith("\x1b[0m")


def test_ansi_color_resolves_palette_attr() -> None:
    primary = _ansi_color(DEFAULT_THEME, "primary")
    info = _ansi_color(DEFAULT_THEME, "info")
    assert primary.startswith("\x1b[38;2;")
    assert info.startswith("\x1b[38;2;")
    assert primary != info  # different palette attrs → different codes


def test_ansi_color_defaults_to_primary() -> None:
    assert _ansi_color(DEFAULT_THEME) == _ansi_color(DEFAULT_THEME, "primary")


@pytest.mark.asyncio
async def test_make_command_status_line_starts_and_stops_cleanly() -> None:
    """The factory returns a LiveLine; start() makes it visible, stop()
    cancels the loop and clears the row. No exceptions, no hangs."""
    console = Console(file=io.StringIO(), force_terminal=False, color_system=None, width=80)
    lock = asyncio.Lock()
    line = make_command_status_line(
        console, lock, DEFAULT_THEME, label="Compacting", color="info",
    )
    await line.start()
    # Yield a couple ticks so the loop actually renders at least one frame.
    await asyncio.sleep(0.05)
    await line.stop()
    assert line.task is None or line.task.done()
