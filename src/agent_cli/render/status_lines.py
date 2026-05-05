"""ThinkingLine and SubagentLine — heartbeat renderers over LiveLine.

Own their own visual state (ANSI accent colour) and, for SubagentLine, the
subagent execution counters (active_count / steps / tools / was_parallel).
Adapter triggers start / stop / tick; all rendering concerns live here.
"""
from __future__ import annotations

import asyncio
import random
from collections.abc import Callable

from rich.console import Console

from agent_cli.render.live_line import LiveLine
from agent_cli.theme import ELLIPSIS_FRAMES, SPINNER_FRAMES, CliTheme

THINKING_DEBOUNCE_S = 0.4
THINKING_TICK_S = 0.1          # ~10 fps — smooth spinner motion
SUBAGENT_DEBOUNCE_S = 0.5
SUBAGENT_TICK_S = 0.1
_DETAIL_AFTER_S = 5
_ELLIPSIS_TICKS = 10

_ANSI_DIM = "\x1b[2m"
_ANSI_RESET = "\x1b[0m"

# Curated flavor pool
_THINKING_WORDS: tuple[str, ...] = (
    "Meditating", "Centering", "Settling", "Abiding", "Dwelling",
    "Simmering", "Brewing", "Stewing", "Marinating",
    "Percolating", "Distilling", "Steeping", "Infusing",
    "Fermenting", "Baking", "Kneading", "Reducing",
    "Churning", "Swirling", "Bubbling", "Drifting",
    "Floating", "Soaking", "Tumbling", "Whirling", "Lingering",
    "Crystallizing", "Coalescing", "Condensing",
    "Musing", "Mulling", "Noodling", "Wandering", "Daydreaming",
    "Weaving", "Untangling", "Unfurling"
    "Wrestling", "Puzzling", "Grappling", "Wading",
    "Digesting", "Whatchamacalliting", "Philosophising"
)


def _fmt_duration(n: int) -> str:
    if n < 60:
        return f"{n}s"
    m, s = divmod(n, 60)
    return f"{m}m {s}s"


def _ansi_accent(theme: CliTheme) -> str:
    hex_color = theme.palette.primary.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"\x1b[38;2;{r};{g};{b}m"


class ThinkingLine:
    def __init__(
        self,
        console: Console,
        lock: asyncio.Lock,
        theme: CliTheme,
        effort: str | None = None,
        run_elapsed_provider: Callable[[], int | None] | None = None,
    ) -> None:
        self._accent = _ansi_accent(theme)
        self._effort = effort
        self._run_elapsed_provider = run_elapsed_provider
        self._current_word: str = _THINKING_WORDS[0]
        self._line = LiveLine(
            console=console,
            lock=lock,
            writer=self._write,
            debounce_s=THINKING_DEBOUNCE_S,
            tick_s=THINKING_TICK_S,
        )

    def _write(self, frame: int, elapsed: int) -> str:
        glyph = SPINNER_FRAMES[frame % len(SPINNER_FRAMES)]
        # Decouple dot breathing from spinner: dots advance ~once per second
        dots = ELLIPSIS_FRAMES[(frame // _ELLIPSIS_TICKS) % len(ELLIPSIS_FRAMES)]

        head = f"{self._accent}{glyph} {self._current_word}{dots}{_ANSI_RESET}"
        detail = ""
        if elapsed >= _DETAIL_AFTER_S:
            parts = [_fmt_duration(elapsed)]
            if self._effort:
                parts.append(f"thinking with {self._effort} effort")
            detail = f"{_ANSI_DIM} ({' · '.join(parts)}){_ANSI_RESET}"
        elif self._run_elapsed_provider is not None:
            run_elapsed = self._run_elapsed_provider()
            if run_elapsed is not None:
                detail = (
                    f"{_ANSI_DIM} (Working {_fmt_duration(run_elapsed)}){_ANSI_RESET}"
                )
        return f"{head}{detail}"

    async def start(self) -> None:
        self._current_word = random.choice(_THINKING_WORDS)
        await self._line.start()

    async def stop(self) -> None:
        await self._line.stop()

    async def stop_no_lock(self) -> None:
        await self._line.stop_no_lock()

    def clear_no_lock(self) -> None:
        self._line.clear_no_lock()

    @property
    def visible(self) -> bool:
        return self._line.visible

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._line.task


class SubagentLine:
    def __init__(self, console: Console, lock: asyncio.Lock, theme: CliTheme) -> None:
        self._accent = _ansi_accent(theme)
        self._active_count: int = 0
        self._steps: int = 0
        self._tools: int = 0
        self._was_parallel: bool = False
        self._line = LiveLine(
            console=console,
            lock=lock,
            writer=self._write,
            debounce_s=SUBAGENT_DEBOUNCE_S,
            tick_s=SUBAGENT_TICK_S,
        )

    def _write(self, frame: int, elapsed: int) -> str:
        glyph = SPINNER_FRAMES[frame % len(SPINNER_FRAMES)]
        count = self._active_count
        label = "SubAgent" if count == 1 else f"{count} Subagents"
        show_total = count >= 2 or self._was_parallel
        suffix = " · total" if show_total else ""
        return (
            f"{self._accent}{glyph} {label}{_ANSI_RESET} "
            f"{_ANSI_DIM}· {_fmt_duration(elapsed)} · "
            f"{self._steps} steps · {self._tools} tools{suffix}{_ANSI_RESET}"
        )

    @property
    def active_count(self) -> int:
        return self._active_count

    @property
    def was_parallel(self) -> bool:
        return self._was_parallel

    @property
    def visible(self) -> bool:
        return self._line.visible

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._line.task

    async def start(self) -> None:
        self._active_count += 1
        if self._active_count == 1:
            # State mutations MUST happen before any await so concurrent
            # parallel start() calls can't race: if we awaited first and
            # then reset was_parallel=False, a concurrent 1→2 elif branch
            # (was_parallel=True) could be clobbered when we resume.
            self._steps = 0
            self._tools = 0
            self._was_parallel = False
            await self._line.start()
        elif self._active_count >= 2:
            self._was_parallel = True

    async def stop(self) -> None:
        if self._active_count <= 0:
            return
        self._active_count -= 1
        if self._active_count == 0:
            await self._line.stop()

    async def force_stop(self) -> None:
        if self._active_count > 0:
            self._active_count = 0
            self._was_parallel = False
            await self._line.stop()

    def tick_step(self) -> None:
        self._steps += 1

    def tick_tool(self) -> None:
        self._tools += 1

    async def stop_no_lock(self) -> None:
        await self._line.stop_no_lock()

    def clear_no_lock(self) -> None:
        self._line.clear_no_lock()
