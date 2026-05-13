"""Static UI chrome for the CLI"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from prompt_toolkit.application import get_app
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.text import Text

from agent_cli.commands.ui import MODE_INFO
from agent_cli.runtime.session import current_mode_key
from agent_cli.theme import SEP_DOT
from agent_harness.agent.base import BaseAgent
from agent_harness.session.base import BaseSession

_BANNER_LINES: tuple[str, ...] = (
    "██╗  ██╗ █████╗ ██████╗ ███╗   ██╗███████╗███████╗███████╗",
    "██║  ██║██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝██╔════╝",
    "███████║███████║██████╔╝██╔██╗ ██║█████╗  ███████╗███████╗",
    "██╔══██║██╔══██║██╔══██╗██║╚██╗██║██╔══╝  ╚════██║╚════██║",
    "██║  ██║██║  ██║██║  ██║██║ ╚████║███████╗███████║███████║",
    "╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚══════╝",
)
_BANNER_WIDTH = 58
_LABEL_WIDTH = 9


_DEFAULT_SESSION_LABEL = f"fresh {SEP_DOT} /resume to restore"


def render_welcome(
    console: Console,
    *,
    version: str,
    model: str,
    cwd: str,
    config_source: str,
    session_label: str = _DEFAULT_SESSION_LABEL,
) -> None:
    """Print the welcome block — banner, tagline, meta rows, command hint."""
    console.print()

    for line in _BANNER_LINES:
        console.print(Text(line, style="primary"))

    tag = "Agents, harnessed."
    ver = f"v{version}"
    pad = max(_BANNER_WIDTH - len(tag) - len(ver), 1)
    row = Text()
    row.append(tag, style="bold text")
    row.append(" " * pad)
    row.append(ver, style="muted")
    console.print(row)
    console.print()

    console.print(_meta_row("model", model))
    console.print(_meta_row("cwd", shorten_home(cwd)))
    console.print(_meta_row("config", shorten_home(config_source)))
    console.print(_session_row(session_label))
    console.print()
    console.print(_hint_row())
    console.print()


def _meta_row(label: str, value: str) -> Text:
    row = Text()
    row.append("  ")
    row.append(label.ljust(_LABEL_WIDTH), style="muted")
    row.append(" ")
    row.append(value, style="text")
    return row


def _session_row(label: str) -> Text:
    row = Text()
    row.append("  ")
    row.append("session".ljust(_LABEL_WIDTH), style="muted")
    row.append(" ")
    row.append(label, style="text")
    return row


def _hint_row() -> Text:
    row = Text()
    row.append("  ")
    for i, (glyph, label) in enumerate((("/", "commands"), ("@", "files"), ("!", "shell"))):
        if i > 0:
            row.append("     ")
        row.append(glyph, style="primary")
        row.append(" ")
        row.append(label, style="muted")
    return row


def shorten_home(cwd: str) -> str:
    home = str(Path.home())
    return cwd.replace(home, "~", 1) if cwd.startswith(home) else cwd


async def print_exit_reminder(console: Console, backend: BaseSession) -> None:
    # Use backend.session_id (current), not the startup-resolved one — REPL
    # commands like /new and /resume mutate backend in place.
    sid = backend.session_id
    if not await backend.has_session(sid):
        return
    console.print("[muted]Resume this session with:[/muted]")
    console.print(f"[muted]  harness --resume {sid}[/muted]")


def make_status_bar_text(agent: BaseAgent) -> Callable[[], HTML]:
    """Return a closure that re-collects every render so mode/tokens stay live."""
    from agent_cli.runtime.status import collect as collect_status

    def _render() -> HTML:
        snap = collect_status(agent)
        cur = _fmt(snap.input_tokens) if snap.input_tokens is not None else "—"
        info = MODE_INFO[current_mode_key(agent)]

        hint = "(shift+tab to cycle)"
        right = f"{snap.model} · {cur}/{_fmt(snap.max_tokens)}"
        left_plain = f" {info.label} mode {hint}"
        right_plain = f"{right} "
        width = get_app().output.get_size().columns
        pad = max(1, width - len(left_plain) - len(right_plain))

        return HTML(
            f' <{info.style}>{info.label}</{info.style}> mode <muted>{hint}</muted>'
            f'{" " * pad}{right} '
        )

    return _render


def _fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n // 1000}k"
    return str(n)