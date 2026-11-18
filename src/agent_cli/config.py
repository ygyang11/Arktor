"""CLI startup config discovery + first-run bootstrap."""
from __future__ import annotations

import asyncio
import copy
import logging
import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.application.current import get_app_or_none
from rich.logging import RichHandler

if TYPE_CHECKING:
    from rich.console import Console


@dataclass(frozen=True, slots=True)
class ConfigLoadResult:
    path: Path | None
    bootstrapped: bool


def current_effort() -> str | None:
    from agent_harness.core.config import HarnessConfig
    return HarnessConfig.get().llm.reasoning_effort


def _bootstrap_user_config(dest: Path) -> bool:
    temp_path: Path | None = None
    try:
        template = (
            files("agent_harness")
            .joinpath("resources", "arktor.example.yaml")
            .read_text(encoding="utf-8")
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=dest.parent,
            prefix=f".{dest.name}.", suffix=".tmp", delete=False,
        ) as output:
            temp_path = Path(output.name)
            output.write(template)
        os.link(temp_path, dest)
    except FileExistsError:
        return False
    except OSError:
        return False
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                logging.getLogger(__name__).debug(
                    "Temporary config cleanup failed.", exc_info=True,
                )
    return True


def load_config() -> ConfigLoadResult:
    from dotenv import find_dotenv, load_dotenv

    from agent_harness.core.config import HarnessConfig

    load_dotenv(find_dotenv(usecwd=True))

    if explicit := os.environ.get("ARKTOR_CONFIG"):
        config_path = Path(explicit).expanduser()
        HarnessConfig.load(config_path)
        return ConfigLoadResult(path=config_path, bootstrapped=False)

    project_cfg = Path.cwd() / "arktor.yaml"
    if project_cfg.exists():
        HarnessConfig.load(project_cfg)
        return ConfigLoadResult(path=project_cfg, bootstrapped=False)

    user_cfg = Path.home() / ".arktor" / "arktor.yaml"
    bootstrapped = False
    if not user_cfg.exists():
        bootstrapped = _bootstrap_user_config(user_cfg)

    if user_cfg.exists():
        HarnessConfig.load(user_cfg)
        return ConfigLoadResult(path=user_cfg, bootstrapped=bootstrapped)

    HarnessConfig.load()
    return ConfigLoadResult(path=None, bootstrapped=False)


class _PromptAwareRichHandler(RichHandler):
    """RichHandler that renders records above an active prompt_toolkit prompt
    instead of writing into the input line. Falls back to plain RichHandler
    behaviour when no prompt is running."""

    def emit(self, record: logging.LogRecord) -> None:
        app = get_app_or_none()
        loop = app.loop if app is not None else None
        if app is None or not app.is_running or loop is None:
            super().emit(record)
            return

        record = copy.copy(record)

        async def _emit_above_prompt() -> None:
            try:
                await run_in_terminal(lambda: RichHandler.emit(self, record))
            except Exception:
                self.handleError(record)

        try:
            on_app_loop = asyncio.get_running_loop() is loop
        except RuntimeError:
            on_app_loop = False

        coro = _emit_above_prompt()
        try:
            if on_app_loop:
                app.create_background_task(coro)
            else:
                loop.call_soon_threadsafe(app.create_background_task, coro)
        except Exception:
            coro.close()
            super().emit(record)


def attach_rich_logging(console: Console) -> None:
    """Route agent_harness/agent_app logs through the shared rich Console.

    setup_logging() binds a bare StreamHandler to the original stderr at
    startup; its writes bypass any later rich Live and corrupt in-place
    repaint. Swapping in a RichHandler on the same Console makes log
    records render above an active Live instead of through it. The
    handler stays at NOTSET so the logger level (which /debug toggles
    via setup_logging) remains the sole gate.

    The same handler is also attached to the root logger so third-party
    library records (pypdf, httpx, docker, …) — which propagate to root
    and would otherwise hit logging.lastResort and write raw to stderr,
    corrupting the Live region — render above the Live too. Our two
    namespaces keep propagate=False so they are not double-handled.
    """
    handler = _PromptAwareRichHandler(
        console=console,
        show_time=False,
        show_path=False,
        markup=False,
        rich_tracebacks=False,
    )
    handler.set_name("cli-rich")
    for name in ("agent_harness", "agent_app"):
        lg = logging.getLogger(name)
        for h in list(lg.handlers):
            if type(h) is logging.StreamHandler or h.get_name() == "cli-rich":
                lg.removeHandler(h)
        lg.addHandler(handler)
        lg.propagate = False

    root = logging.getLogger()
    for h in list(root.handlers):
        if h.get_name() == "cli-rich":
            root.removeHandler(h)
    root.addHandler(handler)
