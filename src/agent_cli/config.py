"""CLI startup config discovery + first-run bootstrap."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import agent_cli

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
    template = Path(agent_cli.__file__).resolve().parents[2] / "arktor_example.yaml"
    if not template.exists():
        return False
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    except OSError:
        return False
    return True


def load_config() -> ConfigLoadResult:
    from dotenv import find_dotenv, load_dotenv

    from agent_harness.core.config import HarnessConfig

    load_dotenv(find_dotenv(usecwd=True))

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

    HarnessConfig.load(None, env_override=True)
    return ConfigLoadResult(path=None, bootstrapped=False)


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
    import logging

    from rich.logging import RichHandler

    handler = RichHandler(
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
