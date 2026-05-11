"""CLIApp — top-level lifecycle and entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import uuid
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness",
        description="Agent-Harness interactive CLI",
    )
    parser.add_argument("--version", action="store_true", help="show version and exit")
    return parser


async def _async_main() -> int:
    from agent_cli import __version__, _check_deps

    _check_deps()

    # Framework emits INFO logs via stderr StreamHandler; these break Rich
    # Live's in-place repaint. Raise to WARNING for interactive CLI.
    from agent_harness import setup_logging

    setup_logging("WARNING")

    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_input
    from prompt_toolkit.output import create_output
    from rich.console import Console

    from agent_cli.adapter import CliAdapter
    from agent_cli.agent_factory import create_cli_agent
    from agent_cli.approval_handler import CliApprovalHandler
    from agent_cli.commands.builtin import register_builtin
    from agent_cli.commands.registry import CommandRegistry
    from agent_cli.config import load_config
    from agent_cli.hooks import CliHooks
    from agent_cli.render.ui import render_welcome
    from agent_cli.repl.lexer import ShellLineLexer
    from agent_cli.repl.loop import run_repl
    from agent_cli.runtime.shell import ShellState
    from agent_cli.theme import load_saved_theme
    from agent_harness.session.file_session import FileSession

    config_result = load_config()

    theme = load_saved_theme()
    console = Console(theme=theme.rich)
    adapter = CliAdapter(console, theme, effort=config_result.effort)
    pt_session: PromptSession[str] = PromptSession(
        input=create_input(),
        output=create_output(),
        refresh_interval=0.1,
        style=theme.completion,
        lexer=ShellLineLexer(),
    )
    shell_state = ShellState()
    approval_handler = CliApprovalHandler(
        console=console,
        adapter=adapter,
        pt_session=pt_session,
    )
    hooks = CliHooks(adapter, approval_handler=approval_handler)
    agent = create_cli_agent(hooks=hooks, approval_handler=approval_handler)
    session_id = str(uuid.uuid4())

    registry = CommandRegistry()
    register_builtin(registry)
    backend = FileSession(session_id)

    config_source = str(config_result.path) if config_result.path else "Defaults env"
    render_welcome(
        console,
        version=__version__,
        model=agent.llm.model_name,
        cwd=str(Path.cwd()),
        config_source=config_source,
    )
    if config_result.bootstrapped:
        console.print(
            f"[dim]config created at [bold]{config_result.path}[/bold] — edit to customize[/dim]\n"
        )
        console.print()

    # Take over SIGINT from asyncio.run's default handler. Between-turn
    # idle absorbs Ctrl+C silently; per-turn task-bound handlers (via
    # runtime.sigint.bind_work) cancel only their own work task.
    from agent_cli.runtime.sigint import install_idle, uninstall

    install_idle()

    try:
        await run_repl(
            agent,
            console,
            session_id,
            registry,
            backend,
            adapter,
            approval_handler,
            pt_session,
            shell_state,
            hooks,
            theme,
        )
    finally:
        shell_state.cleanup()
        uninstall()
        await adapter.end_step()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.version:
        from agent_cli import __version__
        print(f"harness {__version__}")
        return 0
    try:
        return asyncio.run(_async_main())
    except KeyboardInterrupt:
        return 130
