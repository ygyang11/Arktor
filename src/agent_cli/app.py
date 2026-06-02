"""CLIApp — top-level lifecycle and entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arktor",
        description="Arktor interactive CLI",
    )
    parser.add_argument("--version", action="store_true", help="show version and exit")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "-c", "--continue", dest="resume_latest", action="store_true",
        help="resume the most recently updated session",
    )
    grp.add_argument(
        "-r", "--resume", metavar="ID", default=None,
        help="resume the session with the given id",
    )
    grp.add_argument(
        "-s", "--session-id", dest="session_id", metavar="ID", default=None,
        help="start a new session with the given id ([a-zA-Z0-9_-])",
    )
    return parser


async def _async_main(args: argparse.Namespace) -> int:
    from agent_cli import __version__, _check_deps

    _check_deps()

    # Framework emits INFO logs via stderr StreamHandler; these break Rich
    # Live's in-place repaint. Raise to WARNING for interactive CLI.
    from agent_harness import setup_logging

    setup_logging("WARNING")

    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_input
    from prompt_toolkit.output import create_output
    from prompt_toolkit.output.color_depth import ColorDepth
    from rich.console import Console

    from agent_cli.adapter import CliAdapter
    from agent_cli.agent_factory import create_cli_agent
    from agent_cli.approval_handler import CliApprovalHandler
    from agent_cli.commands.builtin import register_builtin, register_dynamic
    from agent_cli.commands.registry import CommandRegistry
    from agent_cli.config import attach_rich_logging, load_config
    from agent_cli.hooks import CliHooks
    from agent_cli.render.ui import print_exit_reminder, render_welcome
    from agent_cli.repl.loop import run_repl
    from agent_cli.runtime.shell import ShellState
    from agent_cli.theme import DEPTH_MAP, load_saved_theme
    from agent_harness.session.file_session import FileSession

    config_result = load_config()

    theme = load_saved_theme()
    console = Console(theme=theme.rich)
    attach_rich_logging(console)
    adapter = CliAdapter(console, theme)
    color_depth = (
        ColorDepth.DEPTH_1_BIT if console.no_color
        else DEPTH_MAP.get(console.color_system or "")
    )
    pt_session: PromptSession[str] = PromptSession(
        input=create_input(),
        output=create_output(),
        refresh_interval=0.1,
        style=theme.completion,
        color_depth=color_depth,
    )
    shell_state = ShellState()
    approval_handler = CliApprovalHandler(
        console=console,
        adapter=adapter,
        pt_session=pt_session,
    )
    hooks = CliHooks(adapter, approval_handler=approval_handler)
    agent = create_cli_agent(hooks=hooks, approval_handler=approval_handler)

    from agent_cli.runtime.session import resolve_session_id, restore_session

    probe = FileSession("_probe")
    session_id = await resolve_session_id(args, probe)
    if session_id is None:
        return 2

    registry = CommandRegistry()
    register_builtin(registry)
    register_dynamic(registry, agent)
    backend = FileSession(session_id)

    if args.resume_latest or args.resume:
        if await restore_session(agent, backend) is None:
            print(f"arktor: session corrupted: {session_id}", file=sys.stderr)
            return 2

    config_source = str(config_result.path) if config_result.path else "Defaults env"
    welcome_kwargs: dict[str, str] = {}
    if args.resume_latest or args.resume or args.session_id:
        welcome_kwargs["session_label"] = session_id
    render_welcome(
        console,
        version=__version__,
        model=agent.llm.model_name,
        cwd=str(Path.cwd()),
        config_source=config_source,
        **welcome_kwargs,
    )
    if args.resume_latest or args.resume:
        from agent_cli.render.replay import render_session_replay

        msgs = await agent.context.short_term_memory.get_context_messages()
        render_session_replay(console, theme, msgs, session_id)
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
        await asyncio.sleep(0)

    await print_exit_reminder(console, backend)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.version:
        from agent_cli import __version__
        print(f"arktor {__version__}")
        return 0
    try:
        return asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        return 130
