"""Headless single-shot run mode.

Runs one task non-interactively: resolve the session exactly like the
interactive path, run the agent unrestricted (no approval prompts), print
only the final result to stdout, and exit. No renderer, no REPL, no chrome.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_harness import AgentResult


async def _safe(coro: Awaitable[object], what: str) -> None:
    try:
        await coro
    except Exception as e:
        print(f"arktor: failed to {what}: {e}", file=sys.stderr)


def _emit_json(session_id: str, result: AgentResult | None, err: str) -> None:
    """Emit the run as NDJSON on stdout: one object per step, then a final
    result line carrying session_id, output and usage. On failure a single
    is_error result line stands in for the whole run."""
    lines: list[dict[str, Any]] = []
    if result is None:
        lines.append({
            "type": "result", "is_error": True, "session_id": session_id,
            "output": None, "error": err,
        })
    else:
        for i, step in enumerate(result.steps):
            lines.append({"type": "step", "index": i, **step.model_dump(mode="json")})
        lines.append({
            "type": "result", "is_error": False, "session_id": session_id,
            "output": result.output, "num_steps": len(result.steps),
            "usage": result.usage.model_dump(mode="json"),
        })
    for line in lines:
        sys.stdout.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


async def run_headless(args: argparse.Namespace) -> int:
    task = (args.prompt or "").strip()
    if not task:
        print("arktor: -p/--prompt requires a non-empty task", file=sys.stderr)
        return 2

    from agent_cli import _check_deps

    _check_deps()

    from agent_cli.agent_factory import create_cli_agent
    from agent_cli.config import load_config
    from agent_cli.runtime import background
    from agent_cli.runtime.session import (
        get_policy,
        make_save_session,
        resolve_session_id,
        restore_session,
        stop_sandbox,
    )
    from agent_harness import AutoApproveHandler, setup_logging
    from agent_harness.hooks.base import DefaultHooks
    from agent_harness.session.file_session import FileSession

    setup_logging("WARNING")
    load_config()

    agent = create_cli_agent(
        hooks=DefaultHooks(),
        approval_handler=AutoApproveHandler(),
    )

    probe = FileSession("_probe")
    session_id = await resolve_session_id(args, probe)
    if session_id is None:
        return 2
    backend = FileSession(session_id)

    if args.resume_latest or args.resume:
        if await restore_session(agent, backend) is None:
            print(f"arktor: session corrupted: {session_id}", file=sys.stderr)
            return 2

    policy = get_policy(agent)
    original_mode = policy.mode
    policy.set_mode("never")
    save = make_save_session(agent, backend)

    fmt = args.output_format or "text"
    rc = 0
    result = None
    err = ""
    try:
        result = await agent.run(task, session=backend)
    except Exception as e:
        err = str(e)
        rc = 1
    finally:
        policy.set_mode(original_mode)
        await _safe(save(), "persist session")
        if background.has_running(agent):
            background.cancel_all(agent)
        await _safe(background.shutdown(agent), "shut down background tasks")
        await _safe(stop_sandbox(agent), "stop sandbox")

    if fmt == "json":
        _emit_json(session_id, result, err)
    elif result is not None:
        print(result.output)
    else:
        print(f"arktor: {err}", file=sys.stderr)
    return rc
