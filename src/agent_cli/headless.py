"""Headless non-interactive run mode."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_cli.runtime.goal.mode import GoalState
    from agent_cli.runtime.session import SaveSession
    from agent_cli.runtime.status import WindowView
    from agent_harness import AgentResult
    from agent_harness.agent.base import BaseAgent
    from agent_harness.core.message import Message
    from agent_harness.llm.types import Usage
    from agent_harness.session.base import BaseSession


@dataclass(slots=True)
class _TaskOutput:
    output: str
    num_steps: int
    usage: Usage


@dataclass(slots=True)
class _GoalOutput:
    output: str = ""
    num_steps: int = 0


async def _safe(coro: Awaitable[object], what: str) -> None:
    try:
        await coro
    except Exception as e:
        print(f"arktor: failed to {what}: {e}", file=sys.stderr)


def _emit_json_line(event: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def _emit_step_events(result: AgentResult, start_index: int) -> None:
    for index, step in enumerate(result.steps, start=start_index):
        _emit_json_line({
            "type": "step",
            "index": index,
            **step.model_dump(mode="json"),
        })


def _emit_error_result(session_id: str, err: str) -> None:
    _emit_json_line({
        "type": "result",
        "is_error": True,
        "session_id": session_id,
        "output": None,
        "error": err,
    })


def _emit_success_result(
    session_id: str,
    output: str,
    num_steps: int,
    usage: Usage,
    window: WindowView,
) -> None:
    _emit_json_line({
        "type": "result",
        "is_error": False,
        "session_id": session_id,
        "output": output,
        "num_steps": num_steps,
        "usage": usage.model_dump(mode="json"),
        "context": {
            "input_tokens": window.displayed_input_tokens,
            "max_tokens": window.max_tokens,
        },
    })


def _emit_goal_status(
    fmt: str,
    agent: BaseAgent,
    goal: GoalState | None,
    status: str,
    reason: str,
) -> None:
    if fmt == "json":
        process_tokens = agent.context.usage_meter.total.total_tokens
        _emit_json_line({
            "type": "goal",
            "status": status,
            "reason": reason,
            "turns": goal.turns if goal is not None else 0,
            "elapsed_s": goal.elapsed_s() if goal is not None else 0,
            "tokens": (
                goal.tokens_used(process_tokens)
                if goal is not None
                else 0
            ),
        })
        return
    print(
        f"[goal {status}] {reason}",
        file=sys.stderr if status == "error" else sys.stdout,
        flush=True,
    )


def _parse_task(
    args: argparse.Namespace,
) -> tuple[str, str | None] | None:
    task = (args.prompt or "").strip()
    if not task:
        print(
            "arktor: -p/--prompt requires a non-empty task",
            file=sys.stderr,
        )
        return None

    parts = task.split(maxsplit=1)
    is_goal = parts[0].lower() == "/goal"
    objective = parts[1].strip() if is_goal and len(parts) == 2 else None

    if is_goal and not objective:
        print("arktor: /goal requires a non-empty objective", file=sys.stderr)
        return None
    if objective in {"pause", "resume", "clear"}:
        print(
            "arktor: headless /goal only accepts a new objective; "
            "manage existing goals interactively",
            file=sys.stderr,
        )
        return None
    if args.max_steps is not None and args.max_steps <= 0:
        print("arktor: --max-steps must be positive", file=sys.stderr)
        return None
    if args.max_turns is not None:
        if args.max_turns <= 0:
            print("arktor: --max-turns must be positive", file=sys.stderr)
            return None
        if not is_goal:
            print(
                "arktor: --max-turns requires a /goal prompt",
                file=sys.stderr,
            )
            return None

    return task, objective


def _validate_new_goal(agent: BaseAgent, objective: str) -> bool:
    from agent_cli.runtime import plan_mode
    from agent_cli.runtime.goal import mode as goal_mode
    from agent_harness.utils.token_counter import count_tokens

    if plan_mode.is_active(agent):
        print("arktor: can't set a goal in plan mode", file=sys.stderr)
        return False
    if goal_mode.has_live_goal(agent):
        print(
            "arktor: session already has an active or paused goal; "
            "resume or clear it interactively, or use a fresh session",
            file=sys.stderr,
        )
        return False

    objective_tokens = count_tokens(objective, model=agent.llm.model_name)
    if objective_tokens > goal_mode.MAX_OBJECTIVE_TOKENS:
        print(
            "arktor: goal objective is too large for a persistent goal "
            f"(estimated {objective_tokens:,} tokens, "
            f"max {goal_mode.MAX_OBJECTIVE_TOKENS:,}). "
            "Put details in a file and reference it.",
            file=sys.stderr,
        )
        return False
    return True


async def run_headless(args: argparse.Namespace) -> int:
    parsed = _parse_task(args)
    if parsed is None:
        return 2
    task, objective = parsed
    fmt = args.output_format or "text"

    from agent_cli import _check_deps

    _check_deps()

    from agent_cli.agent_factory import create_cli_agent
    from agent_cli.config import load_config
    from agent_cli.runtime import background
    from agent_cli.runtime.goal import mode as goal_mode
    from agent_cli.runtime.session import (
        get_policy,
        make_save_session,
        resolve_session_id,
        restore_session,
        stop_sandbox,
    )
    from agent_cli.runtime.status import collect_window
    from agent_harness import AutoApproveHandler, setup_logging
    from agent_harness.hooks.base import DefaultHooks
    from agent_harness.session.file_session import FileSession

    setup_logging("WARNING")
    load_config()

    probe = FileSession("_probe")
    session_id = await resolve_session_id(args, probe)
    if session_id is None:
        return 2

    backend = FileSession(session_id)
    agent = create_cli_agent(
        hooks=DefaultHooks(),
        approval_handler=AutoApproveHandler(),
    )
    save = make_save_session(agent, backend)

    task_output: _TaskOutput | None = None
    rc = 0
    err = ""
    run_started = False
    original_mode: str | None = None
    goal: GoalState | None = None
    goal_output: _GoalOutput | None = None
    interrupted = False

    try:
        if args.resume_latest or args.resume:
            if await restore_session(agent, backend) is None:
                print(
                    f"arktor: session corrupted: {session_id}",
                    file=sys.stderr,
                )
                return 2

        if objective is not None and not _validate_new_goal(agent, objective):
            return 2

        policy = get_policy(agent)
        original_mode = policy.mode
        policy.set_mode("never")
        run_started = True

        if objective is not None:
            goal = goal_mode.begin(agent, objective)
            goal_output = _GoalOutput()
            await _drive_goal(
                agent,
                goal,
                goal_output,
                backend,
                save,
                fmt,
                args.max_turns,
                args.max_steps,
            )
        else:
            task_output = await _drive_task(
                agent,
                task,
                backend,
                fmt,
                args.max_steps,
            )
    except asyncio.CancelledError:
        interrupted = True
        raise
    except Exception as exc:
        err = str(exc) or type(exc).__name__
        rc = 1
    finally:
        if goal is not None:
            await _safe(
                background.collect_results(agent),
                "collect completed background results",
            )
            if background.has_running(agent):
                await _safe(
                    background.cancel_all_with_note(agent),
                    "cancel background tasks",
                )
            await _safe(
                background.shutdown(agent),
                "shut down background tasks",
            )
            await _safe(
                background.collect_results(agent),
                "collect completed background results",
            )
            if goal_mode.get_state(agent) is goal and goal.status == "active":
                reason = (
                    "interrupted"
                    if interrupted
                    else f"error: {err}"
                    if err
                    else "goal stopped without a terminal verdict"
                )
                goal_mode.pause(agent, reason=reason)

        if original_mode is not None:
            get_policy(agent).set_mode(original_mode)
        if run_started:
            await _safe(save(), "persist session")
        await _safe(stop_sandbox(agent), "stop sandbox")
        await _safe(agent.aclose(), "close llm client")

    if objective is not None:
        if (
            rc == 1
            or goal is None
            or goal_output is None
            or goal_mode.get_state(agent) is not goal
            or goal.status not in ("complete", "blocked")
        ):
            reason = (
                err
                or (goal.reason if goal is not None else "")
                or "goal stopped without a terminal verdict"
            )
            _emit_goal_status(fmt, agent, goal, "error", reason)
            if fmt == "json":
                _emit_error_result(session_id, reason)
            return 1
        if fmt == "json":
            _emit_success_result(
                session_id,
                goal_output.output,
                goal_output.num_steps,
                agent.context.usage_meter.total,
                collect_window(agent),
            )
        return 0 if goal.status == "complete" else 3

    if fmt == "json":
        if rc != 0 or task_output is None:
            _emit_error_result(session_id, err)
        else:
            _emit_success_result(
                session_id,
                task_output.output,
                task_output.num_steps,
                task_output.usage,
                collect_window(agent),
            )
    elif rc == 0 and task_output is not None:
        print(task_output.output)
    else:
        print(f"arktor: {err}", file=sys.stderr)
    return rc


async def _drive_task(
    agent: BaseAgent,
    task: str,
    backend: BaseSession,
    fmt: str,
    max_steps: int | None = None,
) -> _TaskOutput:
    next_index = 0

    async def emit_turn(result: AgentResult) -> None:
        nonlocal next_index
        if fmt == "json":
            _emit_step_events(result, next_index)
        next_index += len(result.steps)

    result = await agent.run(
        task, session=backend, on_turn_result=emit_turn, max_steps=max_steps,
    )
    return _TaskOutput(
        output=result.output,
        num_steps=len(result.steps),
        usage=result.usage,
    )


async def _run_goal_turn(
    agent: BaseAgent,
    goal: GoalState,
    output: _GoalOutput,
    inp: str | Message,
    backend: BaseSession,
    save: SaveSession,
    fmt: str,
    step_limit: int,
) -> bool:
    from agent_cli.runtime.goal import mode as goal_mode

    result = await agent.run_turn(
        inp, session=backend, max_steps=step_limit - output.num_steps,
    )
    if goal_mode.get_state(agent) is not goal or goal.status != "active":
        return False
    if goal_mode.record_completed_turn(agent) is None:
        return False
    await save()
    if fmt == "json":
        _emit_step_events(result, output.num_steps)
    else:
        print(result.output, flush=True)
    output.output = result.output
    output.num_steps += len(result.steps)
    return True


async def _drain_background_for_goal(
    agent: BaseAgent,
    goal: GoalState,
    output: _GoalOutput,
    backend: BaseSession,
    save: SaveSession,
    fmt: str,
    max_turns: int | None,
    step_limit: int,
) -> None:
    from agent_cli.runtime import background
    from agent_cli.runtime.goal import mode as goal_mode

    while True:
        if output.num_steps >= step_limit:
            await background.collect_results(agent)
            await save()
            return
        if not await _collect_next_background_batch(agent):
            return
        if (
            goal_mode.get_state(agent) is not goal
            or goal.status != "active"
        ):
            return
        if max_turns is not None and goal.turns >= max_turns:
            await save()
            continue
        if not await _run_goal_turn(
            agent,
            goal,
            output,
            _background_notification(),
            backend,
            save,
            fmt,
            step_limit,
        ):
            return


async def _drive_goal(
    agent: BaseAgent,
    goal: GoalState,
    output: _GoalOutput,
    backend: BaseSession,
    save: SaveSession,
    fmt: str,
    max_turns: int | None,
    max_steps: int | None = None,
) -> None:
    from agent_cli.runtime.goal import driver as goal_driver
    from agent_cli.runtime.goal import mode as goal_mode

    step_limit = agent.max_steps if max_steps is None else max_steps

    if not await _run_goal_turn(
        agent,
        goal,
        output,
        goal_mode.make_start_input(goal.objective),
        backend,
        save,
        fmt,
        step_limit,
    ):
        return

    while True:
        await _drain_background_for_goal(
            agent,
            goal,
            output,
            backend,
            save,
            fmt,
            max_turns,
            step_limit,
        )

        if (
            goal_mode.get_state(agent) is not goal
            or goal.status != "active"
        ):
            return
        decision = await goal_driver.decide(agent)
        if decision is None:
            return
        status = decision.status
        reason = decision.reason
        continuation = decision.continuation
        if continuation is not None:
            limit_reason = ""
            if max_turns is not None and goal.turns >= max_turns:
                limit_reason = f"reached --max-turns ({max_turns})"
            elif output.num_steps >= step_limit:
                limit_reason = f"reached max_steps ({step_limit})"
            if limit_reason:
                status = "blocked"
                reason = limit_reason
                if goal_mode.finish(agent, "blocked", reason) is None:
                    return
                continuation = None

        await save()
        _emit_goal_status(fmt, agent, goal, status, reason)

        if continuation is None:
            return

        if not await _run_goal_turn(
            agent,
            goal,
            output,
            continuation,
            backend,
            save,
            fmt,
            step_limit,
        ):
            return


def _background_notification() -> Message:
    from agent_harness.core.message import Message

    return Message.system(
        "[Background Task Notification] Process the completed "
        "background task results.",
        metadata={"is_background_result": True},
    )


async def _collect_next_background_batch(agent: BaseAgent) -> bool:
    from agent_cli.runtime import background

    return await background.collect_next_batch(agent)
