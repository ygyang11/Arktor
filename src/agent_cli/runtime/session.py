"""Runtime-layer wrappers around session state mutation."""
from __future__ import annotations

import argparse
import logging
import sys
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from agent_app.observability import file_freshness
from agent_cli.runtime import background, plan_mode
from agent_harness.agent.base import BaseAgent
from agent_harness.approval.policy import ApprovalPolicy
from agent_harness.core.message import Message, Role
from agent_harness.llm import BaseLLM
from agent_harness.session.base import BaseSession, SessionState
from agent_harness.session.file_session import FileSession

if TYPE_CHECKING:
    from agent_cli.approval_handler import CliApprovalHandler

logger = logging.getLogger(__name__)

SaveSession = Callable[[], Awaitable[None]]

_MODE_CYCLE: tuple[str, ...] = ("auto", "ask", "never", "plan")


def current_mode_key(agent: BaseAgent) -> str:
    if plan_mode.is_active(agent):
        return "plan"
    return agent._approval.mode


def apply_mode(agent: BaseAgent, target: str) -> None:
    if target == "plan":
        plan_mode.enter(agent)
        return
    agent._approval.set_mode(target)


def cycle_next_mode(current: str) -> str:
    if current not in _MODE_CYCLE:
        return _MODE_CYCLE[0]
    return _MODE_CYCLE[(_MODE_CYCLE.index(current) + 1) % len(_MODE_CYCLE)]


def get_policy(agent: BaseAgent) -> ApprovalPolicy:
    return agent._approval


def reset_approval(agent: BaseAgent) -> None:
    if agent._approval is not None:
        agent._approval.reset_session()


def reset_stateful_tools(agent: BaseAgent) -> None:
    agent._reset_stateful_tools()


def export_approval_grants(agent: BaseAgent) -> dict[str, Any] | None:
    if agent._approval is None:
        return None
    return agent._approval.export_session_grants()


async def stop_sandbox(agent: BaseAgent) -> None:
    await agent._sandbox.stop()


def get_messages(agent: BaseAgent) -> list[Message]:
    return list(agent.context.short_term_memory._messages)


def set_messages(agent: BaseAgent, messages: list[Message]) -> None:
    agent.context.short_term_memory.replace_messages(messages)


def update_compressor_model(
    agent: BaseAgent, new_model: str, new_llm: BaseLLM,
) -> None:
    compressor = agent.context.short_term_memory.compressor
    cfg = agent.context.config
    if compressor is not None and cfg.memory.compression.summary_model is None:
        compressor._model = new_model
        compressor._llm = new_llm


async def resolve_session_id(
    args: argparse.Namespace, probe: FileSession,
) -> str | None:
    """Resolve the startup session id from CLI flags. Returns the id, or
    ``None`` on failure (error already printed to stderr; caller exits 2)."""
    new_id: str | None = args.session_id
    if new_id:
        if not probe._is_valid_id(new_id):
            print(
                f"arktor: invalid session id (allowed: [a-zA-Z0-9_-]): {new_id}",
                file=sys.stderr,
            )
            return None
        if await probe.has_session(new_id):
            print(
                f"arktor: session id already exists: {new_id} "
                f"(use --resume to resume it)",
                file=sys.stderr,
            )
            return None
        return new_id
    resume_id: str | None = args.resume
    if resume_id:
        if not await probe.has_session(resume_id):
            print(f"arktor: session not found: {resume_id}", file=sys.stderr)
            return None
        return resume_id
    if args.resume_latest:
        # Limitation: list_states() silently drops unparseable files. If the
        # most-recently-saved session got corrupted, -c falls back to the
        # next readable one without surfacing it. Use -r <id> if you need
        # explicit corruption reporting on a specific session.
        metas = await probe.list_states()
        if not metas:
            print("arktor: no prior session found", file=sys.stderr)
            return None
        return metas[0].session_id
    return str(uuid.uuid4())


async def restore_session(
    agent: BaseAgent, backend: BaseSession,
) -> SessionState | None:
    state = await backend.load_state()
    plan_mode.exit(agent)
    if state is None:
        return None
    await agent.apply_session_state(state)
    if state.metadata.get("_plan_mode"):
        plan_mode.enter(agent)
    return state


def make_save_session(
    agent: BaseAgent, backend: BaseSession,
) -> SaveSession:
    """Build a no-arg async closure that snapshots + persists the current session."""
    async def _save() -> None:
        now = datetime.now()
        ss = agent.context.to_session_state(backend.session_id, agent_name=agent.name)
        ss.created_at = agent._ensure_session_created_at()
        ss.updated_at = now
        ss.metadata.update(agent._session_metadata_extras)
        tool_states = agent.tool_registry.save_states()
        if tool_states:
            ss.metadata["_tool_states"] = tool_states
        ss.metadata["_approval_mode"] = agent._approval.mode
        grants = export_approval_grants(agent)
        if grants:
            ss.metadata["_approval_grants"] = grants
        await backend.save_state(ss)

    return _save


async def switch_session(
    agent: BaseAgent,
    backend: BaseSession,
    handler: CliApprovalHandler,
    save: SaveSession,
    new_id: str,
) -> None:
    handler.cancel_pending()
    # Drain freshly-completed bg results into current-session transcript and
    # persist before tear-down, so results that landed between turns are
    # not silently dropped when the user switches.
    collected = await agent._collect_background_results()
    if collected:
        await save()
    await background.shutdown(agent)
    background.clear_tasks(agent)
    await stop_sandbox(agent)

    backend.set_session_id(new_id)
    if await restore_session(agent, backend) is None:
        await agent.reset_session_state(new_id)


async def rename_session(
    agent: BaseAgent, backend: BaseSession, new_id: str,
) -> None:
    """Rename the live session in place, keeping the conversation intact."""
    await backend.rename(new_id)
    agent._bind_session_id(new_id)


# ── Turn lifecycle (cancel-rollback) ──


@dataclass
class _TurnContext:
    snapshot_messages: list[Message]
    snapshot_compressor_state: tuple[int, list[str]] | None
    snapshot_ids: frozenset[int]
    main_system_id: int | None
    fs_state: dict[str, Any]
    committed: bool = False


def take_snapshot(agent: BaseAgent) -> _TurnContext:
    originals = get_messages(agent)

    main_system_id: int | None = None
    if agent.system_prompt:
        for m in originals:
            if m.role == Role.SYSTEM and m.content == agent.system_prompt:
                main_system_id = id(m)
                break

    compressor_state: tuple[int, list[str]] | None = None
    compressor = agent.context.short_term_memory.compressor
    if compressor is not None:
        compressor_state = (
            compressor._compression_count,
            list(compressor._archive_paths),
        )

    return _TurnContext(
        snapshot_messages=[m.model_copy(deep=True) for m in originals],
        snapshot_compressor_state=compressor_state,
        snapshot_ids=frozenset(id(m) for m in originals),
        main_system_id=main_system_id,
        fs_state=file_freshness.snapshot_state(agent),
    )


def transcript_changed(ctx: _TurnContext, current: list[Message]) -> bool:
    relevant = ctx.snapshot_ids
    if ctx.main_system_id is not None:
        relevant = relevant - {ctx.main_system_id}
    current_ids = {id(m) for m in current}
    return not relevant.issubset(current_ids)


def should_rollback(ctx: _TurnContext, current: list[Message]) -> bool:
    if ctx.committed:
        return False
    return not transcript_changed(ctx, current)


async def rollback(
    agent: BaseAgent,
    ctx: _TurnContext,
    save: SaveSession,
) -> None:
    bg_added = [
        m for m in get_messages(agent)
        if id(m) not in ctx.snapshot_ids
        and m.metadata.get("is_background_result")
    ]
    set_messages(agent, list(ctx.snapshot_messages) + bg_added)
    file_freshness.restore_state(agent, ctx.fs_state)

    compressor = agent.context.short_term_memory.compressor
    if ctx.snapshot_compressor_state is not None and compressor is not None:
        count, archives = ctx.snapshot_compressor_state
        compressor._compression_count = count
        compressor._archive_paths = list(archives)
        compressor._last_result = None

    try:
        await save()
    except Exception as e:
        logger.debug("rollback: save() failed; disk left in pre-rollback state: %s", e)
