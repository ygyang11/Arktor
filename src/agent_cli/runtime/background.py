"""Runtime-layer wrapper around BackgroundTaskManager private access."""
from __future__ import annotations

import asyncio

from agent_harness.agent.base import BaseAgent
from agent_harness.background import BackgroundTask
from agent_harness.core.message import Message


def has_running(agent: BaseAgent) -> bool:
    return agent._bg_manager.has_running()


async def wait_next(agent: BaseAgent) -> None:
    await agent._bg_manager.wait_next()


async def collect_results(agent: BaseAgent) -> bool:
    return bool(await agent._collect_background_results())


def cancel_all(agent: BaseAgent) -> int:
    return agent._bg_manager.cancel_all()


async def shutdown(agent: BaseAgent) -> None:
    await agent._bg_manager.shutdown()


def get_all(agent: BaseAgent) -> list[BackgroundTask]:
    return agent._bg_manager.get_all()


def clear_tasks(agent: BaseAgent) -> None:
    agent._bg_manager._tasks.clear()


def is_current_task_background(agent: BaseAgent) -> bool:
    current = asyncio.current_task()
    if current is None:
        return False
    return any(bg.asyncio_task is current for bg in agent._bg_manager.get_all())


async def cancel_with_note(agent: BaseAgent, task_id: str) -> bool:
    task = agent._bg_manager.get_task(task_id)
    if task is None or task.status != "running":
        return False
    if not agent._bg_manager.cancel(task_id):
        return False
    await _emit_cancellation_note(agent, task)
    return True


async def cancel_all_with_note(agent: BaseAgent) -> list[str]:
    cancelled: list[str] = []
    for t in list(agent._bg_manager.get_all()):
        if t.status != "running":
            continue
        if not agent._bg_manager.cancel(t.task_id):
            continue
        cancelled.append(t.task_id)
        await _emit_cancellation_note(agent, t)
    return cancelled


async def _emit_cancellation_note(
    agent: BaseAgent, task: BackgroundTask,
) -> None:
    note = (
        f"[Background Task Cancelled] {task.task_id} ({task.tool_name}): "
        f"{task.description} — **Cancelled By User**"
    )
    await agent.context.short_term_memory.add_message(
        Message.system(note, metadata={"is_background_result": True}),
    )
