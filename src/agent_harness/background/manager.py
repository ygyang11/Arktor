"""Background task lifecycle management."""
from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TASK_SEQ = 0


def _next_task_id() -> str:
    global _TASK_SEQ
    _TASK_SEQ += 1
    return f"bg_{_TASK_SEQ:03d}"


@dataclass
class BackgroundResult:
    """Result of a completed background task."""

    summary: str
    output_path: str | None = None


@dataclass
class BackgroundTask:
    """A tracked background task."""

    task_id: str
    tool_name: str
    description: str
    asyncio_task: asyncio.Task[BackgroundResult] | None
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "running"
    result: BackgroundResult | None = None
    error: str | None = None
    _collected: bool = field(default=False, repr=False)


class BackgroundTaskManager:
    """Manages background asyncio tasks with lifecycle tracking."""

    _BASE_OUTPUT_DIR = ".agent-harness/background"

    def __init__(self) -> None:
        self._tasks: dict[str, BackgroundTask] = {}
        self._anonymous_dir = f"{self._BASE_OUTPUT_DIR}/{uuid.uuid4().hex[:8]}"
        self._output_dir = self._anonymous_dir
        self._cleanup_output_dir()

    def bind_session(self, session_id: str | None) -> None:
        """Bind to a session for output directory isolation.

        ``None`` falls back to this manager's per-instance anonymous
        directory (assigned at construction), so sessionless runs stay
        isolated from one another and never write into the shared base.
        """
        new_dir = (
            f"{self._BASE_OUTPUT_DIR}/{session_id}" if session_id
            else self._anonymous_dir
        )
        if new_dir == self._output_dir:
            return
        self._output_dir = new_dir

    def _cleanup_output_dir(self) -> None:
        out_dir = Path(self._output_dir)
        if out_dir.exists():
            shutil.rmtree(out_dir)

    def spawn(
        self,
        tool_name: str,
        description: str,
        coro: Any,
    ) -> str:
        """Spawn a coroutine as a background task. Returns task_id.

        The coroutine must return tuple[str, str] = (full_output, summary).
        """
        task_id = _next_task_id()
        bg_task = BackgroundTask(
            task_id=task_id,
            tool_name=tool_name,
            description=description,
            asyncio_task=None,
        )
        self._tasks[task_id] = bg_task
        bg_task.asyncio_task = asyncio.create_task(self._run(task_id, coro))
        logger.debug("Background task %s started: %s", task_id, description)
        return task_id

    async def _run(self, task_id: str, coro: Any) -> BackgroundResult:
        """Internal wrapper. Does NOT re-raise on failure (prevents asyncio warnings)."""
        task = self._tasks[task_id]
        try:
            output, summary = await coro
            output_path = self._write_output(task_id, output) if output else None
            task.result = BackgroundResult(summary=summary, output_path=output_path)
            task.status = "completed"
            logger.debug("Background task %s completed", task_id)
            return task.result
        except asyncio.CancelledError:
            coro.close()
            task.status = "cancelled"
            logger.debug("Background task %s cancelled", task_id)
            raise
        except Exception as e:
            task.result = BackgroundResult(summary=f"Error: {e}")
            task.error = str(e)
            task.status = "failed"
            logger.warning("Background task %s failed: %s", task_id, e)
            return task.result

    def collect_completed(self) -> list[BackgroundTask]:
        """Harvest completed/failed tasks not yet collected. Each returned only once."""
        ready: list[BackgroundTask] = []
        for task in self._tasks.values():
            if task.status in ("completed", "failed") and not task._collected:
                task._collected = True
                ready.append(task)
        return ready

    def has_running(self) -> bool:
        return any(t.status == "running" for t in self._tasks.values())

    async def wait_next(self) -> BackgroundTask | None:
        """Wait for the next running task to complete. Returns None if no running tasks."""
        running = [t for t in self._tasks.values() if t.status == "running" and t.asyncio_task]
        if not running:
            return None
        # asyncio.wait requires a set; each spawn creates a unique Task object, no dedup risk
        tasks_set = {t.asyncio_task for t in running if t.asyncio_task is not None}
        done, _ = await asyncio.wait(tasks_set, return_when=asyncio.FIRST_COMPLETED)
        for task in running:
            if task.asyncio_task in done:
                return task
        return None

    def get_task(self, task_id: str) -> BackgroundTask | None:
        return self._tasks.get(task_id)

    def get_all(self) -> list[BackgroundTask]:
        return list(self._tasks.values())

    def cancel(self, task_id: str) -> bool:
        """Cancel a running task. Returns True if cancelled."""
        task = self._tasks.get(task_id)
        if task and task.status == "running" and task.asyncio_task:
            task.asyncio_task.cancel()
            task.status = "cancelled"
            return True
        return False

    def cancel_all(self) -> int:
        """Cancel all running tasks. Returns count cancelled."""
        count = 0
        for task in self._tasks.values():
            if task.status == "running" and task.asyncio_task:
                task.asyncio_task.cancel()
                task.status = "cancelled"
                count += 1
        return count

    async def shutdown(self) -> None:
        """Cancel all incomplete tasks and wait for them to finish."""
        tasks_to_wait: list[asyncio.Task[BackgroundResult]] = []
        for task in self._tasks.values():
            if task.asyncio_task and not task.asyncio_task.done():
                task.asyncio_task.cancel()
                task.status = "cancelled"
                tasks_to_wait.append(task.asyncio_task)
        if tasks_to_wait:
            await asyncio.gather(*tasks_to_wait, return_exceptions=True)

    def get_running_summary(self) -> str | None:
        """Build brief status summary for LLM context (running tasks only)."""
        running = [t for t in self._tasks.values() if t.status == "running"]
        if not running:
            return None
        lines = [f"{len(running)} background task(s) running:"]
        for t in running:
            elapsed = (datetime.now() - t.created_at).total_seconds()
            lines.append(f"- {t.task_id} ({t.tool_name}): {t.description} [{elapsed:.0f}s]")
        lines.append(
            "\nOnly running tasks shown above. Use background_task(action='list') to see all previously submitted tasks and their status."
        )
        return "\n".join(lines)

    def _write_output(self, task_id: str, output: str) -> str:
        out_dir = Path(self._output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{task_id}.txt"
        path.write_text(output, encoding="utf-8")
        return str(path)
