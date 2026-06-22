"""Background task lifecycle management."""
from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HARNESS_DIR = Path.home() / ".arktor"


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
    output_dir: str = ""
    streaming: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "running"
    result: BackgroundResult | None = None
    error: str | None = None
    _collected: bool = field(default=False, repr=False)

    @property
    def log_path(self) -> str:
        return str(Path(self.output_dir) / f"{self.task_id}.txt")


class BackgroundTaskManager:
    """Manages background asyncio tasks with lifecycle tracking."""

    def __init__(self) -> None:
        self._tasks: dict[str, BackgroundTask] = {}
        self._anonymous_dir = str(
            _HARNESS_DIR / "anonymous" / "backgrounds" / uuid.uuid4().hex[:8]
        )
        self._output_dir = self._anonymous_dir
        self._task_seq = 0

    def bind_session(self, session_id: str | None) -> None:
        """Bind to a session for output directory isolation.

        ``None`` falls back to this manager's per-instance anonymous
        directory (assigned at construction), so sessionless runs stay
        isolated from one another and never write into the shared base.
        """
        new_dir = (
            str(_HARNESS_DIR / "sessions" / session_id / "backgrounds")
            if session_id
            else self._anonymous_dir
        )
        if new_dir == self._output_dir:
            return
        self._output_dir = new_dir
        self._restore_task_seq()

    def _next_task_id(self) -> str:
        self._task_seq += 1
        return f"bg_{self._task_seq:04d}"

    def _restore_task_seq(self) -> None:
        """When binding to a persisted session dir, advance the counter
        past any ``bg_*.txt`` artifacts from prior runs so we don't
        overwrite content that the persisted message history still
        references by path."""
        out = Path(self._output_dir)
        if not out.exists():
            return
        for p in out.glob("bg_*.txt"):
            try:
                n = int(p.stem.split("_", 1)[1])
            except (ValueError, IndexError):
                continue
            if n > self._task_seq:
                self._task_seq = n

    def _new_task(self, tool_name: str, description: str, *, streaming: bool) -> BackgroundTask:
        task = BackgroundTask(
            task_id=self._next_task_id(),
            tool_name=tool_name,
            description=description,
            asyncio_task=None,
            output_dir=self._output_dir,
            streaming=streaming,
        )
        self._tasks[task.task_id] = task
        return task

    def spawn(self, tool_name: str, description: str, coro: Any) -> str:
        """Spawn a ready coroutine (resolving to (output, summary)). Returns task_id."""
        task = self._new_task(tool_name, description, streaming=False)
        at = asyncio.create_task(self._run(task.task_id, coro))
        at.add_done_callback(lambda _t: coro.close())
        task.asyncio_task = at
        logger.debug("Background task %s started: %s", task.task_id, description)
        return task.task_id

    def spawn_streaming(
        self,
        tool_name: str,
        description: str,
        make_coro: Callable[[Path], Coroutine[Any, Any, tuple[str, str]]],
    ) -> str:
        """Spawn a streaming task. ``make_coro`` receives the task's log path."""
        task = self._new_task(tool_name, description, streaming=True)
        coro = make_coro(Path(task.log_path))
        at = asyncio.create_task(self._run(task.task_id, coro))
        at.add_done_callback(lambda _t: coro.close())
        task.asyncio_task = at
        logger.debug("Background task %s started: %s", task.task_id, description)
        return task.task_id

    async def _run(self, task_id: str, coro: Any) -> BackgroundResult:
        """Internal wrapper. Does NOT re-raise on failure (prevents asyncio warnings)."""
        task = self._tasks[task_id]
        log = Path(task.log_path)
        try:
            output, summary = await coro
            if not task.streaming and output:
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(output, encoding="utf-8")
            task.result = BackgroundResult(
                summary=summary,
                output_path=str(log) if log.exists() and log.stat().st_size > 0 else None,
            )
            task.status = "completed"
            logger.debug("Background task %s completed", task_id)
            return task.result
        except asyncio.CancelledError:
            task.status = "cancelled"
            logger.debug("Background task %s cancelled", task_id)
            raise
        except Exception as e:
            task.result = BackgroundResult(
                summary=f"Error: {e}",
                output_path=str(log) if log.exists() and log.stat().st_size > 0 else None,
            )
            task.error = str(e)
            task.status = "failed"
            logger.debug("Background task %s failed: %s", task_id, e)
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
