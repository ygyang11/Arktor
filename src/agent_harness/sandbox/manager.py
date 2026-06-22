"""Sandbox lifecycle manager — backend selection and lifecycle."""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from agent_harness.core.config import SandboxConfig
from agent_harness.sandbox.backend import LocalBackend, SandboxBackend

if TYPE_CHECKING:
    from pathlib import Path

    from agent_harness.sandbox.backend import ExecuteResult

logger = logging.getLogger(__name__)


class SandboxManager:
    """Manage sandbox backend lifecycle.

    Constructor takes a backend directly. Use from_config() for config-driven
    backend selection.
    """

    def __init__(self, backend: SandboxBackend) -> None:
        self._backend = backend
        self._started = False
        self._start_lock = asyncio.Lock()

    @classmethod
    def from_config(cls, config: SandboxConfig) -> SandboxManager:
        """Create a SandboxManager based on configuration.

        enabled=True  -> DockerBackend
        enabled=False -> LocalBackend (passthrough)
        """
        if config.enabled:
            from agent_harness.sandbox.docker import DockerBackend

            return cls(DockerBackend(config.docker))
        return cls(LocalBackend())

    @property
    def backend(self) -> SandboxBackend:
        return self._backend

    @property
    def is_sandboxed(self) -> bool:
        """Whether using a real isolation backend (not Local)."""
        return self._started and not isinstance(self._backend, LocalBackend)

    async def start(self, workspace: str | None = None) -> None:
        """Initialize and start the backend. Idempotent."""
        if self._started:
            return
        await self._backend.start(workspace=workspace)
        self._started = True

    async def stop(self) -> None:
        """Stop and clean up the backend."""
        if self._started:
            await self._backend.stop()
            self._started = False

    async def execute(
        self,
        command: str,
        *,
        timeout: float = 30.0,
        workdir: str | None = None,
        stream_to: Path | None = None,
    ) -> ExecuteResult:
        """Delegate to the current backend. Lazy-starts on first call."""
        if not self._started:
            async with self._start_lock:
                if not self._started:
                    try:
                        await self.start(workspace=workdir)
                    except Exception as e:
                        raise RuntimeError(
                            f"Sandbox failed to initialize — terminal tool is unavailable. "
                            f"Cause: {e}"
                        ) from e
        if stream_to is None:
            return await self._backend.execute(command, timeout=timeout, workdir=workdir)
        return await self._backend.execute(
            command, timeout=timeout, workdir=workdir, stream_to=stream_to
        )
