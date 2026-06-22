"""Sandbox backend base class and local (passthrough) implementation."""
from __future__ import annotations

import asyncio
import codecs
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_STREAM_CHUNK_SIZE = 4096


@dataclass
class ExecuteResult:
    """Result of a sandboxed command execution."""

    exit_code: int | None
    stdout: str
    stderr: str = ""


class SandboxBackend(ABC):
    """Base class for sandbox backends.

    Built-in: LocalBackend (passthrough), DockerBackend (container isolation).
    Extend this class to integrate third-party sandbox providers.
    """

    async def start(self, workspace: str | None = None) -> None:
        """Initialize the backend (e.g. create Docker container)."""

    @abstractmethod
    async def execute(
        self,
        command: str,
        *,
        timeout: float = 30.0,
        workdir: str | None = None,
        stream_to: Path | None = None,
    ) -> ExecuteResult:
        """Execute a command in the sandbox."""

    async def stop(self) -> None:
        """Clean up backend resources (e.g. destroy container)."""


class LocalBackend(SandboxBackend):
    """Passthrough backend — executes directly on the host, no isolation.

    Output is returned as raw stdout/stderr without truncation.
    Callers (terminal_tool) handle merging and truncation.
    """

    async def execute(
        self,
        command: str,
        *,
        timeout: float = 30.0,
        workdir: str | None = None,
        stream_to: Path | None = None,
    ) -> ExecuteResult:
        if stream_to is not None:
            return await self._execute_streaming(
                command, timeout=timeout, workdir=workdir, sink=stream_to
            )
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                cwd=workdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            return ExecuteResult(
                exit_code=None, stdout=f"Error: execution timed out after {timeout}s",
            )
        except asyncio.CancelledError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            raise
        except Exception as exc:  # noqa: BLE001
            return ExecuteResult(exit_code=None, stdout=f"Error: failed to execute command: {exc}")

        stdout_text = stdout_bytes.decode(errors="replace")
        stderr_text = stderr_bytes.decode(errors="replace")

        return ExecuteResult(
            exit_code=proc.returncode,
            stdout=stdout_text,
            stderr=stderr_text,
        )

    async def _execute_streaming(
        self,
        command: str,
        *,
        timeout: float,
        workdir: str | None,
        sink: Path,
    ) -> ExecuteResult:
        sink.parent.mkdir(parents=True, exist_ok=True)
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                cwd=workdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except Exception as exc:  # noqa: BLE001
            return ExecuteResult(exit_code=None, stdout=f"Error: failed to execute command: {exc}")

        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        async def _drive() -> int | None:
            assert proc.stdout is not None
            with sink.open("w", encoding="utf-8") as f:
                while True:
                    data = await proc.stdout.read(_STREAM_CHUNK_SIZE)
                    if not data:
                        break
                    text = decoder.decode(data)
                    if text:
                        f.write(text)
                        f.flush()
                tail = decoder.decode(b"", final=True)
                if tail:
                    f.write(tail)
                    f.flush()
            return await proc.wait()

        try:
            exit_code = await asyncio.wait_for(_drive(), timeout=timeout)
        except TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            return ExecuteResult(
                exit_code=None, stdout=f"Error: execution timed out after {timeout}s"
            )
        except asyncio.CancelledError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            raise
        except Exception as exc:  # noqa: BLE001
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            return ExecuteResult(exit_code=None, stdout=f"Error: command execution failed: {exc}")

        return ExecuteResult(exit_code=exit_code, stdout="", stderr="")
