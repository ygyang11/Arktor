"""Docker sandbox backend — container-isolated command execution."""
from __future__ import annotations

import asyncio
import atexit
import logging
from pathlib import Path
from typing import Any

from agent_harness.core.config import DockerConfig
from agent_harness.sandbox.backend import ExecuteResult, SandboxBackend

logger = logging.getLogger(__name__)

try:
    import docker as _docker_lib
except ImportError:
    _docker_lib = None

_CONTAINER_WORKDIR = "/workspace"
_STOP_TIMEOUT = 5


class DockerBackend(SandboxBackend):
    """Docker sandbox backend.

    Lifecycle:
      start()   — Create container (mount workspace, run setup command)
      execute() — exec_create + exec_start (millisecond-level, container stays running)
      stop()    — Stop and remove container

    Timeout/cancel semantics:
      Uses Docker API low-level exec_create/exec_start to obtain an exec_id.
      On timeout or cancellation, exec_inspect retrieves the container PID,
      then kill -9 -PID terminates the entire process group.
    """

    def __init__(self, config: DockerConfig) -> None:
        self._config = config
        self._workspace: str = ""
        self._container: Any = None
        self._client: Any = None

    async def start(self, workspace: str | None = None) -> None:
        """Create and start a Docker container."""
        self._workspace = workspace or str(Path.cwd().resolve())

        if _docker_lib is None:
            raise ImportError(
                "Docker backend requires the 'docker' package.\n"
                "Install it with: pip install arktor[sandbox]"
            )

        loop = asyncio.get_running_loop()

        def _create() -> tuple[Any, Any]:
            client = _docker_lib.from_env()

            volumes: dict[str, dict[str, str]] = {
                self._workspace: {"bind": _CONTAINER_WORKDIR, "mode": "rw"},
            }
            for vol in self._config.volumes:
                parts = vol.split(":")
                if len(parts) >= 2:
                    host_path = parts[0]
                    container_path = parts[1]
                    mode = parts[2] if len(parts) > 2 else "rw"
                    volumes[host_path] = {"bind": container_path, "mode": mode}

            kwargs: dict[str, Any] = {}
            if self._config.memory:
                kwargs["mem_limit"] = self._config.memory
            if self._config.cpus > 0:
                kwargs["nano_cpus"] = int(self._config.cpus * 1e9)

            container = client.containers.run(
                self._config.image,
                "sleep infinity",
                detach=True,
                working_dir=_CONTAINER_WORKDIR,
                network_mode=self._config.network,
                volumes=volumes,
                **kwargs,
            )
            return client, container

        self._client, self._container = await loop.run_in_executor(None, _create)
        atexit.register(self._sync_cleanup)
        logger.info("Sandbox started")
        logger.debug(
            "Container %s: image=%s, network=%s, workspace=%s",
            self._container.short_id,
            self._config.image,
            self._config.network,
            self._workspace,
        )

        if self._config.setup:
            logger.info("Running setup command...")
            result = await self.execute(self._config.setup, timeout=self._config.setup_timeout)
            if result.exit_code != 0:
                await self.stop()
                raise RuntimeError(
                    f"Sandbox setup failed (exit {result.exit_code}):\n{result.stdout}"
                )
            logger.info("Setup completed")

    async def execute(
        self,
        command: str,
        *,
        timeout: float = 30.0,
        workdir: str | None = None,
    ) -> ExecuteResult:
        """Execute a command inside the container."""
        if self._container is None:
            raise RuntimeError("Docker sandbox not started — call start() first")

        container_workdir = _CONTAINER_WORKDIR
        if workdir:
            host_workspace = Path(self._workspace).resolve()
            target = Path(workdir).resolve()
            try:
                rel = target.relative_to(host_workspace)
                container_workdir = (
                    f"{_CONTAINER_WORKDIR}/{rel}" if str(rel) != "." else _CONTAINER_WORKDIR
                )
            except ValueError:
                logger.warning(
                    "workdir '%s' is outside sandbox workspace '%s', using default",
                    workdir,
                    self._workspace,
                )

        loop = asyncio.get_running_loop()
        container = self._container
        api_client = self._client.api

        def _create_exec() -> str:
            resp = api_client.exec_create(
                container.id,
                ["bash", "-c", command],
                workdir=container_workdir,
                stdout=True,
                stderr=True,
            )
            return str(resp["Id"])

        exec_id: str = await loop.run_in_executor(None, _create_exec)

        def _run_exec() -> tuple[bytes, bytes]:
            stdout, stderr = api_client.exec_start(exec_id, demux=True)
            return stdout or b"", stderr or b""

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                loop.run_in_executor(None, _run_exec),
                timeout=timeout,
            )
        except TimeoutError:
            await self._kill_exec(exec_id)
            return ExecuteResult(
                exit_code=None,
                stdout=f"Error: execution timed out after {timeout}s",
            )
        except asyncio.CancelledError:
            await self._kill_exec(exec_id)
            raise
        except Exception as exc:  # noqa: BLE001
            return ExecuteResult(
                exit_code=None,
                stdout=f"Error: command execution failed: {exc}",
            )

        def _inspect() -> int:
            info = api_client.exec_inspect(exec_id)
            return int(info.get("ExitCode", -1))

        exit_code = await loop.run_in_executor(None, _inspect)

        stdout_text = stdout_bytes.decode(errors="replace")
        stderr_text = stderr_bytes.decode(errors="replace")

        return ExecuteResult(exit_code=exit_code, stdout=stdout_text, stderr=stderr_text)

    async def _kill_exec(self, exec_id: str) -> None:
        """Kill the exec process group inside the container."""
        loop = asyncio.get_running_loop()
        container = self._container
        api_client = self._client.api

        def _do_kill() -> None:
            try:
                info = api_client.exec_inspect(exec_id)
                pid = info.get("Pid", 0)
                if pid > 0:
                    container.exec_run(
                        ["bash", "-c", f"kill -9 -{pid} 2>/dev/null; kill -9 {pid} 2>/dev/null"]
                    )
                    logger.debug(
                        "Killed process group: pid=%d, exec_id=%s", pid, exec_id[:12]
                    )
            except Exception:  # noqa: BLE001
                logger.warning("Failed to kill exec process: exec_id=%s", exec_id[:12])

        await loop.run_in_executor(None, _do_kill)

    async def stop(self) -> None:
        """Stop and remove the container."""
        if self._container is None:
            return

        loop = asyncio.get_running_loop()
        container = self._container
        self._container = None

        def _cleanup() -> None:
            try:
                container.stop(timeout=_STOP_TIMEOUT)
            except Exception:  # noqa: BLE001
                try:
                    container.kill()
                except Exception:  # noqa: BLE001
                    pass
            try:
                container.remove(force=True)
            except Exception:  # noqa: BLE001
                pass

        await loop.run_in_executor(None, _cleanup)
        logger.info("Sandbox stopped")

    def _sync_cleanup(self) -> None:
        """Synchronous cleanup for atexit — process exit fallback."""
        if self._container:
            try:
                self._container.stop(timeout=_STOP_TIMEOUT)
                self._container.remove(force=True)
            except Exception:  # noqa: BLE001
                pass
            self._container = None
