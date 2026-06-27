"""Tests for DockerBackend — mock docker-py to avoid real Docker dependency."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_harness.core.config import DockerConfig
from agent_harness.sandbox.docker import _CONTAINER_WORKDIR, DockerBackend


class TestDockerBackend:
    @pytest.fixture
    def config(self) -> DockerConfig:
        return DockerConfig(image="python:3.11-slim", network="none")

    @pytest.fixture
    def backend(self, config: DockerConfig) -> DockerBackend:
        return DockerBackend(config)

    def _make_started_backend(
        self, config: DockerConfig | None = None, tmp_path: str = "/tmp/test",
    ) -> DockerBackend:
        """Create a backend with mocked container and client, bypassing start()."""
        backend = DockerBackend(config or DockerConfig())
        backend._workspace = tmp_path
        mock_container = MagicMock()
        mock_container.id = "container_abc"
        mock_container.short_id = "abc123"
        mock_api = MagicMock()
        mock_client = MagicMock()
        mock_client.api = mock_api
        backend._container = mock_container
        backend._client = mock_client
        return backend

    async def test_start_creates_container(self, backend: DockerBackend, tmp_path: Path) -> None:
        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container

        with patch("agent_harness.sandbox.docker._docker_lib") as mock_docker:
            mock_docker.from_env.return_value = mock_client
            await backend.start(workspace=str(tmp_path))

        mock_client.containers.run.assert_called_once()
        call_kwargs = mock_client.containers.run.call_args
        assert call_kwargs[0][0] == "python:3.11-slim"
        assert call_kwargs[1]["network_mode"] == "none"

    async def test_execute_uses_exec_create_start(self, backend: DockerBackend) -> None:
        b = self._make_started_backend()
        b._client.api.exec_create.return_value = {"Id": "exec_123"}
        b._client.api.exec_start.return_value = (b"hello\n", b"")
        b._client.api.exec_inspect.return_value = {"ExitCode": 0}

        result = await b.execute("echo hello")
        assert result.exit_code == 0
        assert "hello" in result.stdout
        b._client.api.exec_create.assert_called_once()
        b._client.api.exec_start.assert_called_once()

    async def test_stdout_stderr_separated(self, backend: DockerBackend) -> None:
        b = self._make_started_backend()
        b._client.api.exec_create.return_value = {"Id": "exec_sep"}
        b._client.api.exec_start.return_value = (b"out\n", b"err\n")
        b._client.api.exec_inspect.return_value = {"ExitCode": 0}

        result = await b.execute("cmd")
        assert "out" in result.stdout
        assert "err" in result.stderr

    async def test_stop_removes_container(self, backend: DockerBackend) -> None:
        mock_container = MagicMock()
        backend._container = mock_container

        await backend.stop()
        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()
        assert backend._container is None

    async def test_setup_command(self, tmp_path: Path) -> None:
        config = DockerConfig(setup="pip install pytest")
        backend = DockerBackend(config)

        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        mock_container.id = "container_setup"
        mock_api = MagicMock()
        mock_api.exec_create.return_value = {"Id": "exec_setup"}
        mock_api.exec_start.return_value = (b"installed\n", b"")
        mock_api.exec_inspect.return_value = {"ExitCode": 0}
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_client.api = mock_api

        with patch("agent_harness.sandbox.docker._docker_lib") as mock_docker:
            mock_docker.from_env.return_value = mock_client
            await backend.start(workspace=str(tmp_path))

        mock_api.exec_create.assert_called()
        mock_api.exec_start.assert_called()

    async def test_setup_failure_stops_container(self, tmp_path: Path) -> None:
        config = DockerConfig(setup="bad_command")
        backend = DockerBackend(config)

        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        mock_container.id = "container_fail"
        mock_api = MagicMock()
        mock_api.exec_create.return_value = {"Id": "exec_fail"}
        mock_api.exec_start.return_value = (b"error\n", b"")
        mock_api.exec_inspect.return_value = {"ExitCode": 1}
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_client.api = mock_api

        with patch("agent_harness.sandbox.docker._docker_lib") as mock_docker:
            mock_docker.from_env.return_value = mock_client
            with pytest.raises(RuntimeError, match="setup failed"):
                await backend.start(workspace=str(tmp_path))

    async def test_timeout_kills_process_group(self) -> None:
        b = self._make_started_backend()
        b._client.api.exec_create.return_value = {"Id": "exec_timeout"}
        b._client.api.exec_start.side_effect = lambda *a, **k: __import__("time").sleep(10)
        b._client.api.exec_inspect.return_value = {"Pid": 12345, "ExitCode": -1}
        b._container.exec_run.return_value = (0, b"")

        result = await b.execute("sleep 100", timeout=0.1)
        assert result.exit_code is None
        assert "timed out" in result.stdout
        b._container.exec_run.assert_called()
        kill_call_args = str(b._container.exec_run.call_args)
        assert "12345" in kill_call_args

    async def test_cancel_kills_process(self) -> None:
        b = self._make_started_backend()
        b._client.api.exec_create.return_value = {"Id": "exec_cancel"}
        b._client.api.exec_start.side_effect = lambda *a, **k: __import__("time").sleep(10)
        b._client.api.exec_inspect.return_value = {"Pid": 99999}
        b._container.exec_run.return_value = (0, b"")

        task = asyncio.create_task(b.execute("sleep 100", timeout=60))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        b._container.exec_run.assert_called()

    async def test_command_wrapped_with_pgid_marker(self) -> None:
        b = self._make_started_backend()
        b._client.api.exec_create.return_value = {"Id": "exec_w"}
        b._client.api.exec_start.return_value = (b"", b"")
        b._client.api.exec_inspect.return_value = {"ExitCode": 0, "Pid": 0}

        await b.execute("echo hi  # trailing comment")
        wrapped = b._client.api.exec_create.call_args[0][1][2]
        assert "printf" in wrapped and '"$$"' in wrapped
        assert "/tmp/.arktor-pg-" in wrapped
        assert wrapped.endswith("echo hi  # trailing comment")

    async def test_completion_sweeps_process_group(self) -> None:
        b = self._make_started_backend()
        b._client.api.exec_create.return_value = {"Id": "exec_done"}
        b._client.api.exec_start.return_value = (b"done\n", b"")
        b._client.api.exec_inspect.return_value = {"ExitCode": 0, "Pid": 0}

        result = await b.execute("echo done")
        assert result.exit_code == 0
        b._container.exec_run.assert_called()
        kill = str(b._container.exec_run.call_args)
        assert 'kill -9 -- "-$pgid"' in kill and "/tmp/.arktor-pg-" in kill

    async def test_workdir_mapping(self, tmp_path: Path) -> None:
        b = self._make_started_backend(tmp_path=str(tmp_path))
        b._client.api.exec_create.return_value = {"Id": "exec_wd"}
        b._client.api.exec_start.return_value = (b"/workspace/src\n", b"")
        b._client.api.exec_inspect.return_value = {"ExitCode": 0}

        subdir = tmp_path / "src"
        subdir.mkdir()
        await b.execute("pwd", workdir=str(subdir))

        create_call = b._client.api.exec_create.call_args
        assert create_call[1]["workdir"] == f"{_CONTAINER_WORKDIR}/src"

    async def test_volumes_config(self, tmp_path: Path) -> None:
        config = DockerConfig(volumes=["/data:/data:ro", "/tmp:/tmp"])
        backend = DockerBackend(config)

        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container

        with patch("agent_harness.sandbox.docker._docker_lib") as mock_docker:
            mock_docker.from_env.return_value = mock_client
            await backend.start(workspace=str(tmp_path))

        call_kwargs = mock_client.containers.run.call_args[1]
        volumes = call_kwargs["volumes"]
        assert "/data" in volumes
        assert volumes["/data"]["mode"] == "ro"

    async def test_resource_limits(self, tmp_path: Path) -> None:
        config = DockerConfig(memory="512m", cpus=1.5)
        backend = DockerBackend(config)

        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container

        with patch("agent_harness.sandbox.docker._docker_lib") as mock_docker:
            mock_docker.from_env.return_value = mock_client
            await backend.start(workspace=str(tmp_path))

        call_kwargs = mock_client.containers.run.call_args[1]
        assert call_kwargs["mem_limit"] == "512m"
        assert call_kwargs["nano_cpus"] == 1_500_000_000

    def test_sync_cleanup_stops_container(self) -> None:
        mock_container = MagicMock()
        backend = DockerBackend(DockerConfig())
        backend._container = mock_container

        backend._sync_cleanup()

        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()
        assert backend._container is None

    def test_sync_cleanup_noop_when_no_container(self) -> None:
        backend = DockerBackend(DockerConfig())
        backend._sync_cleanup()
