"""Tests for RuntimeContextProvider."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_harness.core.message import Role
from agent_harness.prompt.runtime_context import RuntimeContextProvider


class TestBuildContextMessage:
    def test_returns_system_message(self) -> None:
        provider = RuntimeContextProvider()
        msg = provider.build_context_message()
        assert msg is not None
        assert msg.role == Role.SYSTEM

    def test_contains_environment_header(self) -> None:
        msg = RuntimeContextProvider().build_context_message()
        assert msg is not None
        assert "# Environment" in msg.content

    def test_contains_working_directory(self) -> None:
        msg = RuntimeContextProvider().build_context_message()
        assert msg is not None
        assert "Primary working directory" in msg.content

    def test_contains_platform(self) -> None:
        msg = RuntimeContextProvider().build_context_message()
        assert msg is not None
        assert "Platform" in msg.content

    def test_contains_date(self) -> None:
        msg = RuntimeContextProvider().build_context_message()
        assert msg is not None
        assert "Current date" in msg.content

    def test_date_is_local_not_utc(self) -> None:
        from datetime import datetime

        msg = RuntimeContextProvider().build_context_message()
        assert msg is not None
        today = datetime.now().strftime("%Y-%m-%d")
        assert today in msg.content

    def test_contains_git_repo_flag(self) -> None:
        msg = RuntimeContextProvider().build_context_message()
        assert msg is not None
        # Test runs from project root which is a git repo
        assert "Is a git repository: True" in msg.content

    def test_git_failure_silent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        provider = RuntimeContextProvider()
        msg = provider.build_context_message()
        assert msg is not None
        assert "# Environment" in msg.content
        assert "Is a git repository: False" in msg.content


class TestGitHelpers:
    def test_is_git_repo_true(self) -> None:
        # Test runs from project root
        assert RuntimeContextProvider._is_git_repo(".") is True

    def test_is_git_repo_false(self, tmp_path: Path) -> None:
        assert RuntimeContextProvider._is_git_repo(str(tmp_path)) is False
