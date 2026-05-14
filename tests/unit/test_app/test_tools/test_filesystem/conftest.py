"""Shared fixtures for filesystem tool tests."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from agent_app.tools.filesystem.edit_file import edit_file
from agent_app.tools.filesystem.read_file import read_file
from agent_app.tools.filesystem.write_file import write_file
from agent_harness.agent.base import BaseAgent
from agent_harness.context.variables import ContextVariables

_PATCH_TARGETS = [
    "agent_app.tools.filesystem._security.get_workspace_root",
    "agent_app.tools.filesystem.list_dir.get_workspace_root",
    "agent_app.tools.filesystem.glob_files.get_workspace_root",
    "agent_app.tools.filesystem.grep_files.get_workspace_root",
]


@pytest.fixture(autouse=True)
def _workspace_root(tmp_path: Path) -> None:  # type: ignore[misc]
    patches = [patch(target, return_value=tmp_path) for target in _PATCH_TARGETS]
    for p in patches:
        p.start()
    yield
    for p in reversed(patches):
        p.stop()


class _StubContext:
    def __init__(self) -> None:
        self.variables = ContextVariables()


class _StubAgent:
    def __init__(self) -> None:
        self.context = _StubContext()


@pytest.fixture
def fs_agent() -> BaseAgent:
    return cast(BaseAgent, _StubAgent())


@pytest.fixture(autouse=True)
def _bind_filesystem_singletons(fs_agent: BaseAgent) -> None:
    read_file.bind_agent(fs_agent)
    edit_file.bind_agent(fs_agent)
    write_file.bind_agent(fs_agent)
