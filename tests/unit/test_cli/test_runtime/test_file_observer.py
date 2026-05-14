"""Tests for the CLI file_observer ContextPatch."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

from agent_app.observability.file_freshness import record_signature
from agent_cli.runtime import file_observer
from agent_cli.runtime.file_observer import _format_notice, _patch_for, enable
from agent_harness.agent.base import BaseAgent
from agent_harness.context.context import AgentContext


def _agent() -> BaseAgent:
    agent = MagicMock()
    agent.context = AgentContext()
    return cast(BaseAgent, agent)


def teardown_function() -> None:
    file_observer._patch_for.cache_clear()


def test_enable_appends_one_patch() -> None:
    agent = _agent()
    enable(agent)
    assert len(agent.context.context_patches) == 1
    assert agent.context.context_patches[0].at == "tail"


def test_enable_is_idempotent() -> None:
    agent = _agent()
    enable(agent)
    enable(agent)
    assert len(agent.context.context_patches) == 1


def test_enable_reattaches_after_clear() -> None:
    agent = _agent()
    enable(agent)
    agent.context.context_patches.clear()
    enable(agent)
    assert len(agent.context.context_patches) == 1


def test_build_returns_none_when_no_drift(tmp_path: Path) -> None:
    agent = _agent()
    enable(agent)
    f = tmp_path / "a.txt"
    f.write_text("x")
    record_signature(agent, f)
    assert agent.context.context_patches[0].build() is None


def test_build_returns_message_with_each_drifted_path(tmp_path: Path) -> None:
    agent = _agent()
    enable(agent)
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    for p in (a, b):
        p.write_text("x")
        record_signature(agent, p)
    future = time.time() + 10
    os.utime(b, (future, future))

    msg = agent.context.context_patches[0].build()
    assert msg is not None
    content = msg.content or ""
    assert "b.txt" in content
    assert "a.txt" not in content


def test_build_marks_modified_files(tmp_path: Path) -> None:
    agent = _agent()
    f = tmp_path / "code.py"
    f.write_text("x")
    record_signature(agent, f)
    future = time.time() + 10
    os.utime(f, (future, future))

    patch = _patch_for(agent)
    msg = patch.build()
    assert msg is not None
    assert "(modified)" in (msg.content or "")


def test_build_marks_deleted_files(tmp_path: Path) -> None:
    agent = _agent()
    f = tmp_path / "code.py"
    f.write_text("x")
    record_signature(agent, f)
    os.unlink(f)

    patch = _patch_for(agent)
    msg = patch.build()
    assert msg is not None
    assert "(deleted)" in (msg.content or "")


def test_build_detects_same_size_content_change(tmp_path: Path) -> None:
    agent = _agent()
    f = tmp_path / "code.py"
    f.write_text("aaa")
    record_signature(agent, f)
    future = time.time() + 10
    os.utime(f, (future, future))
    f.write_text("bbb")
    future = time.time() + 20
    os.utime(f, (future, future))

    patch = _patch_for(agent)
    msg = patch.build()
    assert msg is not None
    assert "(modified)" in (msg.content or "")


def test_notice_contains_system_reminder_tag(tmp_path: Path) -> None:
    agent = _agent()
    f = tmp_path / "a.txt"
    f.write_text("x")
    record_signature(agent, f)
    os.unlink(f)

    patch = _patch_for(agent)
    msg = patch.build()
    assert msg is not None
    content = msg.content or ""
    assert content.startswith("<system-reminder>")
    assert content.rstrip().endswith("</system-reminder>")
    assert "re-read it with read_file" in content


def test_notice_uses_workspace_relative_paths(tmp_path: Path) -> None:
    agent = _agent()
    sub = tmp_path / "src"
    sub.mkdir()
    f = sub / "foo.py"
    f.write_text("x")
    record_signature(agent, f)
    os.unlink(f)

    with patch(
        "agent_app.tools.filesystem._security.get_workspace_root",
        return_value=tmp_path,
    ):
        text = _format_notice([d for d in [_drift_for(agent, f)] if d is not None])
    assert "src/foo.py" in text
    assert str(tmp_path) not in text


def _drift_for(agent: BaseAgent, p: Path):  # type: ignore[no-untyped-def]
    from agent_app.observability.file_freshness import poll_dirty
    drifts = poll_dirty(agent)
    for d in drifts:
        if Path(d.path) == p.resolve():
            return d
    return None
