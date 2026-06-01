"""Tests for the CLI file_observer drift reminder."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

from agent_app.observability.file_freshness import mark_read, poll_drift
from agent_cli.runtime import file_observer
from agent_cli.runtime.file_observer import _format_notice
from agent_harness.agent.base import BaseAgent
from agent_harness.context.context import AgentContext
from agent_harness.core.message import Message


def _agent() -> BaseAgent:
    agent = MagicMock()
    agent.context = AgentContext()
    return cast(BaseAgent, agent)


def test_annotate_noop_when_no_drift(tmp_path: Path) -> None:
    agent = _agent()
    f = tmp_path / "a.txt"
    f.write_text("x")
    mark_read(agent, f)
    msg = Message.user("hi")
    file_observer.annotate_drift(agent, msg)
    assert msg.content == "hi"


def test_annotate_appends_only_drifted_path(tmp_path: Path) -> None:
    agent = _agent()
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    for p in (a, b):
        p.write_text("x")
        mark_read(agent, p)
    future = time.time() + 10
    os.utime(b, (future, future))
    msg = Message.user("hi")
    file_observer.annotate_drift(agent, msg)
    assert msg.content.startswith("hi")
    assert "b.txt" in msg.content
    assert "a.txt" not in msg.content


def test_annotate_marks_modified(tmp_path: Path) -> None:
    agent = _agent()
    f = tmp_path / "code.py"
    f.write_text("x")
    mark_read(agent, f)
    future = time.time() + 10
    os.utime(f, (future, future))
    msg = Message.user("hi")
    file_observer.annotate_drift(agent, msg)
    assert "(modified)" in msg.content


def test_annotate_marks_deleted(tmp_path: Path) -> None:
    agent = _agent()
    f = tmp_path / "code.py"
    f.write_text("x")
    mark_read(agent, f)
    os.unlink(f)
    msg = Message.user("hi")
    file_observer.annotate_drift(agent, msg)
    assert "(deleted)" in msg.content


def test_annotate_dedupes_same_drift(tmp_path: Path) -> None:
    agent = _agent()
    f = tmp_path / "a.txt"
    f.write_text("x")
    mark_read(agent, f)
    future = time.time() + 10
    os.utime(f, (future, future))

    first = Message.user("one")
    file_observer.annotate_drift(agent, first)
    assert "a.txt" in first.content

    second = Message.user("two")
    file_observer.annotate_drift(agent, second)
    assert second.content == "two"


def test_annotate_again_after_new_change(tmp_path: Path) -> None:
    agent = _agent()
    f = tmp_path / "a.txt"
    f.write_text("x")
    mark_read(agent, f)

    f.write_text("xx")
    first = Message.user("one")
    file_observer.annotate_drift(agent, first)
    assert "a.txt" in first.content

    f.write_text("xxx")
    second = Message.user("two")
    file_observer.annotate_drift(agent, second)
    assert "a.txt" in second.content


def test_notice_contains_system_reminder_tag(tmp_path: Path) -> None:
    agent = _agent()
    f = tmp_path / "a.txt"
    f.write_text("x")
    mark_read(agent, f)
    os.unlink(f)
    content = _format_notice(poll_drift(agent))
    assert content.startswith("<system-reminder>")
    assert content.rstrip().endswith("</system-reminder>")
    assert "re-read it with read_file" in content


def test_notice_lists_causes_without_already_aware(tmp_path: Path) -> None:
    agent = _agent()
    f = tmp_path / "a.txt"
    f.write_text("x")
    mark_read(agent, f)
    os.unlink(f)
    content = _format_notice(poll_drift(agent))
    assert "already aware" not in content
    assert "terminal_tool command" in content
    assert "changed on disk" in content


def test_notice_uses_workspace_relative_paths(tmp_path: Path) -> None:
    agent = _agent()
    sub = tmp_path / "src"
    sub.mkdir()
    f = sub / "foo.py"
    f.write_text("x")
    mark_read(agent, f)
    os.unlink(f)

    with patch(
        "agent_app.tools.filesystem._security.get_workspace_root",
        return_value=tmp_path,
    ):
        text = _format_notice(poll_drift(agent))
    assert "src/foo.py" in text
    assert str(tmp_path) not in text
