"""Tests for headless run mode — `arktor -p / --prompt`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agent_cli.app import _build_parser
from agent_cli.headless import run_headless
from agent_harness import AgentResult
from agent_harness.agent.base import StepResult
from agent_harness.approval.handler import AutoApproveHandler
from agent_harness.approval.types import ApprovalDecision, ApprovalRequest
from agent_harness.core.message import ToolCall
from agent_harness.hooks.base import DefaultHooks
from agent_harness.llm.types import Usage


@pytest.fixture
def session_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        "agent_harness.session.file_session._DEFAULT_SESSION_DIR", tmp_path,
    )
    return tmp_path


def _args(*argv: str) -> argparse.Namespace:
    return _build_parser().parse_args(list(argv))


class _FakeApproval:
    def __init__(self, mode: str) -> None:
        self._mode = mode
        self.history = [mode]

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, m: str) -> None:
        self._mode = m
        self.history.append(m)


class _FakeAgent:
    def __init__(
        self, *, output: str = "", error: Exception | None = None,
        mode: str = "auto", steps: list[StepResult] | None = None,
    ) -> None:
        self._approval = _FakeApproval(mode)
        self._session_metadata_extras: dict[str, Any] = {}
        self.run_mode_seen: str | None = None
        if error is not None:
            self.run = AsyncMock(side_effect=error)
        else:
            res = AgentResult(output=output, steps=steps or [], usage=Usage())

            async def _run(task: str, *, session: Any = None, **kw: Any) -> AgentResult:
                self.run_mode_seen = self._approval.mode
                return res

            self.run = _run  # type: ignore[assignment]


@pytest.fixture
def patch_runtime(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    captured: dict[str, Any] = {}

    def _install(agent: _FakeAgent) -> dict[str, Any]:
        def fake_create(*, hooks: Any = None, approval_handler: Any = None, **kw: Any) -> Any:
            captured["hooks"] = hooks
            captured["approval_handler"] = approval_handler
            return agent

        monkeypatch.setattr("agent_cli.agent_factory.create_cli_agent", fake_create)
        monkeypatch.setattr("agent_cli.config.load_config", lambda: None)
        monkeypatch.setattr(
            "agent_cli.runtime.session.make_save_session", lambda a, b: AsyncMock(),
        )
        monkeypatch.setattr("agent_cli.runtime.session.stop_sandbox", AsyncMock())
        monkeypatch.setattr("agent_cli.runtime.background.has_running", lambda a: False)
        monkeypatch.setattr("agent_cli.runtime.background.shutdown", AsyncMock())
        return captured

    return _install


async def test_empty_prompt_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    rc = await run_headless(_args("-p", "   "))
    assert rc == 2
    assert "non-empty task" in capsys.readouterr().err


async def test_runs_and_prints_only_output(
    session_dir: Path, patch_runtime: Any, capsys: pytest.CaptureFixture[str],
) -> None:
    patch_runtime(_FakeAgent(output="final answer"))
    rc = await run_headless(_args("-p", "do the thing"))
    out = capsys.readouterr()
    assert rc == 0
    assert out.out == "final answer\n"
    assert out.err == ""


async def test_unrestricted_then_restores_mode(
    session_dir: Path, patch_runtime: Any,
) -> None:
    agent = _FakeAgent(output="x", mode="auto")
    patch_runtime(agent)
    await run_headless(_args("-p", "go"))
    assert agent.run_mode_seen == "never"
    assert agent._approval.mode == "auto"
    assert "never" in agent._approval.history


async def test_uses_silent_hooks_and_auto_approve(
    session_dir: Path, patch_runtime: Any,
) -> None:
    captured = patch_runtime(_FakeAgent(output="x"))
    await run_headless(_args("-p", "go"))
    assert isinstance(captured["hooks"], DefaultHooks)
    assert isinstance(captured["approval_handler"], AutoApproveHandler)


async def test_run_error_returns_1_and_restores_mode(
    session_dir: Path, patch_runtime: Any, capsys: pytest.CaptureFixture[str],
) -> None:
    agent = _FakeAgent(error=RuntimeError("boom"), mode="auto")
    patch_runtime(agent)
    rc = await run_headless(_args("-p", "go"))
    out = capsys.readouterr()
    assert rc == 1
    assert "boom" in out.err
    assert out.out == ""
    assert agent._approval.mode == "auto"


async def test_cleanup_failure_preserves_output(
    session_dir: Path,
    patch_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patch_runtime(_FakeAgent(output="result"))
    monkeypatch.setattr(
        "agent_cli.runtime.session.stop_sandbox",
        AsyncMock(side_effect=RuntimeError("docker boom")),
    )
    rc = await run_headless(_args("-p", "go"))
    out = capsys.readouterr()
    assert rc == 0
    assert out.out == "result\n"
    assert "stop sandbox" in out.err
    assert "docker boom" in out.err


async def test_json_output_emits_ndjson_steps_and_result(
    session_dir: Path, patch_runtime: Any, capsys: pytest.CaptureFixture[str],
) -> None:
    steps = [StepResult(thought="thinking"), StepResult(response="done")]
    patch_runtime(_FakeAgent(output="done", steps=steps))
    rc = await run_headless(_args("-p", "go", "-s", "sess1", "--output-format", "json"))
    out = capsys.readouterr()
    assert rc == 0
    assert out.err == ""
    lines = [json.loads(x) for x in out.out.splitlines()]
    assert [line["type"] for line in lines] == ["step", "step", "result"]
    assert lines[0]["index"] == 0
    assert lines[0]["thought"] == "thinking"
    res = lines[-1]
    assert res["is_error"] is False
    assert res["session_id"] == "sess1"
    assert res["output"] == "done"
    assert res["num_steps"] == 2
    assert "usage" in res


async def test_json_output_error_is_single_result_line(
    session_dir: Path, patch_runtime: Any, capsys: pytest.CaptureFixture[str],
) -> None:
    patch_runtime(_FakeAgent(error=RuntimeError("kaboom")))
    rc = await run_headless(_args("-p", "go", "-s", "sess2", "--output-format", "json"))
    out = capsys.readouterr()
    assert rc == 1
    assert out.err == ""  # json mode: error lives in the result line, not stderr
    lines = [json.loads(x) for x in out.out.splitlines()]
    assert lines == [{
        "type": "result", "is_error": True, "session_id": "sess2",
        "output": None, "error": "kaboom",
    }]


async def test_corrupted_resume_returns_2(
    session_dir: Path, patch_runtime: Any, capsys: pytest.CaptureFixture[str],
) -> None:
    (session_dir / "wrecked.json").write_text("garbage", encoding="utf-8")
    patch_runtime(_FakeAgent(output="never reached"))
    rc = await run_headless(_args("-p", "go", "-r", "wrecked"))
    assert rc == 2
    assert "corrupted" in capsys.readouterr().err


async def test_auto_approve_handler_allows_once() -> None:
    handler = AutoApproveHandler()
    req = ApprovalRequest(
        tool_call=ToolCall(id="t1", name="terminal_tool", arguments={}),
        agent_name="cli",
    )
    res = await handler.request_approval(req)
    assert res.decision == ApprovalDecision.ALLOW_ONCE
    assert res.tool_call_id == "t1"
    assert res.tool_name == "terminal_tool"
