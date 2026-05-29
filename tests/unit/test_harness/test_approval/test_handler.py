"""Tests for approval handlers."""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

from agent_harness.approval.handler import StdinApprovalHandler
from agent_harness.approval.types import ApprovalDecision, ApprovalRequest
from agent_harness.core.message import ToolCall


class TestStdinApprovalHandler:
    async def test_default_is_allow_once(self) -> None:
        output = io.StringIO()
        handler = StdinApprovalHandler(output=output, color=False)
        tc = ToolCall(name="my_tool", arguments={"x": "1"})
        request = ApprovalRequest(tool_call=tc, agent_name="agent")

        with patch("builtins.input", return_value=""):
            result = await handler.request_approval(request)

        assert result.decision == ApprovalDecision.ALLOW_ONCE
        assert result.tool_call_id == tc.id

    async def test_yes_is_allow_once(self) -> None:
        handler = StdinApprovalHandler(output=io.StringIO(), color=False)
        tc = ToolCall(name="my_tool", arguments={})
        request = ApprovalRequest(tool_call=tc, agent_name="agent")

        with patch("builtins.input", return_value="y"):
            result = await handler.request_approval(request)
        assert result.decision == ApprovalDecision.ALLOW_ONCE

    async def test_always_is_allow_session(self) -> None:
        handler = StdinApprovalHandler(output=io.StringIO(), color=False)
        tc = ToolCall(name="my_tool", arguments={})
        request = ApprovalRequest(tool_call=tc, agent_name="agent")

        with patch("builtins.input", return_value="a"):
            result = await handler.request_approval(request)
        assert result.decision == ApprovalDecision.ALLOW_SESSION

    async def test_no_is_deny(self) -> None:
        handler = StdinApprovalHandler(output=io.StringIO(), color=False)
        tc = ToolCall(name="my_tool", arguments={})
        request = ApprovalRequest(tool_call=tc, agent_name="agent")

        with patch("builtins.input", return_value="n"):
            result = await handler.request_approval(request)
        assert result.decision == ApprovalDecision.DENY

    async def test_output_shows_tool_info(self) -> None:
        output = io.StringIO()
        handler = StdinApprovalHandler(output=output, color=False)
        tc = ToolCall(name="web_search", arguments={"query": "test"})
        request = ApprovalRequest(tool_call=tc, agent_name="agent")

        with patch("builtins.input", return_value="y"):
            await handler.request_approval(request)

        written = output.getvalue()
        assert "web_search" in written
        assert "query" in written
        assert "Allow?" in written

    def test_always_label_command(self) -> None:
        handler = StdinApprovalHandler(output=io.StringIO(), color=False)
        tc = ToolCall(name="terminal_tool", arguments={"command": "git status"})
        request = ApprovalRequest(
            tool_call=tc, agent_name="agent", resource="git status", resource_kind="command",
        )
        label = handler._always_label(request)
        assert "'git'" in label
        assert "commands" in label

    def test_always_label_path_dir(self) -> None:
        handler = StdinApprovalHandler(output=io.StringIO(), color=False)
        tc = ToolCall(name="write_file", arguments={"file_path": "src/main.py"})
        request = ApprovalRequest(
            tool_call=tc, agent_name="agent", resource="src/main.py", resource_kind="path",
        )
        label = handler._always_label(request)
        assert "src" in label
        assert "under" in label

    def test_always_label_path_root_file(self) -> None:
        handler = StdinApprovalHandler(output=io.StringIO(), color=False)
        tc = ToolCall(name="read_file", arguments={"file_path": "README.md"})
        request = ApprovalRequest(
            tool_call=tc, agent_name="agent", resource="README.md", resource_kind="path",
        )
        label = handler._always_label(request)
        assert "README.md" in label
        assert "on" in label  # "on" not "under" for root files

    def test_always_label_url(self) -> None:
        handler = StdinApprovalHandler(output=io.StringIO(), color=False)
        tc = ToolCall(name="web_fetch", arguments={"url": "https://github.com/x"})
        request = ApprovalRequest(
            tool_call=tc, agent_name="agent", resource="https://github.com/x", resource_kind="url",
        )
        label = handler._always_label(request)
        assert "github.com" in label

    def test_always_label_no_resource(self) -> None:
        handler = StdinApprovalHandler(output=io.StringIO(), color=False)
        tc = ToolCall(name="my_tool", arguments={})
        request = ApprovalRequest(tool_call=tc, agent_name="agent")
        label = handler._always_label(request)
        assert "my_tool" in label


class TestAlwaysLabelPaths:
    @staticmethod
    def _label(tool: str, resource: str) -> str:
        handler = StdinApprovalHandler(output=io.StringIO(), color=False)
        request = ApprovalRequest(
            tool_call=ToolCall(name=tool, arguments={}),
            agent_name="agent",
            resource=resource,
            resource_kind="path",
        )
        return handler._always_label(request)

    def test_absolute_single_file_uses_on(self) -> None:
        label = self._label("read_file", "/foo.txt")
        assert "on '/foo.txt'" in label
        assert "under" not in label

    def test_absolute_file_in_dir_uses_under(self) -> None:
        label = self._label("read_file", "/etc/hosts")
        assert "under '/etc/'" in label

    def test_directory_uses_under(self, tmp_path: Path) -> None:
        label = self._label("list_dir", str(tmp_path))
        assert f"under '{tmp_path}/'" in label
