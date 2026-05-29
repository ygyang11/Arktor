"""Tests for the refactored terminal_tool."""

from __future__ import annotations

import pytest

from agent_app.tools.terminal.terminal_tool import TerminalTool
from agent_harness.core.config import ToolConfig
from agent_harness.core.errors import ToolValidationError
from agent_harness.core.message import ToolCall
from agent_harness.sandbox import SandboxManager
from agent_harness.sandbox.backend import LocalBackend
from agent_harness.tool.decorator import tool
from agent_harness.tool.executor import ToolExecutor
from agent_harness.tool.registry import ToolRegistry


def _make_tool() -> TerminalTool:
    """Create a TerminalTool bound to a mock agent with LocalBackend sandbox."""
    t = TerminalTool()

    class _MockAgent:
        _sandbox = SandboxManager(LocalBackend())
        _bg_manager = None

    t.bind_agent(_MockAgent())
    return t


class TestTerminalTool:
    """Unit tests — direct tool execution."""

    @pytest.fixture
    def tt(self) -> TerminalTool:
        return _make_tool()

    # --- Basic execution ---

    @pytest.mark.asyncio
    async def test_simple_command(self, tt: TerminalTool) -> None:
        result = await tt.execute(command="echo hello")
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_empty_command(self, tt: TerminalTool) -> None:
        with pytest.raises(ToolValidationError, match="command cannot be empty"):
            await tt.execute(command="")

    @pytest.mark.asyncio
    async def test_whitespace_only_command(self, tt: TerminalTool) -> None:
        with pytest.raises(ToolValidationError, match="command cannot be empty"):
            await tt.execute(command="   ")

    @pytest.mark.asyncio
    async def test_no_output(self, tt: TerminalTool) -> None:
        result = await tt.execute(command="true")
        assert result == "(no output)"

    @pytest.mark.asyncio
    async def test_non_zero_exit(self, tt: TerminalTool) -> None:
        result = await tt.execute(
            command='python3 -c "import sys; sys.exit(2)"', timeout=5,
        )
        assert "[exit code 2]" in result

    @pytest.mark.asyncio
    async def test_stderr_merged(self, tt: TerminalTool) -> None:
        result = await tt.execute(command="echo err >&2 && echo out", timeout=5)
        assert "out" in result
        assert "err" in result

    # --- Full bash syntax ---

    @pytest.mark.asyncio
    async def test_pipe(self, tt: TerminalTool) -> None:
        result = await tt.execute(command="echo hello | tr 'h' 'H'", timeout=5)
        assert "Hello" in result

    @pytest.mark.asyncio
    async def test_variable_expansion(self, tt: TerminalTool) -> None:
        result = await tt.execute(command="X=42 && echo $X", timeout=5)
        assert "42" in result

    @pytest.mark.asyncio
    async def test_git(self, tt: TerminalTool) -> None:
        result = await tt.execute(command="git --version", timeout=5)
        assert "git version" in result

    @pytest.mark.asyncio
    async def test_subshell(self, tt: TerminalTool) -> None:
        result = await tt.execute(command="echo $(echo nested)", timeout=5)
        assert "nested" in result

    @pytest.mark.asyncio
    async def test_redirect(self, tt: TerminalTool) -> None:
        result = await tt.execute(
            command="echo data > out.txt && cat out.txt", timeout=5,
        )
        assert "data" in result

    # --- Timeout ---

    @pytest.mark.asyncio
    async def test_timeout_enforced(self, tt: TerminalTool) -> None:
        result = await tt.execute(command="sleep 10", timeout=1)
        assert "timed out" in result

    @pytest.mark.asyncio
    async def test_timeout_capped_at_max(self, tt: TerminalTool) -> None:
        result = await tt.execute(command="echo ok", timeout=1000)
        assert "ok" in result

    @pytest.mark.asyncio
    async def test_timeout_zero(self, tt: TerminalTool) -> None:
        with pytest.raises(ToolValidationError, match="timeout must be greater than 0"):
            await tt.execute(command="echo x", timeout=0)

    @pytest.mark.asyncio
    async def test_timeout_negative(self, tt: TerminalTool) -> None:
        with pytest.raises(ToolValidationError, match="timeout must be greater than 0"):
            await tt.execute(command="echo x", timeout=-1)

    # --- Output truncation ---

    @pytest.mark.asyncio
    async def test_large_output_truncated(self, tt: TerminalTool) -> None:
        result = await tt.execute(
            command='python3 -c "print(\'x \' * 50000)"', timeout=10,
        )
        assert "... (truncated)" in result

    # --- Exit code N/A ---

    @pytest.mark.asyncio
    async def test_timeout_shows_na_exit_code(self, tt: TerminalTool) -> None:
        result = await tt.execute(command="sleep 10", timeout=1)
        assert "[exit code N/A]" in result


class TestTerminalToolIntegration:
    """Integration tests through ToolExecutor — validates timeout chain."""

    @pytest.fixture
    def tt(self) -> TerminalTool:
        return _make_tool()

    @pytest.mark.asyncio
    async def test_executor_timeout_overrides_short_default(self, tt: TerminalTool) -> None:
        registry = ToolRegistry()
        registry.register(tt)
        config = ToolConfig(default_timeout=1.0)
        executor = ToolExecutor(registry, config=config)

        tc = ToolCall(
            name="terminal_tool",
            arguments={"command": "sleep 2 && echo done", "timeout": 5},
        )
        result = await executor.execute(tc)
        assert not result.is_error
        assert "done" in result.content

    @pytest.mark.asyncio
    async def test_internal_timeout_fires_before_executor(self, tt: TerminalTool) -> None:
        registry = ToolRegistry()
        registry.register(tt)
        executor = ToolExecutor(registry)

        tc = ToolCall(
            name="terminal_tool",
            arguments={"command": "sleep 60", "timeout": 1},
        )
        result = await executor.execute(tc)
        assert "timed out after 1s" in result.content
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_executor_timeout_attribute(self) -> None:
        tt = _make_tool()
        assert tt.executor_timeout is not None

    @pytest.mark.asyncio
    async def test_validation_error_becomes_error_result(self, tt: TerminalTool) -> None:
        registry = ToolRegistry()
        registry.register(tt)
        executor = ToolExecutor(registry)

        tc = ToolCall(name="terminal_tool", arguments={"command": "   "})
        result = await executor.execute(tc)
        assert result.is_error is True
        assert "command cannot be empty" in result.content
        assert "terminal_tool" in result.content
        assert tt.executor_timeout > 600

        @tool
        async def my_tool(x: str) -> str:
            return x

        assert my_tool.executor_timeout is None
