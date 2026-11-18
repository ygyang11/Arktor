"""Tests for agent_harness.tool.executor — ToolExecutor execution, errors, timeout, batch."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from agent_harness.core.config import HarnessConfig, ToolConfig
from agent_harness.core.message import ToolCall
from agent_harness.core.errors import ToolError
from agent_harness.tool.base import BaseTool, ToolSchema
from agent_harness.tool.executor import ToolExecutor
from agent_harness.tool.registry import ToolRegistry

from tests.conftest import MockTool


class _SlowTool(BaseTool):
    """Tool that sleeps to test timeout."""

    def __init__(self, delay: float = 5.0) -> None:
        super().__init__(name="slow_tool", description="A slow tool")
        self.delay = delay

    async def execute(self, **kwargs: Any) -> str:
        await asyncio.sleep(self.delay)
        return "done"


class _FailingTool(BaseTool):
    """Tool that always raises."""

    def __init__(self) -> None:
        super().__init__(name="fail_tool", description="Always fails")

    async def execute(self, **kwargs: Any) -> str:
        raise RuntimeError("tool exploded")


class _ToolErrorTool(BaseTool):
    """Tool that raises a structured ToolError."""

    def __init__(self) -> None:
        super().__init__(name="tool_error_tool", description="Raises ToolError")

    async def execute(self, **kwargs: Any) -> str:
        raise ToolError("structured failure")


def _make_executor(*tools: BaseTool, config: ToolConfig | None = None) -> ToolExecutor:
    registry = ToolRegistry()
    for t in tools:
        registry.register(t)
    return ToolExecutor(registry, config=config)


class TestToolExecutorSuccess:
    @pytest.mark.asyncio
    async def test_successful_execution(self) -> None:
        mock = MockTool(response="hello world")
        executor = _make_executor(mock)
        tc = ToolCall(name="mock_tool", arguments={"query": "test"})

        result = await executor.execute(tc)
        assert result.content == "hello world"
        assert result.is_error is False
        assert result.tool_call_id == tc.id
        assert mock.call_history == [{"query": "test"}]

    @pytest.mark.asyncio
    async def test_execution_with_no_args(self) -> None:
        mock = MockTool(response="ok")
        executor = _make_executor(mock)
        tc = ToolCall(name="mock_tool", arguments={})
        result = await executor.execute(tc)
        assert result.content == "ok"

    @pytest.mark.asyncio
    async def test_accepts_harness_config(self) -> None:
        mock = MockTool(response="ok")
        cfg = HarnessConfig(tool=ToolConfig(max_concurrency=2, default_timeout=0.2))
        executor = _make_executor(mock, config=cfg)

        tc = ToolCall(name="mock_tool", arguments={})
        result = await executor.execute(tc)

        assert executor.config.max_concurrency == 2
        assert executor.config.default_timeout == 0.2
        assert result.content == "ok"


class TestToolExecutorErrors:
    @pytest.mark.asyncio
    async def test_tool_not_found(self) -> None:
        executor = _make_executor()  # empty registry
        tc = ToolCall(name="missing_tool", arguments={})
        result = await executor.execute(tc)
        assert result.is_error is True
        assert "not found" in result.content.lower()

    @pytest.mark.asyncio
    async def test_tool_raises_exception(self) -> None:
        executor = _make_executor(_FailingTool())
        tc = ToolCall(name="fail_tool", arguments={})
        result = await executor.execute(tc)
        assert result.is_error is True
        assert "tool exploded" in result.content.lower() or "error" in result.content.lower()

    @pytest.mark.asyncio
    async def test_tool_error_logs_at_debug_only(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        executor = _make_executor(_ToolErrorTool())
        tc = ToolCall(name="tool_error_tool", arguments={})

        target = logging.getLogger("agent_harness.tool.executor")
        target.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.DEBUG, logger="agent_harness.tool.executor"):
                result = await executor.execute(tc)
        finally:
            target.removeHandler(caplog.handler)

        assert result.is_error is True
        assert any(
            r.levelno == logging.DEBUG and "structured failure" in r.message
            for r in caplog.records
        )
        assert not any(r.levelno >= logging.WARNING for r in caplog.records)

    @pytest.mark.asyncio
    async def test_unexpected_exception_logs_traceback_at_debug_only(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        executor = _make_executor(_FailingTool())
        tc = ToolCall(name="fail_tool", arguments={})

        target = logging.getLogger("agent_harness.tool.executor")
        target.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.DEBUG, logger="agent_harness.tool.executor"):
                result = await executor.execute(tc)
        finally:
            target.removeHandler(caplog.handler)

        assert result.is_error is True
        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("unexpected error" in r.message for r in debug_records)
        assert any(r.exc_info is not None for r in debug_records)
        assert not any(r.levelno >= logging.WARNING for r in caplog.records)


class TestToolExecutorTimeout:
    @pytest.mark.asyncio
    async def test_timeout_handling(self) -> None:
        slow = _SlowTool(delay=10.0)
        executor = _make_executor(slow)
        tc = ToolCall(name="slow_tool", arguments={})
        result = await executor.execute(tc, timeout=0.1)
        assert result.is_error is True
        assert "timed out" in result.content.lower()

    @pytest.mark.asyncio
    async def test_default_timeout_from_config(self) -> None:
        slow = _SlowTool(delay=10.0)
        config = ToolConfig(default_timeout=0.1)
        executor = _make_executor(slow, config=config)
        tc = ToolCall(name="slow_tool", arguments={})
        result = await executor.execute(tc)
        assert result.is_error is True
        assert "timed out" in result.content.lower()

    @pytest.mark.asyncio
    async def test_timeout_logs_at_debug_only(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        slow = _SlowTool(delay=10.0)
        executor = _make_executor(slow)
        tc = ToolCall(name="slow_tool", arguments={})

        target = logging.getLogger("agent_harness.tool.executor")
        target.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.DEBUG, logger="agent_harness.tool.executor"):
                result = await executor.execute(tc, timeout=0.1)
        finally:
            target.removeHandler(caplog.handler)

        assert result.is_error is True
        assert any(
            r.levelno == logging.DEBUG and "timed out" in r.message
            for r in caplog.records
        )
        assert not any(r.levelno >= logging.WARNING for r in caplog.records)


class _GatedTool(BaseTool):
    def __init__(self, *, swallow: bool = False, fail: bool = False) -> None:
        super().__init__(name="gated", description="Test resource lifetime")
        self.started = asyncio.Event()
        self.cleaning = asyncio.Event()
        self.release = asyncio.Event()
        self.released = False
        self.swallow = swallow
        self.fail = fail

    async def execute(self, **kwargs: Any) -> str:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cleaning.set()
            await self.release.wait()
            self.released = True
            if self.fail:
                raise RuntimeError("cleanup failure")
            if self.swallow:
                return "recovered"
            raise
        raise AssertionError("tool should be cancelled")


class TestToolExecutorCancellation:
    @pytest.mark.parametrize("stream", [False, True])
    async def test_repeated_cancel_waits_for_resource_release(self, stream: bool) -> None:
        tool = _GatedTool()
        executor = _make_executor(tool, config=ToolConfig(default_timeout=0.03))
        call = ToolCall(name=tool.name, arguments={})

        async def consume() -> None:
            async for _ in executor.execute_stream([call]):
                pass

        task = asyncio.create_task(consume()) if stream else asyncio.create_task(executor.execute(call))
        try:
            if stream:
                await asyncio.wait_for(tool.started.wait(), timeout=2)
                task.cancel()
                await asyncio.wait_for(tool.cleaning.wait(), timeout=2)
                await asyncio.sleep(0.06)
                task.cancel()
            else:
                await asyncio.wait_for(tool.cleaning.wait(), timeout=2)
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()

            await asyncio.sleep(0)
            assert not task.done()
        finally:
            tool.release.set()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert tool.released

    @pytest.mark.parametrize("swallow,fail", [(False, False), (True, False), (False, True)])
    async def test_timeout_preserves_cleanup_result(self, swallow: bool, fail: bool) -> None:
        tool = _GatedTool(swallow=swallow, fail=fail)
        executor = _make_executor(tool, config=ToolConfig(default_timeout=0.03))
        task = asyncio.create_task(executor.execute(ToolCall(name=tool.name, arguments={})))
        try:
            await asyncio.wait_for(tool.cleaning.wait(), timeout=2)
        finally:
            tool.release.set()
        result = await asyncio.wait_for(task, timeout=2)

        assert tool.released
        assert result.is_error == (not swallow)
        expected = "cleanup failure" if fail else "recovered" if swallow else "timed out"
        assert expected in result.content

    async def test_external_cancel_not_swallowed(self) -> None:
        tool = _GatedTool(swallow=True)
        executor = _make_executor(tool)
        task = asyncio.create_task(executor.execute(ToolCall(name=tool.name, arguments={})))
        try:
            await asyncio.wait_for(tool.started.wait(), timeout=2)
            task.cancel()
            await asyncio.wait_for(tool.cleaning.wait(), timeout=2)
        finally:
            tool.release.set()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert tool.released


class TestToolExecutorBatch:
    @pytest.mark.asyncio
    async def test_batch_execution(self) -> None:
        mock = MockTool(response="batch_result")
        executor = _make_executor(mock)

        calls = [
            ToolCall(name="mock_tool", arguments={"query": "a"}),
            ToolCall(name="mock_tool", arguments={"query": "b"}),
            ToolCall(name="mock_tool", arguments={"query": "c"}),
        ]
        results = await executor.execute_batch(calls)

        assert len(results) == 3
        assert all(r.content == "batch_result" for r in results)
        assert all(r.is_error is False for r in results)
        for call, result in zip(calls, results):
            assert result.tool_call_id == call.id

    @pytest.mark.asyncio
    async def test_batch_empty(self) -> None:
        executor = _make_executor()
        results = await executor.execute_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_batch_with_mixed_results(self) -> None:
        mock = MockTool(response="ok")
        executor = _make_executor(mock)
        calls = [
            ToolCall(name="mock_tool", arguments={}),
            ToolCall(name="nonexistent", arguments={}),
        ]
        results = await executor.execute_batch(calls)
        assert len(results) == 2
        assert results[0].is_error is False
        assert results[1].is_error is True


class TestToolExecutorStream:
    async def test_stream_yields_in_completion_order(self) -> None:
        fast = _SlowTool(delay=0.05)
        fast.name = "fast_tool"
        slow = _SlowTool(delay=0.2)
        slow.name = "slow_tool"
        registry = ToolRegistry()
        registry.register(fast)
        registry.register(slow)
        executor = ToolExecutor(registry)

        calls = [
            ToolCall(id="slow", name="slow_tool", arguments={}),
            ToolCall(id="fast", name="fast_tool", arguments={}),
        ]

        results: list[str] = []
        async for result in executor.execute_stream(calls):
            results.append(result.tool_call_id)

        assert results[0] == "fast"
        assert results[1] == "slow"

    async def test_stream_empty(self) -> None:
        executor = _make_executor()
        results = [r async for r in executor.execute_stream([])]
        assert results == []

    async def test_stream_cleanup_on_early_exit(self) -> None:
        cancelled = False

        class _CancelTracker(BaseTool):
            def __init__(self) -> None:
                super().__init__(name="trackable", description="t")

            async def execute(self, **kwargs: Any) -> str:
                nonlocal cancelled
                try:
                    await asyncio.sleep(10)
                    return "done"
                except asyncio.CancelledError:
                    cancelled = True
                    raise

            def get_schema(self) -> ToolSchema:
                return ToolSchema(name=self.name, description=self.description)

        fast = MockTool(response="fast")
        tracker = _CancelTracker()
        registry = ToolRegistry()
        registry.register(fast)
        registry.register(tracker)
        executor = ToolExecutor(registry)

        calls = [
            ToolCall(id="1", name="mock_tool", arguments={}),
            ToolCall(id="2", name="trackable", arguments={}),
        ]

        async for _ in executor.execute_stream(calls):
            break  # exit after first result

        await asyncio.sleep(0.1)
        assert cancelled

    async def test_stream_batch_still_works(self) -> None:
        mock = MockTool(response="batch_ok")
        executor = _make_executor(mock)
        calls = [
            ToolCall(name="mock_tool", arguments={"query": "a"}),
            ToolCall(name="mock_tool", arguments={"query": "b"}),
        ]
        results = await executor.execute_batch(calls)
        assert len(results) == 2
        assert all(r.content == "batch_ok" for r in results)


class _ToolOutputTool(BaseTool):
    def __init__(self, output: Any) -> None:
        super().__init__(name="tool_output_tool", description="Returns ToolOutput")
        self._output = output

    async def execute(self, **kwargs: Any) -> Any:
        return self._output


class TestToolExecutorToolOutput:
    @pytest.mark.asyncio
    async def test_str_return_is_normalized_without_attachments(self) -> None:
        tool = _ToolOutputTool("plain text")
        executor = _make_executor(tool)
        tc = ToolCall(name="tool_output_tool", arguments={})
        result = await executor.execute(tc)
        assert result.content == "plain text"
        assert result.attachments is None
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_tool_output_with_attachments_threads_through(self) -> None:
        from agent_harness.core.message import Attachment, ToolOutput

        att = Attachment(
            digest="a" * 64, mime="image/png", size=4, filename="x.png",
        )
        tool = _ToolOutputTool(ToolOutput(content="see image", attachments=[att]))
        executor = _make_executor(tool)
        tc = ToolCall(name="tool_output_tool", arguments={})
        result = await executor.execute(tc)
        assert result.content == "see image"
        assert result.attachments == [att]
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_tool_output_without_attachments(self) -> None:
        from agent_harness.core.message import ToolOutput

        tool = _ToolOutputTool(ToolOutput(content="just text"))
        executor = _make_executor(tool)
        tc = ToolCall(name="tool_output_tool", arguments={})
        result = await executor.execute(tc)
        assert result.content == "just text"
        assert result.attachments is None

    @pytest.mark.asyncio
    async def test_tool_output_metadata_threads_through(self) -> None:
        from agent_harness.core.message import ToolOutput

        tool = _ToolOutputTool(
            ToolOutput(content="header", tool_metadata={"diff": "-a\n+b"}),
        )
        executor = _make_executor(tool)
        tc = ToolCall(name="tool_output_tool", arguments={})
        result = await executor.execute(tc)
        assert result.content == "header"
        assert result.tool_metadata == {"diff": "-a\n+b"}

    @pytest.mark.asyncio
    async def test_str_return_has_no_metadata(self) -> None:
        tool = _ToolOutputTool("plain")
        executor = _make_executor(tool)
        tc = ToolCall(name="tool_output_tool", arguments={})
        result = await executor.execute(tc)
        assert result.tool_metadata is None
