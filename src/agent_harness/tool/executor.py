"""Tool executor with concurrency control, timeout, and error handling."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Iterable
from typing import TypeVar

from agent_harness.core.config import HarnessConfig, ToolConfig, resolve_tool_config
from agent_harness.core.errors import (
    ToolError,
    ToolNotFoundError,
)
from agent_harness.core.event import EventEmitter
from agent_harness.core.message import ToolCall, ToolOutput, ToolResult
from agent_harness.tool.registry import ToolRegistry

logger = logging.getLogger(__name__)


_T = TypeVar("_T")


async def _cancel_and_wait(tasks: Iterable[asyncio.Task[_T]]) -> bool:
    """Cancel once and finish cleanup; report cancellation of this waiter."""
    workers = tuple(tasks)
    for task in workers:
        if not task.done() and task.cancelling() == 0:
            task.cancel()
    completion = asyncio.gather(*workers, return_exceptions=True)
    interrupted = False
    while not completion.done():
        try:
            await asyncio.shield(completion)
        except asyncio.CancelledError:
            interrupted = True
    completion.result()
    return interrupted


class ToolExecutor(EventEmitter):
    """Executes tool calls with concurrency control, timeouts, and error handling.

    Features:
        - Concurrent execution with configurable max concurrency
        - Per-tool timeout
        - Automatic error capture -> ToolResult(is_error=True)
        - Event emission for observability
    """

    def __init__(
        self,
        registry: ToolRegistry,
        config: HarnessConfig | ToolConfig | None = None,
    ) -> None:
        self.registry = registry
        self.config = resolve_tool_config(config)
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)

    async def execute(
        self,
        tool_call: ToolCall,
        timeout: float | None = None,
    ) -> ToolResult:
        """Execute a single tool call.

        Args:
            tool_call: The tool call to execute.
            timeout: Override default timeout (seconds).

        Returns:
            ToolResult with the execution result or error.
        """
        await self.emit(
            "tool.execute.start",
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
            arguments=tool_call.arguments,
        )

        try:
            # Resolve the tool
            if not self.registry.has(tool_call.name):
                raise ToolNotFoundError(
                    f"Tool '{tool_call.name}' not found. "
                    f"Available: {[t.name for t in self.registry.list_tools()]}"
                )

            tool = self.registry.get(tool_call.name)

            # Per-tool timeout: explicit param > tool declaration > global default
            effective_timeout = timeout or tool.executor_timeout or self.config.default_timeout

            # Execute with concurrency limit and timeout
            async with self._semaphore:
                tool_task = asyncio.create_task(tool.execute(**tool_call.arguments))
                try:
                    raw = await asyncio.wait_for(
                        asyncio.shield(tool_task), timeout=effective_timeout,
                    )
                except BaseException as exc:
                    interrupted = await _cancel_and_wait([tool_task])
                    if not isinstance(exc, TimeoutError):
                        raise
                    if interrupted:
                        raise asyncio.CancelledError from exc
                    if tool_task.cancelled():
                        raise
                    raw = tool_task.result()

            if isinstance(raw, ToolOutput):
                result_str = raw.content
                attachments = raw.attachments
                tool_metadata = raw.tool_metadata
            else:
                result_str = raw
                attachments = None
                tool_metadata = None

            result = ToolResult(
                tool_call_id=tool_call.id,
                content=result_str,
                is_error=False,
                attachments=attachments,
                tool_metadata=tool_metadata,
            )

            await self.emit(
                "tool.execute.end",
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                content_length=len(result_str),
            )
            return result

        except TimeoutError:
            error_msg = f"Tool '{tool_call.name}' timed out after {effective_timeout}s"
            logger.debug(error_msg)
            await self.emit(
                "tool.execute.error",
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                error=error_msg,
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                content=error_msg,
                is_error=True,
            )

        except ToolError as e:
            error_msg = f"Tool '{tool_call.name}' error: {e}"
            logger.debug(error_msg)
            await self.emit(
                "tool.execute.error",
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                error=error_msg,
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                content=error_msg,
                is_error=True,
            )

        except Exception as e:
            logger.debug(
                "Tool '%s' unexpected error: %s",
                tool_call.name,
                e,
                exc_info=e,
            )
            await self.emit(
                "tool.execute.error",
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                error=str(e),
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error: {e}",
                is_error=True,
            )

    async def execute_batch(
        self,
        tool_calls: list[ToolCall],
        timeout: float | None = None,
    ) -> list[ToolResult]:
        """Execute multiple tool calls concurrently.

        Results are returned in the same order as input tool_calls.
        Concurrency is bounded by max_concurrency.

        Args:
            tool_calls: List of tool calls to execute.
            timeout: Override default timeout per tool.

        Returns:
            List of ToolResults in the same order as input.
        """
        if not tool_calls:
            return []

        tasks = [self.execute(tc, timeout=timeout) for tc in tool_calls]
        return list(await asyncio.gather(*tasks))

    async def execute_stream(
        self,
        tool_calls: list[ToolCall],
        timeout: float | None = None,
    ) -> AsyncGenerator[ToolResult, None]:
        """Execute tool calls concurrently, yielding results as each completes.

        Unlike execute_batch (which waits for all), this yields each result
        as soon as it finishes. Enables real-time UI updates.

        Concurrency is still bounded by max_concurrency semaphore.
        Results are yielded in completion order, not call order.
        Remaining tasks are cancelled if the consumer stops early.
        """
        if not tool_calls:
            return

        tasks: dict[asyncio.Task[ToolResult], ToolCall] = {
            asyncio.create_task(self.execute(tc, timeout=timeout)): tc
            for tc in tool_calls
        }
        failure: BaseException | None = None
        try:
            for future in asyncio.as_completed(tasks):
                yield await future
        except BaseException as exc:
            failure = exc
            raise
        finally:
            interrupted = await _cancel_and_wait(tasks)
            if interrupted and (failure is None or isinstance(failure, GeneratorExit)):
                raise asyncio.CancelledError
