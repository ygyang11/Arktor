"""Abstract base class for LLM providers in agent_harness."""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, AsyncIterator, Callable, Coroutine, TypeVar

from agent_harness.core.config import LLMConfig
from agent_harness.core.errors import LLMConnectionError, LLMError, LLMRateLimitError
from agent_harness.core.event import EventEmitter
from agent_harness.core.message import Message, Role, ToolCall
from agent_harness.llm.types import (
    FinishReason,
    LLMResponse,
    LLMRetryInfo,
    StreamDelta,
    Usage,
)
from agent_harness.tool.base import ToolSchema

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RateLimiter:
    """Token bucket rate limiter for API calls.

    Limits the number of requests per time window to avoid hitting API limits.
    """

    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0) -> None:
        self._max_requests = max_requests
        self._window = window_seconds
        self._timestamps: deque[float] = deque(maxlen=max_requests)
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request slot is available."""
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self._window:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._max_requests:
                    self._timestamps.append(now)
                    return
                # Calculate wait time but release lock before sleeping
                wait_time = self._timestamps[0] + self._window - now

            if wait_time > 0:
                logger.debug("Rate limiter: waiting %.1fs", wait_time)
                await asyncio.sleep(wait_time)


class BaseLLM(ABC, EventEmitter):
    """Abstract base class for LLM providers.

    All LLM providers (OpenAI, Anthropic, etc.) implement this interface,
    giving agents a uniform way to interact with any model.

    Features:
        - Unified generate/stream interface
        - Tool/function calling support
        - Event emission for observability
        - Token counting
    """

    def __init__(self, config: LLMConfig, rate_limit: dict[str, Any] | None = None) -> None:
        self.config = config
        self._rate_limiter: RateLimiter | None = None
        if rate_limit:
            self._rate_limiter = RateLimiter(
                max_requests=rate_limit.get("max_requests", 60),
                window_seconds=rate_limit.get("window_seconds", 60.0),
            )

    @property
    def model_name(self) -> str:
        return self.config.model

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            messages: Conversation history.
            tools: Available tools for function calling.
            tool_choice: Tool selection strategy:
                - "auto": LLM decides whether to call a tool
                - "required": LLM must call a tool
                - "none": No tool calling
                - specific name: Force calling a specific tool
            temperature: Override config temperature.
            max_tokens: Override config max_tokens.

        Returns:
            LLMResponse with the generated message and metadata.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamDelta]:
        """Stream a response from the LLM.

        Args:
            Same as generate().

        Yields:
            StreamDelta chunks as they arrive.
        """
        ...
        if False:
            yield

    async def generate_with_events(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        on_retry: Callable[[LLMRetryInfo], Coroutine[Any, Any, None]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate with automatic event emission and built-in retry."""
        await self.emit("llm.generate.start", model=self.model_name, message_count=len(messages))
        try:
            if self._rate_limiter:
                await self._rate_limiter.acquire()
            response = await self._with_retry(
                lambda: self.generate(messages, tools=tools, **kwargs),
                on_retry=on_retry,
            )
            await self.emit(
                "llm.generate.end",
                model=self.model_name,
                usage=response.usage.model_dump(),
                finish_reason=response.finish_reason.value,
            )
            return response
        except Exception as e:
            await self.emit("llm.generate.error", model=self.model_name, error=str(e))
            raise

    async def _stream_iter(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamDelta]:
        """Internal streaming iterator with event emission and rate limiting."""
        await self.emit("llm.stream.start", model=self.model_name, message_count=len(messages))
        try:
            if self._rate_limiter:
                await self._rate_limiter.acquire()

            total_usage = Usage()

            async for delta in self.stream(messages, tools=tools, **kwargs):
                if delta.usage:
                    total_usage = total_usage + delta.usage
                yield delta

            await self.emit(
                "llm.stream.end",
                model=self.model_name,
                usage=total_usage.model_dump(),
            )
        except Exception as e:
            await self.emit("llm.stream.error", model=self.model_name, error=str(e))
            raise

    async def stream_with_events(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        on_delta: Callable[[StreamDelta], Coroutine[Any, Any, None]] | None = None,
        on_retry: Callable[[LLMRetryInfo], Coroutine[Any, Any, None]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Stream with event emission, rate limiting, and retry, returning LLMResponse.

        Symmetric with generate_with_events(): both return LLMResponse.
        On transient errors the entire stream is retried from scratch.

        Args:
            messages: Conversation history.
            tools: Available tools for function calling.
            on_delta: Optional async callback invoked for each StreamDelta chunk.
            on_retry: Optional async callback invoked once per retry.
            **kwargs: Passed through to stream().
        """
        max_retries = self.config.max_retries
        delay = self.config.retry_delay

        for attempt in range(max_retries + 1):
            content_parts: list[str] = []
            tool_calls: list[ToolCall] | None = None
            provider_metadata: dict[str, dict[str, Any]] = {}
            total_usage = Usage()
            finish_reason = FinishReason.STOP

            try:
                async for delta in self._stream_iter(messages, tools=tools, **kwargs):
                    if on_delta is not None:
                        await on_delta(delta)

                    if delta.chunk.delta_content:
                        content_parts.append(delta.chunk.delta_content)
                    if delta.chunk.delta_tool_calls:
                        tool_calls = delta.chunk.delta_tool_calls
                    delta.chunk.merge_provider_metadata_into(provider_metadata)
                    if delta.usage:
                        total_usage = total_usage + delta.usage
                    if delta.finish_reason:
                        finish_reason = delta.finish_reason

                return LLMResponse(
                    message=Message(
                        role=Role.ASSISTANT,
                        content="".join(content_parts) or None,
                        tool_calls=tool_calls,
                        provider_metadata=provider_metadata,
                    ),
                    usage=total_usage,
                    finish_reason=finish_reason,
                    model=self.model_name,
                )

            except (LLMRateLimitError, LLMConnectionError, ConnectionError, TimeoutError) as e:
                if attempt >= max_retries:
                    raise
                wait = delay * (2 ** attempt)
                if isinstance(e, LLMRateLimitError) and e.retry_after:
                    wait = max(wait, e.retry_after)
                logger.debug(
                    "Stream error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries, wait, e,
                )
                if on_retry is not None:
                    try:
                        await on_retry(LLMRetryInfo(
                            kind="stream",
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            wait=wait,
                            error=e,
                        ))
                    except Exception:
                        logger.exception(
                            "on_retry callback failed; continuing retry loop"
                        )
                await self.emit(
                    "llm.stream.retry",
                    model=self.model_name,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    wait=wait,
                    error=str(e),
                )
                await asyncio.sleep(wait)

        raise RuntimeError("Unreachable")  # pragma: no cover

    async def _with_retry(
        self,
        call: Callable[[], Coroutine[Any, Any, T]],
        *,
        on_retry: Callable[[LLMRetryInfo], Coroutine[Any, Any, None]] | None = None,
    ) -> T:
        """Retry a coroutine on transient errors with exponential backoff."""
        max_retries = self.config.max_retries
        delay = self.config.retry_delay

        for attempt in range(max_retries + 1):
            try:
                return await call()
            except (LLMRateLimitError, LLMConnectionError, ConnectionError, TimeoutError) as e:
                if attempt >= max_retries:
                    raise
                wait = delay * (2 ** attempt)
                if isinstance(e, LLMRateLimitError) and e.retry_after:
                    wait = max(wait, e.retry_after)
                logger.debug(
                    "Transient error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries, wait, e,
                )
                if on_retry is not None:
                    try:
                        await on_retry(LLMRetryInfo(
                            kind="generate",
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            wait=wait,
                            error=e,
                        ))
                    except Exception:
                        logger.exception(
                            "on_retry callback failed; continuing retry loop"
                        )
                await self.emit(
                    "llm.generate.retry",
                    model=self.model_name,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    wait=wait,
                    error=str(e),
                )
                await asyncio.sleep(wait)

        raise RuntimeError("Unreachable")  # pragma: no cover

    def _resolve_temperature(self, temperature: float | None) -> float:
        return temperature if temperature is not None else self.config.temperature

    def _resolve_max_tokens(self, max_tokens: int | None) -> int:
        return max_tokens if max_tokens is not None else self.config.max_tokens

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model={self.model_name}>"


class FallbackChain:
    """Try multiple LLM providers in order, falling back on failure.

    Example:
        chain = FallbackChain([primary_llm, backup_llm])
        response = await chain.generate(messages)
    """

    def __init__(self, providers: list[BaseLLM]) -> None:
        if not providers:
            raise ValueError("FallbackChain requires at least one provider")
        self.providers = providers

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        errors: list[Exception] = []

        for provider in self.providers:
            try:
                return await provider.generate_with_events(messages, tools=tools, **kwargs)
            except Exception as e:
                logger.debug("Provider %s failed: %s, trying next", provider, e)
                errors.append(e)

        raise LLMError(
            f"All {len(self.providers)} providers failed",
            details={"errors": [str(e) for e in errors]},
        )
