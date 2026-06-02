"""Tracer and Span model for agent_harness observability.

Provides structured tracing inspired by OpenTelemetry, but lightweight
and purpose-built for agent execution tracking.
"""
from __future__ import annotations

import contextvars
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from functools import wraps
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Context variables for async-safe trace/span tracking
_current_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_trace_id", default=None
)
_current_span_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_span_id", default=None
)


class SpanEvent(BaseModel):
    """An event that occurred during a span."""
    name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
    attributes: dict[str, Any] = Field(default_factory=dict)


class Span(BaseModel):
    """A single unit of work in a trace.

    Spans form a tree structure: each span can have a parent and children.
    """
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    span_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_span_id: str | None = None
    name: str
    kind: str = "internal"  # "agent", "llm", "tool", "memory", "internal"
    start_time: datetime = Field(default_factory=lambda: datetime.now())
    end_time: datetime | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[SpanEvent] = Field(default_factory=list)
    status: str = "ok"  # "ok", "error"
    error_message: str | None = None

    @property
    def duration_ms(self) -> float | None:
        """Duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds() * 1000

    def add_event(self, name: str, **attributes: Any) -> None:
        self.events.append(SpanEvent(name=name, attributes=attributes))

    def set_error(self, error: Exception) -> None:
        self.status = "error"
        self.error_message = str(error)

    def finish(self) -> None:
        self.end_time = datetime.now()


class TraceCollector:
    """Collects finished spans for export."""

    def __init__(self) -> None:
        self._spans: list[Span] = []

    def add_span(self, span: Span) -> None:
        self._spans.append(span)

    def get_spans(self) -> list[Span]:
        return list(self._spans)

    def get_trace(self, trace_id: str) -> list[Span]:
        return [s for s in self._spans if s.trace_id == trace_id]

    def clear(self) -> None:
        self._spans.clear()


class Tracer:
    """Creates and manages spans for tracing agent execution.

    Example:
        tracer = Tracer()
        async with tracer.span("agent.run", kind="agent") as s:
            s.attributes["input"] = user_input
            result = await do_work()
            s.attributes["output"] = result
    """

    def __init__(self, collector: TraceCollector | None = None, enabled: bool = True) -> None:
        self._collector = collector or TraceCollector()
        self._enabled = enabled

    @property
    def collector(self) -> TraceCollector:
        return self._collector

    @asynccontextmanager
    async def span(
        self, name: str, kind: str = "internal", **attributes: Any,
    ) -> AsyncIterator[Span]:
        """Create a span as an async context manager.

        Uses contextvars for async-safe parent tracking, so concurrent
        coroutines sharing the same Tracer instance get correct
        parent-child relationships.
        """
        if not self._enabled:
            yield Span(name=name, kind=kind)
            return

        trace_id = _current_trace_id.get() or uuid.uuid4().hex
        parent_span_id = _current_span_id.get()

        s = Span(
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,
            attributes=attributes,
        )

        token_trace = _current_trace_id.set(trace_id)
        token_span = _current_span_id.set(s.span_id)

        try:
            yield s
        except Exception as e:
            s.set_error(e)
            raise
        finally:
            s.finish()
            self._collector.add_span(s)
            _current_trace_id.reset(token_trace)
            _current_span_id.reset(token_span)

    def trace(self, name: str, kind: str = "internal") -> Callable[..., Any]:
        """Decorator to trace an async function."""
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                async with self.span(name, kind=kind) as s:
                    s.attributes["args_count"] = len(args)
                    result = await fn(*args, **kwargs)
                    return result
            return wrapper
        return decorator
