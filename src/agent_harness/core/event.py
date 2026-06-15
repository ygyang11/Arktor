"""Event system for agent_harness.

Provides decoupled, async event-driven communication between components.
Supports wildcard matching on dot-separated event type namespaces.
"""
from __future__ import annotations

import asyncio
import fnmatch
import logging
from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Event(BaseModel):
    """An event emitted by a framework component."""
    type: str  # dot-separated namespace, e.g. "agent.step.start"
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
    source: str | None = None  # component that emitted the event

# Type alias for event handlers
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """Central event bus supporting async handlers and wildcard subscriptions.

    Wildcard matching:
        - "agent.*" matches "agent.step.start", "agent.run.end", etc.
        - "*" matches everything
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}
        self._lock = asyncio.Lock()

    async def emit(self, event: Event) -> None:
        """Emit an event, invoking all matching handlers concurrently."""
        matching_handlers: list[EventHandler] = []
        async with self._lock:
            for pattern, handlers in self._handlers.items():
                if fnmatch.fnmatch(event.type, pattern):
                    matching_handlers.extend(handlers)

        if not matching_handlers:
            return

        results = await asyncio.gather(
            *(h(event) for h in matching_handlers),
            return_exceptions=True,
        )
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Event handler error for '%s': %s", event.type, result, exc_info=result
                )

    def on(self, event_type: str, handler: EventHandler) -> None:
        """Register a handler for an event type pattern."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def off(self, event_type: str, handler: EventHandler) -> None:
        """Remove a handler."""
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h is not handler
            ]
            if not self._handlers[event_type]:
                del self._handlers[event_type]

    def clear(self) -> None:
        """Remove all handlers."""
        self._handlers.clear()


class EventEmitter:
    """Mixin that gives any component the ability to emit events.

    Usage:
        class MyComponent(EventEmitter):
            async def do_work(self):
                await self.emit("my_component.work.start", result="ok")
    """

    _event_bus: EventBus | None = None

    def set_event_bus(self, bus: EventBus) -> None:
        self._event_bus = bus

    async def emit(self, event_type: str, *, source: str | None = None, **data: Any) -> None:
        """Emit an event through the attached EventBus."""
        if self._event_bus is None:
            return
        event = Event(
            type=event_type,
            data=data,
            source=source or getattr(self, "name", self.__class__.__name__),
        )
        await self._event_bus.emit(event)
