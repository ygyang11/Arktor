"""Trace collector for aggregating spans."""
from __future__ import annotations

# TraceCollector is defined in tracer.py to avoid circular imports.
# This module re-exports it and may add additional collectors in the future.
from agent_harness.tracing.tracer import TraceCollector

__all__ = ["TraceCollector"]
