"""Plan-mode runtime — process-local on/off flag keyed by agent identity.

Phase 1 ships only the state-tracking surface. Phase 5 will extend this
module with the ContextPatch that injects the plan-mode system reminder.
"""
from __future__ import annotations

from agent_harness.agent.base import BaseAgent

_active: set[int] = set()


def is_active(agent: BaseAgent) -> bool:
    return id(agent) in _active


def enter(agent: BaseAgent) -> None:
    _active.add(id(agent))


def exit(agent: BaseAgent) -> None:
    _active.discard(id(agent))
