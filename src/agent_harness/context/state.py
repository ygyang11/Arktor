"""Agent execution state management."""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any


class AgentState(str, Enum):
    """Possible states of an agent during execution."""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    PLANNING = "planning"
    FINISHED = "finished"
    ERROR = "error"


# Valid state transitions
_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.IDLE: {AgentState.THINKING, AgentState.PLANNING, AgentState.FINISHED},
    AgentState.THINKING: {
        AgentState.ACTING, AgentState.FINISHED, AgentState.ERROR, AgentState.PLANNING,
    },
    AgentState.ACTING: {AgentState.OBSERVING, AgentState.ERROR},
    AgentState.OBSERVING: {
        AgentState.THINKING, AgentState.FINISHED, AgentState.PLANNING, AgentState.ERROR,
    },
    AgentState.PLANNING: {
        AgentState.THINKING, AgentState.ACTING, AgentState.FINISHED, AgentState.ERROR,
    },
    AgentState.FINISHED: {AgentState.IDLE},  # can reset
    AgentState.ERROR: {AgentState.IDLE},  # can reset
}


class StateManager:
    """Manages agent execution state with transition validation and history."""

    def __init__(self, initial: AgentState = AgentState.IDLE) -> None:
        self._current = initial
        self._history: list[tuple[AgentState, datetime]] = [
            (initial, datetime.now())
        ]
        self._callbacks: list[Callable[[AgentState, AgentState], Any]] = []

    @property
    def current(self) -> AgentState:
        return self._current

    @property
    def history(self) -> list[tuple[AgentState, datetime]]:
        return list(self._history)

    @property
    def is_terminal(self) -> bool:
        return self._current in (AgentState.FINISHED, AgentState.ERROR)

    def transition(self, new_state: AgentState) -> None:
        """Transition to a new state with validation."""
        from agent_harness.core.errors import StateTransitionError

        valid = _TRANSITIONS.get(self._current, set())
        if new_state not in valid:
            raise StateTransitionError(
                f"Invalid transition: {self._current.value} -> {new_state.value}. "
                f"Valid transitions: {[s.value for s in valid]}"
            )
        old = self._current
        self._current = new_state
        self._history.append((new_state, datetime.now()))
        for cb in self._callbacks:
            cb(old, new_state)

    def on_transition(self, callback: Callable[[AgentState, AgentState], Any]) -> None:
        """Register a callback for state transitions. callback(old_state, new_state)."""
        self._callbacks.append(callback)

    def reset(self) -> None:
        """Reset to IDLE state."""
        self._current = AgentState.IDLE
        self._history.append((AgentState.IDLE, datetime.now()))

    def __repr__(self) -> str:
        return f"StateManager(current={self._current.value})"
