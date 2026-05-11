"""In-memory session backend for testing and single-process use."""
from __future__ import annotations

from agent_harness.session.base import BaseSession, SessionMeta, SessionState


class InMemorySession(BaseSession):

    def __init__(self, session_id: str) -> None:
        super().__init__(session_id)
        self._states: dict[str, SessionState] = {}

    async def load_state(self) -> SessionState | None:
        return self._states.get(self.session_id)

    async def save_state(self, state: SessionState) -> None:
        self._states[self.session_id] = state

    async def clear(self) -> None:
        self._states.pop(self.session_id, None)

    async def has_session(self, session_id: str) -> bool:
        if not self._is_valid_id(session_id):
            return False
        return session_id in self._states

    async def list_states(self) -> list[SessionMeta]:
        return sorted(
            (SessionMeta.from_state(s) for s in self._states.values()),
            key=lambda m: m.updated_at,
            reverse=True,
        )
