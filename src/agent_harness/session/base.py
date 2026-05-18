"""Session state model and abstract base for session backends."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from agent_harness.core.message import Message, Role

_SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _first_user_text(messages: list[Message]) -> str:
    """Raw content of the first USER message, untransformed."""
    for m in messages:
        if m.role == Role.USER and m.content:
            return m.content
    return ""


class SessionState(BaseModel):
    """Snapshot of all agent state that needs to survive process restarts."""

    session_id: str
    messages: list[Message] = Field(default_factory=list)
    working_memory_scratchpad: dict[str, Any] = Field(default_factory=dict)
    working_memory_history: list[Message] = Field(default_factory=list)
    variables_agent: dict[str, Any] = Field(default_factory=dict)
    variables_global: dict[str, Any] = Field(default_factory=dict)
    agent_state: str = "idle"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class SessionMeta(BaseModel):
    """Lightweight session descriptor for listing without loading full state."""

    session_id: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    first_user_preview: str = ""

    @classmethod
    def from_state(cls, state: SessionState) -> SessionMeta:
        return cls(
            session_id=state.session_id,
            created_at=state.created_at,
            updated_at=state.updated_at,
            message_count=len(state.messages),
            first_user_preview=_first_user_text(state.messages),
        )


class BaseSession(ABC):
    """Abstract base for session persistence backends."""

    @staticmethod
    def _is_valid_id(session_id: str) -> bool:
        return bool(_SAFE_ID_PATTERN.match(session_id))

    def __init__(self, session_id: str) -> None:
        if not self._is_valid_id(session_id):
            raise ValueError(
                f"session_id must match [a-zA-Z0-9_-], got: {session_id!r}"
            )
        self.session_id = session_id

    def set_session_id(self, new_id: str) -> None:
        if not self._is_valid_id(new_id):
            raise ValueError(
                f"session_id must match [a-zA-Z0-9_-], got: {new_id!r}"
            )
        self.session_id = new_id

    async def rename(self, new_id: str) -> None:
        self.set_session_id(new_id)

    @abstractmethod
    async def load_state(self) -> SessionState | None: ...

    @abstractmethod
    async def save_state(self, state: SessionState) -> None: ...

    @abstractmethod
    async def clear(self) -> None: ...

    @abstractmethod
    async def has_session(self, session_id: str) -> bool: ...

    @abstractmethod
    async def list_states(self) -> list[SessionMeta]: ...


def resolve_session(session: str | BaseSession | None) -> BaseSession | None:
    """Convert a session parameter to a BaseSession instance.

    Accepts str (auto-creates FileSession), BaseSession (pass-through), or None.
    """
    if session is None:
        return None
    if isinstance(session, str):
        from agent_harness.session.file_session import FileSession
        return FileSession(session)
    return session
