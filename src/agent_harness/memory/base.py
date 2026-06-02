"""Base memory interface for agent_harness.

Memory provides agents with the ability to store and retrieve information
across conversation turns and across sessions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from agent_harness.core.message import Message


class MemoryItem(BaseModel):
    """A single item stored in memory."""
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance_score: float | None = None  # importance/relevance score
    time_score: float = 0.0  # time decay score
    timestamp: datetime = Field(default_factory=lambda: datetime.now())

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        score_str = (
            f", importance={self.importance_score:.3f}"
            if self.importance_score is not None else ""
        )
        return f"MemoryItem({preview!r}{score_str})"


class BaseMemory(ABC):
    """Abstract base class for all memory implementations.

    Memory implementations store and retrieve information for agents.
    The interface is intentionally simple to allow diverse backends:
    - Conversation buffer (short-term)
    - Vector store (long-term)
    - Key-value scratchpad (working)
    """

    @abstractmethod
    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add content to memory.

        Args:
            content: The text content to store.
            metadata: Optional metadata associated with the content.
        """
        ...

    @abstractmethod
    async def add_message(self, message: Message) -> None:
        """Add a message to memory.

        Args:
            message: The message to store.
        """
        ...

    @abstractmethod
    async def query(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        """Query memory for relevant items.

        Args:
            query: The query string.
            top_k: Maximum number of items to return.

        Returns:
            List of MemoryItems, sorted by relevance (if applicable).
        """
        ...

    @abstractmethod
    async def get_context_messages(self) -> list[Message]:
        """Get messages suitable for inclusion in LLM context.

        Returns:
            List of Messages representing the memory contents.
        """
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Clear all stored content."""
        ...

    @abstractmethod
    async def forget(self, threshold: float = 0.3) -> int:
        """Remove memories with weighted score below threshold.

        The weighted score is computed as:
            time_decay * (0.8 + importance_score * 0.4)

        Args:
            threshold: Memories with weighted score below this are removed.

        Returns:
            Number of items removed.
        """
        ...

    @abstractmethod
    async def size(self) -> int:
        """Return the number of items currently in memory."""
        ...
