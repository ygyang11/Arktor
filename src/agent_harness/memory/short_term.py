"""Short-term memory: conversation buffer with optional compression and token trim."""
from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from agent_harness.core.message import Message, Role
from agent_harness.memory.base import BaseMemory, MemoryItem
from agent_harness.memory.retrieval import HybridRetriever
from agent_harness.utils.token_counter import count_messages_tokens

if TYPE_CHECKING:
    from agent_harness.memory.compressor import ContextCompressor

logger = logging.getLogger(__name__)


class SectionWeights(BaseModel):
    """Per-section token-count weights captured at LLM call time.

    Used proportionally by /context render. Absolute magnitude cancels in
    calibration to provider-truth prompt_tokens; only ratios matter.
    """
    system_prompt: int
    tools_schema: int
    dynamic_system: int
    history: int


class CallSnapshot(BaseModel):
    """Frozen snapshot of the most recent LLM call.

    Persisted into SessionState.metadata['_call_snapshot'] for resume.
    Drives status bar / /context / compressor trigger / /model invalidation.
    """
    model_config = ConfigDict(extra="ignore")

    input_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_read: int
    cache_creation: int
    reasoning_tokens: int = 0
    model: str
    message_count: int = 0
    section_weights: SectionWeights


class ShortTermMemory(BaseMemory):
    """Conversation buffer with optional LLM compression and token-based trim fallback."""

    def __init__(
        self,
        max_tokens: int = 100000,
        model: str = "gpt-4o",
        compressor: ContextCompressor | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.model = model
        self._messages: list[Message] = []
        self.compressor = compressor
        self.last_call: CallSnapshot | None = None

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        await self.add_message(Message.user(content, metadata=metadata or {}))

    async def add_message(self, message: Message) -> None:
        self._messages.append(message)

    async def query(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        items = [
            MemoryItem(
                content=msg.content or "",
                metadata={"role": msg.role.value},
                importance_score=msg.metadata.get("importance_score", 0.5),
                timestamp=msg.created_at,
            )
            for msg in self._messages
            if msg.content
        ]
        if not items:
            return []
        retriever = HybridRetriever()
        results: list[MemoryItem] = retriever.retrieve(query, items, top_k=top_k)
        return results

    async def get_context_messages(self) -> list[Message]:
        """Pure read of current buffer. No mutation, no compress trigger."""
        return list(self._messages)

    def record_call(self, snapshot: CallSnapshot) -> None:
        """Atomically install snapshot — only legit writer (called from BaseAgent.call_llm)."""
        self.last_call = snapshot

    def clear_call_snapshot(self) -> None:
        self.last_call = None

    def replace_messages(self, new_messages: list[Message]) -> None:
        """Atomic replace + invalidate snapshot."""
        self._messages = list(new_messages)
        self.clear_call_snapshot()

    @property
    def displayed_input_tokens(self) -> int | None:
        if self.last_call is None:
            return None

        base = self.last_call.total_tokens
        if self.last_call.reasoning_tokens > 0:
            last_idx = self.last_call.message_count - 1
            last_msg = (
                self._messages[last_idx]
                if 0 <= last_idx < len(self._messages)
                else None
            )
            persisted = (
                last_msg is not None
                and last_msg.role == Role.ASSISTANT
                and any(
                    any(any(item.values()) for item in v if isinstance(item, dict))
                    if isinstance(v, list)
                    else v
                    for ns in last_msg.provider_metadata.values()
                    for v in ns.values()
                )
            )
            if not persisted:
                base -= self.last_call.reasoning_tokens

        delta = self._messages[self.last_call.message_count:]
        if not delta:
            return base
        return base + count_messages_tokens(delta, self.model)

    async def clear(self) -> None:
        self._messages.clear()
        self.clear_call_snapshot()

    async def forget(self, threshold: float = 0.3) -> int:
        now = datetime.now()
        decay_rate = 0.01
        original_count = len(self._messages)

        kept: list[Message] = []
        for msg in self._messages:
            if msg.role == Role.SYSTEM:
                kept.append(msg)
                continue
            hours = (now - msg.created_at).total_seconds() / 3600
            time_decay = math.exp(-decay_rate * hours)
            importance = msg.metadata.get("importance_score", 0.5)
            weighted = time_decay * (0.8 + importance * 0.4)
            if weighted >= threshold:
                kept.append(msg)

        if len(kept) != original_count:
            self._messages = kept
            self.clear_call_snapshot()
        return original_count - len(self._messages)

    async def size(self) -> int:
        return len(self._messages)

    @property
    def token_count(self) -> int:
        """Local tiktoken estimate; deprecated — prefer displayed_input_tokens (provider truth)."""
        return count_messages_tokens(self._messages, self.model)

    @staticmethod
    def _trim_by_tokens(
        messages: list[Message], max_tokens: int, model: str
    ) -> list[Message]:
        from agent_harness.memory.compressor import ContextCompressor

        current = count_messages_tokens(messages, model)
        if current <= max_tokens:
            return messages

        groups = ContextCompressor._group_atomic_pairs(messages)
        system_groups = [g for g in groups if g.is_protected_system]
        non_system_groups = [g for g in groups if not g.is_protected_system]

        system_msgs = [m for g in system_groups for m in g.messages]
        budget = max_tokens - count_messages_tokens(system_msgs, model)
        kept_groups: list[list[Message]] = []

        for group in reversed(non_system_groups):
            group_tokens = count_messages_tokens(group.messages, model)
            if budget - group_tokens < 0:
                break
            kept_groups.append(group.messages)
            budget -= group_tokens

        kept_groups.reverse()
        kept_msgs = [m for group in kept_groups for m in group]
        return system_msgs + kept_msgs
