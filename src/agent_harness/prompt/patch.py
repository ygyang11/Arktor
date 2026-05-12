"""ContextPatch — dynamic context message injection for call_llm."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from agent_harness.core.message import Message


@dataclass(frozen=True, slots=True)
class ContextPatch:
    at: Literal["system", "tail"]
    build: Callable[[], Message | None]
