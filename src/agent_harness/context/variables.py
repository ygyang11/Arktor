"""Type-safe shared variables with scope support."""
from __future__ import annotations

from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T")


class Scope(str, Enum):
    """Variable scope."""
    AGENT = "agent"    # Private to the current agent
    GLOBAL = "global"  # Shared across all agents in the team


class ContextVariables:
    """Type-safe shared variable storage with agent/global scoping.

    AGENT-scoped variables are private per agent.
    GLOBAL-scoped variables are shared across agents via fork().
    """

    def __init__(self, global_store: dict[str, Any] | None = None) -> None:
        self._agent_store: dict[str, Any] = {}
        self._global_store: dict[str, Any] = global_store if global_store is not None else {}

    def set(self, key: str, value: Any, scope: Scope = Scope.AGENT) -> None:
        """Set a variable."""
        store = self._global_store if scope == Scope.GLOBAL else self._agent_store
        store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a variable. Checks agent scope first, then global."""
        if key in self._agent_store:
            return self._agent_store[key]
        return self._global_store.get(key, default)

    def get_typed(self, key: str, type_: type[T], default: T | None = None) -> T:
        """Get a variable with type checking."""
        value = self.get(key, default)
        if value is not None and not isinstance(value, type_):
            raise TypeError(
                f"Variable '{key}' expected type {type_.__name__}, got {type(value).__name__}"
            )
        return value  # type: ignore[return-value]

    def has(self, key: str) -> bool:
        return key in self._agent_store or key in self._global_store

    def delete(self, key: str) -> None:
        self._agent_store.pop(key, None)
        self._global_store.pop(key, None)

    def get_all(self, scope: Scope | None = None) -> dict[str, Any]:
        """Get all variables, optionally filtered by scope."""
        if scope == Scope.AGENT:
            return dict(self._agent_store)
        if scope == Scope.GLOBAL:
            return dict(self._global_store)
        merged = dict(self._global_store)
        merged.update(self._agent_store)
        return merged

    def fork(self) -> ContextVariables:
        """Create a child ContextVariables sharing the same global store but fresh agent store."""
        return ContextVariables(global_store=self._global_store)

    def __repr__(self) -> str:
        return (
            f"ContextVariables(agent={list(self._agent_store)}, "
            f"global={list(self._global_store)})"
        )
