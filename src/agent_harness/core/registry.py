"""Generic registry pattern for named component discovery."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A generic, thread-safe registry for named components.

    Used as the base for ToolRegistry, PromptLibrary, and other registries.
    """

    def __init__(self) -> None:
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T) -> None:
        """Register an item by name. Overwrites if name already exists."""
        self._items[name] = item

    def get(self, name: str) -> T:
        """Get an item by name. Raises KeyError if not found."""
        if name not in self._items:
            raise KeyError(f"'{name}' not found in registry. Available: {list(self._items.keys())}")
        return self._items[name]

    def has(self, name: str) -> bool:
        """Check if a name is registered."""
        return name in self._items

    def unregister(self, name: str) -> None:
        """Remove an item. Raises KeyError if not found."""
        if name not in self._items:
            raise KeyError(f"'{name}' not found in registry")
        del self._items[name]

    def list_names(self) -> list[str]:
        """List all registered names."""
        return list(self._items.keys())

    def list_all(self) -> dict[str, T]:
        """Return a copy of all registered items."""
        return dict(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def clear(self) -> None:
        """Remove all items."""
        self._items.clear()
