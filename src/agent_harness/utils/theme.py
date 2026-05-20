"""Shared terminal color codes and icon definitions."""
from __future__ import annotations

COLORS: dict[str, str] = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "reset": "\033[0m",
}

ICONS: dict[str, str] = {
    "tool": "⚡",
    "approval": "🔒",
    "marker": "⏺",
    "check": "✓",
    "cross": "✗",
    "denied": "⊘",
    "error": "❌",
    "pending": "⏳",
    "summary": "⎿",
    "double_check": "✓✓",
    "todo_done": "✓",
    "todo_active": "●",
    "todo_pending": "○",
    "subagent": "🤖",
    "heal": "↻",
}
