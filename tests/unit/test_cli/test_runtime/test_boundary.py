"""Guardrail: no direct private-field access outside runtime/ / adapter."""
from __future__ import annotations

import re
from pathlib import Path

import agent_cli

_CLI_ROOT = Path(agent_cli.__file__).resolve().parent

_FORBIDDEN = [
    re.compile(r"\._bg_manager\b"),
    re.compile(r"\._collect_background_results\b"),
    re.compile(r"\._approval\b"),
    re.compile(r"\._sandbox\b"),
    re.compile(r"\._session_created_at\b"),
    re.compile(r"\bcompressor\._model\b"),
    re.compile(r"\bcompressor\._llm\b"),
    re.compile(r"\._messages\b"),
]

_ALLOWED = {
    _CLI_ROOT / "runtime" / "background.py",
    _CLI_ROOT / "runtime" / "session.py",
    _CLI_ROOT / "runtime" / "status.py",
}


def test_no_private_access_outside_runtime() -> None:
    violations: list[str] = []
    for path in _CLI_ROOT.rglob("*.py"):
        if path in _ALLOWED:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in _FORBIDDEN:
            for m in pattern.finditer(text):
                line_no = text[: m.start()].count("\n") + 1
                violations.append(
                    f"{path.relative_to(_CLI_ROOT)}:{line_no}: {m.group()}"
                )
    assert not violations, (
        "Private framework access must go through runtime/:\n"
        + "\n".join(violations)
    )
