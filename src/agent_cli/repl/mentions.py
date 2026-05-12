"""`@path` mention handling for REPL input."""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from agent_cli.adapter import CliAdapter
from agent_cli.runtime.conversation import append_tool_turn
from agent_harness.agent.base import BaseAgent
from agent_harness.core.message import Message, ToolCall, ToolResult

_AT_RE = re.compile(r"(?:^|(?<=\s))@([^\s]*)$")
_MENTION_RE = re.compile(r"(?:^|(?<=\s))@([^\s]+)")

_READ_LIMIT = 500


def find_at_token(text_before_cursor: str) -> tuple[int, str] | None:
    m = _AT_RE.search(text_before_cursor)
    if m is None:
        return None
    return m.start(), m.group(1)


def parse_mentions(text: str) -> list[str]:
    return [m.group(1) for m in _MENTION_RE.finditer(text)]


def is_attachment_turn(user_msg: Message, asst_msg: Message) -> bool:
    """True if `(user_msg, asst_msg)` is the persistence shape produced
    by a `@file` mention expansion.
    """
    if asst_msg.content:
        return False
    tcs = asst_msg.tool_calls or []
    if not tcs:
        return False
    if not user_msg.content:
        return False
    mention_paths = set(parse_mentions(user_msg.content))
    if not mention_paths:
        return False
    return all(
        any(isinstance(v, str) and v in mention_paths for v in tc.arguments.values())
        for tc in tcs
    )


async def expand_mentions(
    agent: BaseAgent, adapter: CliAdapter, text: str,
) -> None:
    """Execute @ mentions concurrently, render, splice into memory.

    Cancel during execute_stream discards partial results and skips
    the memory write; shield only protects the write phase.
    """
    raw = parse_mentions(text)
    if not raw:
        return

    calls = _build_calls(agent, raw)
    if not calls:
        return

    tcs = [
        ToolCall(id=_new_id(), name=name, arguments=args)
        for name, args in calls
    ]

    by_id: dict[str, ToolResult] = {}
    async for tr in agent.tool_executor.execute_stream(tcs):
        by_id[tr.tool_call_id] = tr

    pairs = [(tc, by_id[tc.id]) for tc in tcs]
    await append_tool_turn(agent, pairs, render=adapter.render_attachments)


def _build_calls(
    agent: BaseAgent, paths: list[str],
) -> list[tuple[str, dict[str, Any]]]:
    """Resolve, classify, dedupe, filter against workspace + registry."""
    root = Path.cwd().resolve()
    seen: set[str] = set()
    out: list[tuple[str, dict[str, Any]]] = []
    for raw in paths:
        if raw.startswith("~"):
            continue
        candidate = Path(raw) if Path(raw).is_absolute() else root / raw
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not _within(resolved, root):
            continue
        if not resolved.exists():
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        name: str
        args: dict[str, Any]
        if resolved.is_dir():
            name = "list_dir"
            args = {"path": raw}
        else:
            name = "read_file"
            args = {"file_path": raw, "limit": _READ_LIMIT}
        if not agent.tool_registry.has(name):
            continue
        out.append((name, args))
    return out


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _new_id() -> str:
    return f"inj_{uuid.uuid4().hex[:12]}"
