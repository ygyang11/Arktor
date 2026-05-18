"""`@path` mention handling for REPL input."""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from agent_cli.adapter import CliAdapter
from agent_cli.render.notices import format_attachment_reminders
from agent_cli.render.tool_display import attachment_summary
from agent_harness.agent.base import BaseAgent
from agent_harness.core.message import Role, ToolCall, ToolResult

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


async def expand_mentions(
    agent: BaseAgent, adapter: CliAdapter, text: str,
) -> None:
    """Execute @ mentions concurrently, render the live indicator, and
    embed the result as ``<system-reminder>`` blocks on the user message.

    Cancel during execute_stream discards partial results before the
    in-memory mutate, leaving the user message untouched.
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
    await adapter.render_attachments(pairs)
    await embed_attachments_into_last_user(agent, pairs)


async def embed_attachments_into_last_user(
    agent: BaseAgent, pairs: list[tuple[ToolCall, ToolResult]],
) -> None:
    """Prepend reminder blocks to the last user message content and record
    the attachment summary into its metadata (for replay reconstruction)."""
    from agent_cli.runtime.session import get_messages  # noqa: PLC0415

    msgs = get_messages(agent)
    if not msgs:
        return
    last = msgs[-1]
    if last.role != Role.USER:
        return

    blocks = [format_attachment_reminders(tc, tr) for tc, tr in pairs]
    summaries = [attachment_summary(tc, tr) for tc, tr in pairs]

    original = last.content or ""
    prefix = "\n\n".join(blocks)
    last.content = f"{prefix}\n\n{original}" if original else prefix
    last.metadata.setdefault("attachments", []).extend(summaries)


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
