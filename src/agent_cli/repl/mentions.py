"""`@path` mention handling for REPL input."""
from __future__ import annotations

import mimetypes
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_cli.adapter import CliAdapter
from agent_cli.render.notices import format_attachment_reminders
from agent_cli.render.tool_display import attachment_summary, media_attachment_summary
from agent_harness.agent.base import BaseAgent
from agent_harness.core.message import Attachment, Role, ToolCall, ToolResult
from agent_harness.utils.blob import make_attachment
from agent_harness.utils.media import is_media_mime

_AT_RE = re.compile(r"(?:^|(?<=\s))@([^\s]*)$")
_MENTION_RE = re.compile(r"(?:^|(?<=\s))@([^\s]+)")

_READ_LIMIT = 500


@dataclass
class AttachmentItem:
    summary: dict[str, Any]
    text: str | None = None
    media: Attachment | None = None


@dataclass
class _CallSpec:
    name: str
    args: dict[str, Any]


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
    """Process @ mentions in input order: build items, execute tool calls,
    render and embed onto the last user message in unified shape."""
    raw = parse_mentions(text)
    if not raw:
        return

    items: list[AttachmentItem | _CallSpec] = _build_items(
        agent, raw, Path.cwd().resolve(),
    )
    if not items:
        return

    specs = [(i, it) for i, it in enumerate(items) if isinstance(it, _CallSpec)]
    if specs:
        tcs = [
            ToolCall(id=_new_id(), name=s.name, arguments=s.args)
            for _, s in specs
        ]
        by_id: dict[str, ToolResult] = {}
        async for tr in agent.tool_executor.execute_stream(tcs):
            by_id[tr.tool_call_id] = tr
        for (idx, _), tc in zip(specs, tcs):
            tr = by_id[tc.id]
            items[idx] = AttachmentItem(
                summary=attachment_summary(tc, tr),
                text=format_attachment_reminders(tc, tr),
            )

    resolved: list[AttachmentItem] = [
        it for it in items if isinstance(it, AttachmentItem)
    ]
    if not resolved:
        return
    await adapter.render_attachments([it.summary for it in resolved])
    await embed_attachments_into_last_user(agent, resolved)


async def embed_attachments_into_last_user(
    agent: BaseAgent, items: list[AttachmentItem],
) -> None:
    from agent_cli.runtime.session import get_messages  # noqa: PLC0415

    if not items:
        return
    msgs = get_messages(agent)
    if not msgs:
        return
    last = msgs[-1]
    if last.role != Role.USER:
        return

    reminders = [it.text for it in items if it.text]
    media = [it.media for it in items if it.media is not None]
    summaries = [it.summary for it in items]

    if reminders:
        prefix = "\n\n".join(reminders)
        original = last.content or ""
        last.content = f"{prefix}\n\n{original}" if original else prefix
    if media:
        last.attachments = list(last.attachments or []) + media
    last.metadata.setdefault("attachments", []).extend(summaries)


def _build_items(
    agent: BaseAgent, paths: list[str], root: Path,
) -> list[AttachmentItem | _CallSpec]:
    """Resolve, classify, dedupe, filter against workspace + registry."""
    items: list[AttachmentItem | _CallSpec] = []
    for resolved in _resolve_unique(paths, root):
        rel = str(resolved.relative_to(root))
        if resolved.is_dir():
            if agent.tool_registry.has("list_dir"):
                items.append(_CallSpec(name="list_dir", args={"path": rel}))
            continue
        mime, _ = mimetypes.guess_type(resolved.name)
        if mime and is_media_mime(mime):
            att = make_attachment(resolved.read_bytes(), mime, resolved.name)
            items.append(AttachmentItem(
                summary=media_attachment_summary(att), media=att,
            ))
            continue
        if agent.tool_registry.has("read_file"):
            items.append(_CallSpec(
                name="read_file",
                args={"file_path": rel, "limit": _READ_LIMIT},
            ))
    return items


def _resolve_unique(paths: list[str], root: Path) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for raw in paths:
        if raw.startswith("~"):
            continue
        candidate = Path(raw) if Path(raw).is_absolute() else root / raw
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not _within(resolved, root) or not resolved.exists():
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
    return out


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _new_id() -> str:
    return f"inj_{uuid.uuid4().hex[:12]}"
