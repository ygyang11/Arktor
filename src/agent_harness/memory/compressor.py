"""LLM-driven context compression for ShortTermMemory."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_harness.core.message import Message, Role, ToolResult
from agent_harness.llm.types import Usage
from agent_harness.utils.media import (
    describe_attachment_full,
    describe_attachment_short,
)
from agent_harness.utils.token_counter import count_messages_tokens, count_tokens

if TYPE_CHECKING:
    from agent_harness.core.config import MemoryConfig
    from agent_harness.llm.base import BaseLLM

logger = logging.getLogger(__name__)

_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")
_FILENAME_ID_MAX = 96

_SUMMARY_SYSTEM_PROMPT = """\
You are a context compressor for an AI agent conversation. The conversation is \
approaching its token limit and you must extract the most important information \
to replace the older portion, so the agent can continue working effectively.

## Output Format

Produce a summary with these sections (omit sections that have no content):

### User Goal
The user's original request and any clarifications or scope changes.

### Key Decisions
Decisions made during the conversation, with brief rationale. \
Include what was considered and rejected, and why.

### Completed Work
What has been accomplished:
- Tool calls and their significant results (preserve exact data, not raw output)
- Files read, written, or modified (preserve exact paths)
- Media attachments supplied or fetched (preserve filename and MIME type)
- Commands executed and their outcomes
- Errors encountered and how they were resolved

### Current State
Where things stand right now: what is done and what is in progress.

### Next Steps
What is pending to continue the task, in execution order.

### Important Context
Facts, constraints, requirements, or user preferences that affect ongoing work.

## Rules
- Be factual and specific. Preserve exact file paths, function names, error messages, \
URLs, numeric values, and media attachments when referenced.
- Summarize tool results by their significance, not raw output.
- Drop: intermediate reasoning steps, verbose tool output already captured in results, \
superseded plans, pleasantries.
- Do NOT add interpretation, suggestions, or conclusions beyond what was explicitly stated.
- If the conversation already contains a previous compression summary, integrate its \
content — do not nest summaries.\
"""

_SUMMARY_USER_TEMPLATE = """\
Compress the older conversation below into a structured summary.

The recent messages (in <recent_context>) are NOT being compressed — they are \
provided as reference so you understand the current direction, but your summary \
should be a self-contained record of the older conversation's key information, \
not biased toward recent activity.

<older_conversation>
{older_conversation}
</older_conversation>

<recent_context>
{recent_conversation}
</recent_context>\
{extra_instructions_block}"""

_CONTINUATION_FOOTER = (
    "\n\n---\n"
    "Recent messages below are preserved verbatim. "
    "Continue the conversation from where it left off. "
    "Do not acknowledge this summary, do not recap, and do not ask follow-up "
    "questions about the summarized content — resume directly."
)


class _MessageGroup:
    """A group of messages that must be compressed/retained together."""

    __slots__ = ("messages",)

    def __init__(self, messages: list[Message]) -> None:
        self.messages = messages

    @property
    def is_system(self) -> bool:
        return len(self.messages) == 1 and self.messages[0].role == Role.SYSTEM

    @property
    def is_protected_system(self) -> bool:
        """Agent identity prompt — never compressed.
        Compression summaries and background results are NOT protected."""
        if not self.is_system:
            return False
        meta = self.messages[0].metadata
        return (
            not meta.get("is_compression_summary", False)
            and not meta.get("is_background_result", False)
        )


@dataclass(slots=True)
class CompressionResult:
    original_count: int
    compressed_count: int
    summary_tokens: int
    archive_path: str | None = None
    llm_usage: Usage | None = None


class ContextCompressor:
    """LLM-driven context compression for ShortTermMemory.

    Supports two invocation modes:
    - Automatic: triggered by should_compress() inside get_context_messages()
    - Manual: called directly via compress(messages, extra_instructions="...")
      for future /compact CLI command
    """

    def __init__(
        self,
        llm: BaseLLM,
        *,
        threshold: float = 0.75,
        retain_count: int = 6,
        summary_max_tokens: int | None = None,
        model: str,
        session_id: str | None = None,
        scope: str = "main",
        prune_per_output_threshold: int = 15_000,
        prune_tail_turns: int = 3,
    ) -> None:
        self._llm = llm
        self._threshold = threshold
        self._retain_count = retain_count
        self._summary_max_tokens = summary_max_tokens
        self._model = model
        self._session_id = session_id
        self._scope = self._normalize_scope(scope)
        self._prune_per_output_threshold = prune_per_output_threshold
        self._prune_tail_turns = prune_tail_turns
        self._compression_count: int = 0
        self._archive_paths: list[str] = []
        self._last_result: CompressionResult | None = None

    @property
    def scope(self) -> str:
        return self._scope

    @property
    def model_name(self) -> str:
        return self._llm.model_name

    def clone(self, *, scope: str | None = None) -> ContextCompressor:
        return ContextCompressor(
            llm=self._llm,
            threshold=self._threshold,
            retain_count=self._retain_count,
            summary_max_tokens=self._summary_max_tokens,
            model=self._model,
            session_id=self._session_id,
            scope=scope or self._scope,
            prune_per_output_threshold=self._prune_per_output_threshold,
            prune_tail_turns=self._prune_tail_turns,
        )

    def bind_session(self, session_id: str | None) -> None:
        self._session_id = session_id

    def restore_runtime_state(self, messages: list[Message]) -> None:
        self._compression_count = 0
        self._archive_paths = []
        self._last_result = None

        for msg in messages:
            if not msg.metadata.get("is_compression_summary"):
                continue

            round_num = int(msg.metadata.get("compression_round", 0) or 0)
            archive_paths = msg.metadata.get("archive_paths")
            self._compression_count = max(self._compression_count, round_num)

            if isinstance(archive_paths, list):
                self._archive_paths = [str(path) for path in archive_paths]

    def should_compress(
        self,
        messages: list[Message],
        max_tokens: int,
        *,
        authoritative_input: int | None = None,
    ) -> bool:
        if not messages or max_tokens <= 0:
            return False
        if authoritative_input is not None:
            return authoritative_input > max_tokens * self._threshold
        current: int = count_messages_tokens(messages, self._model)
        return current > max_tokens * self._threshold

    async def compress(
        self,
        messages: list[Message],
        *,
        extra_instructions: str | None = None,
    ) -> list[Message]:
        """Compress older messages into a summary.

        Returns:
            New message list: [protected system] + [summary(system)] + [recent].
            Summary is system-role with is_compression_summary metadata,
            so _partition treats it as non_system (telescoping).
        """
        self._last_result = None
        groups = self._group_atomic_pairs(messages)
        system, older, recent = self._partition(groups)

        if not older:
            return messages

        older_msgs = [m for g in older for m in g.messages]
        recent_msgs = [m for g in recent for m in g.messages]

        summary_text, summary_usage = await self._summarize(
            older_msgs, recent_msgs, extra_instructions
        )

        self._compression_count += 1
        archive_path: str | None = None
        if self._session_id:
            archive_path = str(self._archive(older_msgs, self._compression_count))
            self._archive_paths.append(archive_path)

        summary_content = self._build_summary_content(summary_text, archive_path)
        summary_msg = Message.system(
            summary_content,
            metadata={
                "is_compression_summary": True,
                "compression_round": self._compression_count,
                "archive_paths": list(self._archive_paths),
            },
        )

        self._last_result = CompressionResult(
            original_count=len(older_msgs) + len(recent_msgs),
            compressed_count=1 + len(recent_msgs),
            summary_tokens=summary_usage.completion_tokens,
            archive_path=archive_path,
            llm_usage=summary_usage,
        )

        system_msgs = [m for g in system for m in g.messages]

        return system_msgs + [summary_msg] + recent_msgs

    def _build_summary_content(
        self, summary_text: str, archive_path: str | None
    ) -> str:
        """Assemble summary content with archive info and continuation footer."""
        parts = [
            f"## Conversation Summary (compressed, round {self._compression_count})",
            f"\n\n{summary_text}",
        ]

        if self._archive_paths:
            parts.append("\n\n---\n")
            if len(self._archive_paths) == 1:
                parts.append(
                    f"_Compressed conversation archived at "
                    f"`{self._archive_paths[0]}`_"
                )
            else:
                archive_list = "\n".join(
                    f"  - Round {i + 1}: `{p}`"
                    for i, p in enumerate(self._archive_paths)
                )
                parts.append(
                    f"_Compression archives ({len(self._archive_paths)} rounds):\n"
                    f"{archive_list}_"
                )

        parts.append(_CONTINUATION_FOOTER)
        return "".join(parts)

    def take_last_result(self) -> CompressionResult | None:
        """Return and clear the last compression result (one-shot read)."""
        result = self._last_result
        self._last_result = None
        return result

    # -- Grouping & Partitioning --

    @staticmethod
    def _group_atomic_pairs(messages: list[Message]) -> list[_MessageGroup]:
        """Group messages into atomic units.

        - System messages: standalone group (always protected)
        - Assistant with tool_calls + subsequent tool results: one group
        - All other messages (including compression summaries): standalone group
        """
        groups: list[_MessageGroup] = []
        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg.role == Role.SYSTEM:
                groups.append(_MessageGroup([msg]))
                i += 1
                continue

            if msg.role == Role.ASSISTANT and msg.tool_calls:
                group_msgs = [msg]
                expected_ids = {tc.id for tc in msg.tool_calls}
                j = i + 1
                while j < len(messages) and messages[j].role == Role.TOOL:
                    tool_msg = messages[j]
                    if (
                        tool_msg.tool_result
                        and tool_msg.tool_result.tool_call_id in expected_ids
                    ):
                        group_msgs.append(tool_msg)
                        expected_ids.discard(tool_msg.tool_result.tool_call_id)
                    else:
                        break
                    j += 1
                groups.append(_MessageGroup(group_msgs))
                i = j
                continue

            groups.append(_MessageGroup([msg]))
            i += 1

        return groups

    def _partition(
        self, groups: list[_MessageGroup]
    ) -> tuple[list[_MessageGroup], list[_MessageGroup], list[_MessageGroup]]:
        """Split groups into (system, older, recent).

        Protected system messages (agent identity) are always preserved.
        Compression summaries are system-role but NOT protected — they
        enter non_system and can be re-compressed (telescoping).

        `retain_count` is measured in *user-anchored segments*: each
        segment begins with a user message and includes the assistant /
        tool groups that follow before the next user. This guarantees
        ``recent[0]`` is user-starting 
        """
        system: list[_MessageGroup] = []
        non_system: list[_MessageGroup] = []

        for g in groups:
            if g.is_protected_system:
                system.append(g)
            else:
                non_system.append(g)

        user_starts = [
            i for i, g in enumerate(non_system)
            if g.messages and g.messages[0].role == Role.USER
        ]

        if len(user_starts) <= self._retain_count:
            return system, [], non_system

        split = user_starts[-self._retain_count]
        older = non_system[:split]
        recent = non_system[split:]

        return system, older, recent

    # -- LLM Summarization --

    async def _summarize(
        self,
        older: list[Message],
        recent: list[Message],
        extra_instructions: str | None,
    ) -> tuple[str, Usage]:
        older_text = self._format_messages(older)
        recent_text = (
            self._format_messages(recent) if recent else "(no recent messages)"
        )

        extra_block = ""
        if extra_instructions:
            extra_block = f"\n\n## Additional Requirements\n{extra_instructions}"

        user_content = _SUMMARY_USER_TEMPLATE.format(
            older_conversation=older_text,
            recent_conversation=recent_text,
            extra_instructions_block=extra_block,
        )

        kwargs: dict[str, Any] = {}
        if self._summary_max_tokens is not None:
            kwargs["max_tokens"] = self._summary_max_tokens

        try:
            response = await self._llm.generate_with_events(
                [
                    Message.system(_SUMMARY_SYSTEM_PROMPT),
                    Message.user(user_content),
                ],
                **kwargs,
            )
            return response.message.content or "", response.usage
        except Exception as e:
            logger.debug("Compression LLM call failed: %s — skipping", e)
            raise

    @staticmethod
    def _format_messages(messages: list[Message]) -> str:
        """Format messages for the summary prompt.

        Tool call + result pairs are formatted as coherent action units.
        User/tool attachments are rendered as short ``[Attached <mime>: <name>]``
        markers so the summarizer knows the media existed"""
        lines: list[str] = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.role.value.upper()

            if msg.role == Role.ASSISTANT and msg.tool_calls:
                if msg.content:
                    lines.append(f"[ASSISTANT]: {msg.content}")
                for tc in msg.tool_calls:
                    lines.append(f"[ASSISTANT → {tc.name}({tc.arguments})]")
                    # Find matching tool result
                    for j in range(i + 1, len(messages)):
                        tr = messages[j].tool_result
                        if (
                            messages[j].role == Role.TOOL
                            and tr is not None
                            and tr.tool_call_id == tc.id
                        ):
                            status = "ERROR" if tr.is_error else "OK"
                            lines.append(f"  └─ [{status}]: {tr.content}")
                            if tr.attachments:
                                for att in tr.attachments:
                                    lines.append(f"     └─ {describe_attachment_short(att)}")
                            break
                # Skip past processed tool result messages
                j = i + 1
                while j < len(messages) and messages[j].role == Role.TOOL:
                    j += 1
                i = j
                continue

            if msg.role == Role.TOOL:
                # Orphan tool result (shouldn't happen, graceful fallback)
                if msg.tool_result:
                    status = "ERROR" if msg.tool_result.is_error else "OK"
                    lines.append(f"[TOOL {status}]: {msg.tool_result.content}")
                    if msg.tool_result.attachments:
                        for att in msg.tool_result.attachments:
                            lines.append(f"  └─ {describe_attachment_short(att)}")
                i += 1
                continue

            if msg.content:
                lines.append(f"[{role}]: {msg.content}")
            elif msg.attachments:
                lines.append(f"[{role}]: (Media attachment only)")
            if msg.attachments:
                for att in msg.attachments:
                    lines.append(f"  └─ {describe_attachment_short(att)}")
            i += 1

        return "\n".join(lines)

    # -- Archive --

    @staticmethod
    def _normalize_scope(scope: str | None) -> str:
        raw = (scope or "main").strip()
        cleaned = "".join(
            ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
            for ch in raw
        )
        return cleaned or "main"

    def _archive_filename(self, round_num: int) -> str:
        if self._scope == "main":
            return f"compression_round_{round_num}.md"
        return f"compression_{self._scope}_round_{round_num}.md"

    def _archive(self, messages: list[Message], round_num: int) -> Path:
        """Write compressed messages to session-bound archive file."""
        archive_dir = (
            Path.home() / ".arktor" / "sessions"
            / str(self._session_id) / "compact"
        )
        archive_dir.mkdir(parents=True, exist_ok=True)
        path = archive_dir / self._archive_filename(round_num)

        lines = [
            f"# Compression Archive — Round {round_num}",
            f"Session: {self._session_id}",
            f"Timestamp: {datetime.now().isoformat()}",
            f"Messages: {len(messages)}",
            "",
            "---",
            "",
        ]
        for msg in messages:
            role = msg.role.value.upper()
            lines.append(f"## [{role}] {msg.created_at.isoformat()}")
            lines.append("")
            if msg.content:
                lines.append(msg.content)
                lines.append("")
            if msg.attachments:
                lines.append("**Attachments**:")
                for att in msg.attachments:
                    lines.append(f"- {describe_attachment_full(att)}")
                lines.append("")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    lines.append(f"**Tool Call**: `{tc.name}({tc.arguments})`")
                    lines.append("")
            if msg.tool_result:
                status = "ERROR" if msg.tool_result.is_error else "OK"
                lines.append(f"**Tool Result** ({status}):")
                lines.append(msg.tool_result.content)
                lines.append("")
                if msg.tool_result.attachments:
                    lines.append("**Attached media**:")
                    for att in msg.tool_result.attachments:
                        lines.append(f"- {describe_attachment_full(att)}")
                    lines.append("")
            lines.append("---")
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    # -- Pruning --

    def prune_tool_outputs(self, messages: list[Message]) -> int:
        """Replace oversized tool_result with archive stubs.

        Scans messages outside the last ``prune_tail_turns`` user turns; each
        unpruned TOOL message whose content estimates
        ≥ ``prune_per_output_threshold`` tokens has its
        ``tool_result.content`` swapped for a stub. Originals are archived to
        ``sessions/{sid}/compact/pruned/{tool_call_id}.md``.

        Mutates ``messages`` in place. Returns total tokens reclaimed.
        """
        if not self._session_id:
            return 0
        user_indices = [
            i for i, m in enumerate(messages) if m.role == Role.USER
        ]
        if len(user_indices) < self._prune_tail_turns:
            return 0
        boundary = user_indices[-self._prune_tail_turns]

        reclaimed = 0
        for msg in messages[:boundary]:
            tr = msg.tool_result
            if msg.role != Role.TOOL or tr is None:
                continue
            if msg.metadata.get("tool_pruned"):
                continue
            n_tokens = count_tokens(tr.content, self._model)
            if n_tokens < self._prune_per_output_threshold:
                continue
            archive_path = self._archive_pruned(tr)
            stub = (
                f"[Pruned tool output: ~{n_tokens} tokens reclaimed. "
                f"Original archived at `{archive_path}`]"
            )
            tr.content = stub
            # count_messages_tokens sums both msg.content AND tool_result.content;
            # zero out the duplicate so accounting actually drops.
            msg.content = None
            msg.metadata["tool_pruned"] = {
                "tokens": n_tokens,
                "archive": str(archive_path),
            }
            reclaimed += n_tokens
        return reclaimed

    def _archive_pruned(self, tr: ToolResult) -> Path:
        """Write original tool_result content under sessions/{sid}/compact/pruned/."""
        archive_dir = (
            Path.home() / ".arktor" / "sessions"
            / str(self._session_id) / "compact" / "pruned"
        )
        archive_dir.mkdir(parents=True, exist_ok=True)
        safe_id = (
            _UNSAFE_FILENAME_CHARS.sub("_", tr.tool_call_id)[:_FILENAME_ID_MAX]
            or "unknown"
        )
        path = archive_dir / f"{safe_id}.md"

        lines = [
            f"# Pruned tool result — `{tr.tool_call_id}`",
            f"Timestamp: {datetime.now().isoformat()}",
            f"Status: {'ERROR' if tr.is_error else 'OK'}",
            "",
            "---",
            "",
            tr.content,
        ]
        if tr.attachments:
            lines += ["", "**Attachments**:"]
            lines += [f"- {describe_attachment_full(a)}" for a in tr.attachments]

        path.write_text("\n".join(lines), encoding="utf-8")
        return path


def create_compressor(
    llm: BaseLLM,
    memory_config: MemoryConfig,
    model: str,
    session_id: str | None = None,
    scope: str = "main",
) -> ContextCompressor:
    """Create a ContextCompressor from MemoryConfig."""
    comp = memory_config.compression
    return ContextCompressor(
        llm=llm,
        threshold=comp.threshold,
        retain_count=comp.retain_count,
        summary_max_tokens=comp.summary_max_tokens,
        model=model,
        session_id=session_id,
        scope=scope,
    )
