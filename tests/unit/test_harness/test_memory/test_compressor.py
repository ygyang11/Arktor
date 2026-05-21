"""Tests for agent_harness.memory.compressor — ContextCompressor."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agent_harness.core.message import Attachment, Message, Role, ToolCall, ToolResult
from agent_harness.memory.compressor import ContextCompressor


@pytest.fixture
def mock_llm() -> AsyncMock:
    llm = AsyncMock()
    llm.generate_with_events.return_value = AsyncMock(
        message=Message.assistant(
            "### User Goal\nBuild a web app.\n### Completed Work\nSet up project."
        )
    )
    return llm


@pytest.fixture
def compressor(mock_llm: AsyncMock) -> ContextCompressor:
    return ContextCompressor(
        llm=mock_llm, threshold=0.75, retain_count=4, model="gpt-4o",
    )


def _make_messages(n: int) -> list[Message]:
    msgs: list[Message] = []
    for i in range(n):
        if i % 2 == 0:
            msgs.append(Message.user(f"User message {i}"))
        else:
            msgs.append(Message.assistant(f"Assistant response {i}"))
    return msgs


def _make_tool_pair() -> list[Message]:
    tc = ToolCall(id="call_abc", name="web_search", arguments={"query": "test"})
    return [
        Message.assistant(None, tool_calls=[tc]),
        Message.tool(tool_call_id="call_abc", content="Search results here"),
    ]


class TestShouldCompress:
    def test_below_threshold(self, compressor: ContextCompressor) -> None:
        assert compressor.should_compress([Message.user("short")], 10000) is False

    def test_above_threshold(self, compressor: ContextCompressor) -> None:
        assert compressor.should_compress(_make_messages(100), 500) is True

    def test_empty(self, compressor: ContextCompressor) -> None:
        assert compressor.should_compress([], 1000) is False

    def test_zero_max_tokens(self, compressor: ContextCompressor) -> None:
        assert compressor.should_compress([Message.user("hi")], 0) is False


class TestGroupAtomicPairs:
    def test_simple_messages(self) -> None:
        groups = ContextCompressor._group_atomic_pairs(
            [Message.user("hi"), Message.assistant("hello")]
        )
        assert len(groups) == 2

    def test_tool_call_pair_grouped(self) -> None:
        groups = ContextCompressor._group_atomic_pairs(_make_tool_pair())
        assert len(groups) == 1
        assert len(groups[0].messages) == 2

    def test_system_standalone(self) -> None:
        groups = ContextCompressor._group_atomic_pairs(
            [Message.system("prompt"), Message.user("hi")]
        )
        assert len(groups) == 2
        assert groups[0].is_system
        assert groups[0].is_protected_system

    def test_multi_tool_calls(self) -> None:
        tc1 = ToolCall(id="c1", name="t1", arguments={})
        tc2 = ToolCall(id="c2", name="t2", arguments={})
        groups = ContextCompressor._group_atomic_pairs([
            Message.assistant(None, tool_calls=[tc1, tc2]),
            Message.tool(tool_call_id="c1", content="r1"),
            Message.tool(tool_call_id="c2", content="r2"),
        ])
        assert len(groups) == 1
        assert len(groups[0].messages) == 3

    def test_compression_summary_is_system_but_not_protected(self) -> None:
        summary = Message.system(
            "Summary", metadata={"is_compression_summary": True}
        )
        groups = ContextCompressor._group_atomic_pairs([summary])
        assert groups[0].is_system
        assert not groups[0].is_protected_system


class TestPartition:
    def test_few_messages_no_compression(self, compressor: ContextCompressor) -> None:
        groups = ContextCompressor._group_atomic_pairs(_make_messages(4))
        _, older, recent = compressor._partition(groups)
        assert len(older) == 0
        assert len(recent) == 4

    def test_system_always_separated(self, compressor: ContextCompressor) -> None:
        groups = ContextCompressor._group_atomic_pairs(
            [Message.system("prompt")] + _make_messages(8)
        )
        system, _, _ = compressor._partition(groups)
        assert len(system) == 1
        assert system[0].is_protected_system

    def test_correct_split(self, compressor: ContextCompressor) -> None:
        groups = ContextCompressor._group_atomic_pairs(_make_messages(10))
        _, older, recent = compressor._partition(groups)
        assert len(older) == 6
        assert len(recent) == 4

    def test_summary_can_be_recompressed(self, compressor: ContextCompressor) -> None:
        """Compression summary is system but not protected, enters non_system."""
        summary = Message.system(
            "Previous summary", metadata={"is_compression_summary": True}
        )
        msgs = [summary] + _make_messages(8)
        groups = ContextCompressor._group_atomic_pairs(msgs)
        _, older, _ = compressor._partition(groups)
        assert len(older) > 0


class TestCompress:
    @pytest.mark.asyncio
    async def test_basic_compression(self, compressor: ContextCompressor) -> None:
        compressor._session_id = "test"
        with patch.object(
            ContextCompressor, "_archive", return_value=Path("/tmp/t.md")
        ):
            msgs = [Message.system("prompt")] + _make_messages(10)
            result = await compressor.compress(msgs)

            assert result[0].role.value == "system"
            assert result[0].content == "prompt"
            assert result[1].role.value == "system"
            assert result[1].metadata.get("is_compression_summary") is True
            assert result[1].metadata.get("archive_paths") == ["/tmp/t.md"]
            assert "resume directly" in (result[1].content or "")
            assert len(result) == 1 + 1 + 4

    @pytest.mark.asyncio
    async def test_no_compression_when_few(self, compressor: ContextCompressor) -> None:
        msgs = _make_messages(3)
        result = await compressor.compress(msgs)
        assert result == msgs

    @pytest.mark.asyncio
    async def test_no_archive_without_session_id(
        self, compressor: ContextCompressor
    ) -> None:
        assert compressor._session_id is None
        msgs = _make_messages(10)
        result = await compressor.compress(msgs)
        summary = result[0]
        assert summary.metadata.get("archive_paths") == []

    @pytest.mark.asyncio
    async def test_archive_paths_accumulate(
        self, compressor: ContextCompressor
    ) -> None:
        compressor._session_id = "test"
        with patch.object(
            ContextCompressor, "_archive",
            side_effect=[Path("/tmp/r1.md"), Path("/tmp/r2.md"), Path("/tmp/r3.md")],
        ):
            await compressor.compress(_make_messages(10))
            await compressor.compress(_make_messages(10))
            assert len(compressor._archive_paths) == 2

    @pytest.mark.asyncio
    async def test_extra_instructions_in_prompt(
        self, compressor: ContextCompressor, mock_llm: AsyncMock,
    ) -> None:
        await compressor.compress(
            _make_messages(10),
            extra_instructions="Preserve all research URLs",
        )
        call_args = mock_llm.generate_with_events.call_args
        user_msg = call_args[0][0][1]
        assert "Preserve all research URLs" in user_msg.content

    @pytest.mark.asyncio
    async def test_recent_context_in_prompt(
        self, compressor: ContextCompressor, mock_llm: AsyncMock,
    ) -> None:
        await compressor.compress(_make_messages(10))
        call_args = mock_llm.generate_with_events.call_args
        user_msg = call_args[0][0][1]
        assert "<recent_context>" in user_msg.content

    @pytest.mark.asyncio
    async def test_side_effects_only_after_llm_success(
        self, mock_llm: AsyncMock
    ) -> None:
        """If LLM call fails, no side effects (count, archive, paths)."""
        mock_llm.generate_with_events.side_effect = RuntimeError("LLM down")
        comp = ContextCompressor(
            llm=mock_llm, threshold=0.75, retain_count=4, model="gpt-4o",
            session_id="test",
        )
        with pytest.raises(RuntimeError):
            await comp.compress(_make_messages(10))
        assert comp._compression_count == 0
        assert comp._archive_paths == []

    def test_clone_copies_static_config_but_resets_runtime_state(
        self, compressor: ContextCompressor
    ) -> None:
        compressor._session_id = "test"
        compressor._compression_count = 2
        compressor._archive_paths = ["/tmp/r1.md", "/tmp/r2.md"]

        clone = compressor.clone(scope="executor")

        assert clone is not compressor
        assert clone.scope == "executor"
        assert clone._session_id == "test"
        assert clone._compression_count == 0
        assert clone._archive_paths == []

    def test_restore_runtime_state_recovers_round_and_archive_paths(
        self, compressor: ContextCompressor
    ) -> None:
        messages = [
            Message.system(
                "Summary 1",
                metadata={
                    "is_compression_summary": True,
                    "compression_round": 1,
                    "archive_paths": ["/tmp/r1.md"],
                },
            ),
            Message.system(
                "Summary 2",
                metadata={
                    "is_compression_summary": True,
                    "compression_round": 2,
                    "archive_paths": ["/tmp/r1.md", "/tmp/r2.md"],
                },
            ),
        ]

        compressor.restore_runtime_state(messages)

        assert compressor._compression_count == 2
        assert compressor._archive_paths == ["/tmp/r1.md", "/tmp/r2.md"]

    @pytest.mark.asyncio
    async def test_summary_metadata_uses_archive_paths_only(
        self, compressor: ContextCompressor
    ) -> None:
        compressor._session_id = "test"
        with patch.object(
            ContextCompressor, "_archive", return_value=Path("/tmp/r1.md")
        ):
            result = await compressor.compress(_make_messages(10))

        summary = result[0]
        assert "archive_path" not in summary.metadata
        assert summary.metadata.get("archive_paths") == ["/tmp/r1.md"]


class TestTakeLastResult:
    @pytest.mark.asyncio
    async def test_returns_once(self, compressor: ContextCompressor) -> None:
        await compressor.compress(_make_messages(10))
        result = compressor.take_last_result()
        assert result is not None
        assert result.original_count > 0
        assert compressor.take_last_result() is None


class TestFormatMessages:
    def test_tool_pair_formatted_as_unit(self) -> None:
        text = ContextCompressor._format_messages(_make_tool_pair())
        assert "web_search" in text
        assert "└─" in text

    def test_plain_messages(self) -> None:
        text = ContextCompressor._format_messages(
            [Message.user("hello"), Message.assistant("hi")]
        )
        assert "[USER]: hello" in text
        assert "[ASSISTANT]: hi" in text

    def test_empty_content_skipped(self) -> None:
        text = ContextCompressor._format_messages(
            [Message.assistant(None)]
        )
        assert text == ""

    def test_user_attachments_rendered_as_short_markers(self) -> None:
        att = Attachment(digest="a" * 64, mime="image/png", filename="shot.png", size=512)
        text = ContextCompressor._format_messages(
            [Message.user("look at this", attachments=[att])]
        )
        assert "[USER]: look at this" in text
        assert "  └─ [Attached image/png: shot.png]" in text

    def test_media_only_user_emits_role_marker(self) -> None:
        att = Attachment(digest="a" * 64, mime="image/png", filename="shot.png", size=512)
        text = ContextCompressor._format_messages([
            Message.assistant("previous turn"),
            Message(role=Role.USER, content=None, attachments=[att]),
        ])
        lines = text.splitlines()
        # Role marker must survive so the summarizer doesn't fuse the
        # attachment into the prior assistant turn.
        assert "[USER]: (Media attachment only)" in lines
        assert "  └─ [Attached image/png: shot.png]" in lines

    def test_tool_result_attachments_rendered_in_pair(self) -> None:
        tc = ToolCall(id="call_x", name="web_fetch", arguments={"url": "u"})
        tr = ToolResult(
            tool_call_id="call_x",
            content="fetched ok",
            attachments=[
                Attachment(digest="b" * 64, mime="application/pdf", filename="doc.pdf", size=2048),
            ],
        )
        text = ContextCompressor._format_messages([
            Message.assistant(None, tool_calls=[tc]),
            Message(role=Role.TOOL, tool_result=tr, content="fetched ok"),
        ])
        assert "└─ [OK]: fetched ok" in text
        assert "     └─ [Attached application/pdf: doc.pdf]" in text

    def test_orphan_tool_attachments_rendered(self) -> None:
        tr = ToolResult(
            tool_call_id="orphan",
            content="stray",
            attachments=[
                Attachment(digest="c" * 64, mime="image/jpeg", filename=None, size=1024),
            ],
        )
        text = ContextCompressor._format_messages(
            [Message(role=Role.TOOL, tool_result=tr, content="stray")]
        )
        assert "[TOOL OK]: stray" in text
        # default filename derived from mime: image
        assert "  └─ [Attached image/jpeg: image]" in text


class TestArchive:
    def test_archive_includes_user_and_tool_attachments(
        self, mock_llm: AsyncMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        comp = ContextCompressor(
            llm=mock_llm, threshold=0.75, retain_count=2,
            model="gpt-4o", session_id="s1",
        )
        user_att = Attachment(
            digest="a" * 64, mime="image/png", filename="hello.png", size=300_000,
        )
        tool_att = Attachment(
            digest="b" * 64, mime="application/pdf", filename="paper.pdf", size=1_500_000,
        )
        tc = ToolCall(id="call_q", name="web_fetch", arguments={"url": "u"})
        tr = ToolResult(
            tool_call_id="call_q", content="paper text", attachments=[tool_att],
        )
        msgs = [
            Message.user("look", attachments=[user_att]),
            Message.assistant(None, tool_calls=[tc]),
            Message(role=Role.TOOL, tool_result=tr, content="paper text"),
        ]
        path = comp._archive(msgs, round_num=1)
        text = path.read_text(encoding="utf-8")
        assert "**Attachments**:" in text
        assert "hello.png (image/png, 293.0KB, sha256:" + ("a" * 12) + "…)" in text
        assert "**Attached media**:" in text
        assert "paper.pdf (application/pdf, 1.4MB, sha256:" + ("b" * 12) + "…)" in text


class TestSummaryPrompt:
    def test_prompt_mentions_attachments_and_next_steps(self) -> None:
        from agent_harness.memory.compressor import _SUMMARY_SYSTEM_PROMPT
        assert "### Next Steps" in _SUMMARY_SYSTEM_PROMPT
        assert "Media attachments" in _SUMMARY_SYSTEM_PROMPT
        # "what is pending" was moved out of Current State into Next Steps
        assert (
            "Where things stand right now: what is done and what is in progress."
            in _SUMMARY_SYSTEM_PROMPT
        )
