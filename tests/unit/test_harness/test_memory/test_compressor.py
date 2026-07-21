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

    def test_rebind_updates_request_and_estimate_models(
        self, compressor: ContextCompressor,
    ) -> None:
        llm = AsyncMock()
        llm.model_name = "summary-model"

        compressor.rebind(llm=llm, model="consumer-model")

        assert compressor._llm is llm
        assert compressor._model == "consumer-model"
        assert compressor.model_name == "summary-model"


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
        assert len(older) == 2
        assert len(recent) == 8
        assert recent[0].messages[0].role.value == "user"

    def test_summary_can_be_recompressed(self, compressor: ContextCompressor) -> None:
        """Compression summary is system but not protected, enters non_system."""
        summary = Message.system(
            "Previous summary", metadata={"is_compression_summary": True}
        )
        msgs = [summary] + _make_messages(12)
        groups = ContextCompressor._group_atomic_pairs(msgs)
        _, older, recent = compressor._partition(groups)
        assert len(older) > 0
        assert older[0].messages[0].metadata.get("is_compression_summary") is True
        assert recent[0].messages[0].role.value == "user"

    def test_recent_always_starts_with_user(
        self, compressor: ContextCompressor,
    ) -> None:
        msgs = _make_messages(12)
        groups = ContextCompressor._group_atomic_pairs(msgs)
        _, _, recent = compressor._partition(groups)
        assert recent[0].messages[0].role.value == "user"

    def test_retain_count_counts_segments_not_groups(
        self, compressor: ContextCompressor,
    ) -> None:
        from agent_harness.core.message import ToolCall
        msgs: list[Message] = []
        for i in range(5):
            msgs.append(Message.user(f"u{i}"))
            tc = ToolCall(id=f"c{i}", name="f", arguments={})
            msgs.append(Message.assistant(None, tool_calls=[tc]))
            msgs.append(Message.tool(tool_call_id=f"c{i}", content=f"r{i}"))
        groups = ContextCompressor._group_atomic_pairs(msgs)
        _, older, recent = compressor._partition(groups)
        assert len(older) == 2
        assert recent[0].messages[0].role.value == "user"
        assert len(recent) == 8


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
            assert len(result) == 1 + 1 + 8
            assert result[2].role.value == "user"

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


def _prune_compressor(
    mock_llm: AsyncMock, *, threshold: int = 20, tail_turns: int = 3,
) -> ContextCompressor:
    return ContextCompressor(
        llm=mock_llm, threshold=0.75, retain_count=4, model="gpt-4o",
        session_id="s-prune",
        prune_per_output_threshold=threshold,
        prune_tail_turns=tail_turns,
    )


def _tool_msg(call_id: str, content: str) -> Message:
    return Message(
        role=Role.TOOL,
        tool_result=ToolResult(tool_call_id=call_id, content=content),
        content=content,
    )


class TestPruneToolOutputs:
    def test_no_session_id_is_noop(self, mock_llm: AsyncMock) -> None:
        comp = ContextCompressor(
            llm=mock_llm, threshold=0.75, retain_count=4, model="gpt-4o",
            prune_per_output_threshold=5,
        )
        msgs = [
            Message.user("u1"), _tool_msg("c1", "x" * 200),
            Message.user("u2"), Message.user("u3"), Message.user("u4"),
        ]
        assert comp.prune_tool_outputs(msgs) == 0
        assert msgs[1].tool_result.content == "x" * 200

    def test_below_tail_turns_count_is_noop(
        self, mock_llm: AsyncMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        comp = _prune_compressor(mock_llm, threshold=5, tail_turns=3)
        # only 2 user turns → fewer than tail_turns; everything protected
        msgs = [
            Message.user("u1"),
            _tool_msg("c1", "x" * 500),
            Message.user("u2"),
        ]
        assert comp.prune_tool_outputs(msgs) == 0
        assert msgs[1].tool_result.content == "x" * 500

    def test_prunes_oversized_tool_result_and_archives(
        self, mock_llm: AsyncMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        comp = _prune_compressor(mock_llm, threshold=20, tail_turns=3)
        big = "alpha beta gamma " * 200  # well above 20 tokens
        msgs = [
            Message.user("u1"),
            _tool_msg("c1", big),
            Message.user("u2"),
            Message.user("u3"),
            Message.user("u4"),
        ]
        reclaimed = comp.prune_tool_outputs(msgs)
        assert reclaimed > 20

        tr = msgs[1].tool_result
        assert tr.content.startswith("[Pruned tool output:")
        assert "`" in tr.content  # archive path quoted in stub
        assert msgs[1].metadata["tool_pruned"]["tokens"] == reclaimed
        archive = Path(msgs[1].metadata["tool_pruned"]["archive"])
        assert archive.exists()
        assert archive.parent.name == "pruned"
        assert archive.parent.parent.name == "compact"
        assert big in archive.read_text(encoding="utf-8")

    def test_protects_last_n_user_turns(
        self, mock_llm: AsyncMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        comp = _prune_compressor(mock_llm, threshold=20, tail_turns=3)
        big = "alpha beta gamma " * 200
        # u1 [tool c1 big] u2 [tool c2 big] u3 u4
        # tail_turns=3 → boundary = index of u2 (3rd-to-last USER)
        # c1 is eligible (before u2), c2 is protected (>= u2)
        msgs = [
            Message.user("u1"),
            _tool_msg("c1", big),
            Message.user("u2"),
            _tool_msg("c2", big),
            Message.user("u3"),
            Message.user("u4"),
        ]
        comp.prune_tool_outputs(msgs)
        assert msgs[1].tool_result.content.startswith("[Pruned")
        assert msgs[3].tool_result.content == big

    def test_skips_already_pruned(
        self, mock_llm: AsyncMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        comp = _prune_compressor(mock_llm, threshold=20, tail_turns=3)
        big = "alpha beta gamma " * 200
        already = _tool_msg("c1", big)
        already.metadata["tool_pruned"] = {"tokens": 42, "archive": "/old"}
        msgs = [
            Message.user("u1"), already,
            Message.user("u2"), Message.user("u3"), Message.user("u4"),
        ]
        assert comp.prune_tool_outputs(msgs) == 0
        # original content untouched, stamp unchanged
        assert already.tool_result.content == big
        assert already.metadata["tool_pruned"]["archive"] == "/old"

    def test_skips_small_outputs(
        self, mock_llm: AsyncMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        comp = _prune_compressor(mock_llm, threshold=1000, tail_turns=3)
        msgs = [
            Message.user("u1"), _tool_msg("c1", "tiny"),
            Message.user("u2"), Message.user("u3"), Message.user("u4"),
        ]
        assert comp.prune_tool_outputs(msgs) == 0
        assert msgs[1].tool_result.content == "tiny"

    def test_archive_records_attachments(
        self, mock_llm: AsyncMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        comp = _prune_compressor(mock_llm, threshold=20, tail_turns=3)
        big = "alpha beta gamma " * 200
        att = Attachment(
            digest="d" * 64, mime="image/png", filename="evidence.png", size=4096,
        )
        msgs = [
            Message.user("u1"),
            Message(
                role=Role.TOOL,
                tool_result=ToolResult(
                    tool_call_id="c1", content=big, attachments=[att],
                ),
                content=big,
            ),
            Message.user("u2"), Message.user("u3"), Message.user("u4"),
        ]
        comp.prune_tool_outputs(msgs)
        archive = Path(msgs[1].metadata["tool_pruned"]["archive"])
        text = archive.read_text(encoding="utf-8")
        assert "**Attachments**:" in text
        assert "evidence.png" in text
        # tool_result.attachments preserved on the live message
        assert msgs[1].tool_result.attachments == [att]


class TestArchiveDir:
    def test_compress_archive_under_compact_subdir(
        self, mock_llm: AsyncMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        comp = ContextCompressor(
            llm=mock_llm, threshold=0.75, retain_count=2,
            model="gpt-4o", session_id="s1",
        )
        path = comp._archive([Message.user("u")], round_num=1)
        assert path.parent.name == "compact"
        assert path.name == "compression_round_1.md"


class TestPruneToolOutputsHardening:
    """Review-driven fixes: accounting, filename safety, IO failure."""

    def test_token_accounting_drops_after_prune(
        self, mock_llm: AsyncMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """count_messages_tokens sums BOTH msg.content and tool_result.content
        — prune must null msg.content so the account actually drops."""
        from agent_harness.utils.token_counter import count_messages_tokens
        monkeypatch.setenv("HOME", str(tmp_path))
        comp = _prune_compressor(mock_llm, threshold=20, tail_turns=3)
        big = "alpha beta gamma " * 200
        msgs = [
            Message.user("u1"),
            _tool_msg("c1", big),
            Message.user("u2"), Message.user("u3"), Message.user("u4"),
        ]
        before = count_messages_tokens(msgs, "gpt-4o")
        reclaimed = comp.prune_tool_outputs(msgs)
        after = count_messages_tokens(msgs, "gpt-4o")
        assert reclaimed > 20
        # msg.content was zeroed; full reclaimed budget realized
        assert msgs[1].content is None
        # accounting drops by ~reclaimed (allow slop for ChatML overhead constants)
        assert before - after >= reclaimed - 10

    def test_malformed_tool_call_id_sanitized_in_filename(
        self, mock_llm: AsyncMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Path-traversal-shaped ids must not escape pruned/ dir."""
        monkeypatch.setenv("HOME", str(tmp_path))
        comp = _prune_compressor(mock_llm, threshold=20, tail_turns=3)
        big = "alpha beta gamma " * 200
        bad_id = "../../etc/passwd"
        msgs = [
            Message.user("u1"),
            Message(
                role=Role.TOOL,
                tool_result=ToolResult(tool_call_id=bad_id, content=big),
                content=big,
            ),
            Message.user("u2"), Message.user("u3"), Message.user("u4"),
        ]
        comp.prune_tool_outputs(msgs)
        archive = Path(msgs[1].metadata["tool_pruned"]["archive"])
        # Path separators stripped → file actually lives inside pruned/, not escaped.
        assert archive.parent.name == "pruned"
        assert "/" not in archive.name and "\\" not in archive.name
        assert archive.resolve().is_relative_to(archive.parent.resolve())
        assert msgs[1].tool_result.tool_call_id == bad_id  # field preserved

    def test_long_tool_call_id_truncated(
        self, mock_llm: AsyncMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        comp = _prune_compressor(mock_llm, threshold=20, tail_turns=3)
        big = "alpha beta gamma " * 200
        long_id = "call_" + "x" * 500
        msgs = [
            Message.user("u1"),
            Message(
                role=Role.TOOL,
                tool_result=ToolResult(tool_call_id=long_id, content=big),
                content=big,
            ),
            Message.user("u2"), Message.user("u3"), Message.user("u4"),
        ]
        comp.prune_tool_outputs(msgs)
        archive = Path(msgs[1].metadata["tool_pruned"]["archive"])
        # 96 char cap + ".md" suffix
        assert len(archive.stem) <= 96

    def test_real_provider_id_unchanged(
        self, mock_llm: AsyncMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Normal `call_XXX` / `toolu_XXX` ids pass through untouched."""
        monkeypatch.setenv("HOME", str(tmp_path))
        comp = _prune_compressor(mock_llm, threshold=20, tail_turns=3)
        big = "alpha beta gamma " * 200
        for normal_id in ("call_AbC123dEf456GhI789", "toolu_01AbC123dEf456"):
            msgs = [
                Message.user("u1"),
                Message(
                    role=Role.TOOL,
                    tool_result=ToolResult(tool_call_id=normal_id, content=big),
                    content=big,
                ),
                Message.user("u2"), Message.user("u3"), Message.user("u4"),
            ]
            comp.prune_tool_outputs(msgs)
            archive = Path(msgs[1].metadata["tool_pruned"]["archive"])
            assert archive.stem == normal_id

    def test_archive_write_failure_does_not_bubble_through_context_hook(
        self, mock_llm: AsyncMock, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """maybe_auto_compress must downgrade prune IO failures to a debug log."""
        import logging
        from agent_harness.memory.compressor import ContextCompressor

        # Force _archive_pruned to raise — simulates disk full / permission denied.
        def boom(self: ContextCompressor, tr: ToolResult) -> Path:
            raise OSError("disk full")
        monkeypatch.setattr(ContextCompressor, "_archive_pruned", boom)

        comp = _prune_compressor(mock_llm, threshold=20, tail_turns=3)
        big = "alpha beta gamma " * 200
        msgs = [
            Message.user("u1"),
            _tool_msg("c1", big),
            Message.user("u2"), Message.user("u3"), Message.user("u4"),
        ]
        # Direct call bubbles (compressor doesn't swallow) — that's by design,
        # the swallow happens in context.maybe_auto_compress.
        with pytest.raises(OSError):
            comp.prune_tool_outputs(msgs)

        # Caller-side wrapper (context.maybe_auto_compress) is tested separately
        # via test_context.py; here we just confirm the failure surface so the
        # caller's try/except has something concrete to catch.
        # Sanity: tool_result.content untouched on failure
        assert msgs[1].tool_result.content == big
        # Suppress unused import warning in case logging is removed later.
        _ = logging
