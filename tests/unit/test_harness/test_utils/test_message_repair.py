"""Unit tests for message repair utilities.

CRITICAL: patch_dangling_tool_calls is the last line of defense before messages
reach the LLM API. These tests verify not only that dangling/orphan issues are
fixed, but that the patch NEVER corrupts, reorders, or drops normal messages.
"""
from __future__ import annotations

from agent_harness.core.message import Attachment, Message, Role, ToolCall
from agent_harness.utils.message_repair import (
    _DANGLING_CONTENT,
    _STRIPPED_NOTE,
    patch_dangling_tool_calls,
    strip_last_tool_run_attachments,
)


def _att(filename: str = "x.png", mime: str = "image/png", size: int = 100) -> Attachment:
    return Attachment(digest="a" * 64, mime=mime, size=size, filename=filename)


def _tool_with_attachments(tc: ToolCall, content: str, atts: list[Attachment]) -> Message:
    return Message.tool(tool_call_id=tc.id, content=content, attachments=atts)


def _assistant_with_calls(*names: str) -> tuple[Message, list[ToolCall]]:
    """Create an assistant message with tool calls, return (msg, calls)."""
    calls = [ToolCall(name=n, arguments={}) for n in names]
    msg = Message.assistant(content="thinking", tool_calls=calls)
    return msg, calls


def _tool_result(tc: ToolCall, content: str = "ok") -> Message:
    return Message.tool(tool_call_id=tc.id, content=content)


# -- Fast-path / no-op --


class TestNoop:
    def test_empty(self) -> None:
        result = patch_dangling_tool_calls([])
        assert result == []

    def test_no_tool_calls(self) -> None:
        msgs = [Message.user("hi"), Message.assistant("hello")]
        result = patch_dangling_tool_calls(msgs)
        assert result is msgs

    def test_all_resolved(self) -> None:
        msg, calls = _assistant_with_calls("bash")
        msgs = [msg, _tool_result(calls[0])]
        result = patch_dangling_tool_calls(msgs)
        assert result is msgs


# -- Dangling tool_calls --


class TestDangling:
    def test_single_dangling(self) -> None:
        msg, calls = _assistant_with_calls("bash")
        msgs = [Message.user("hi"), msg]
        result = patch_dangling_tool_calls(msgs)
        assert len(result) == 3
        assert result[2].role == Role.TOOL
        assert result[2].tool_result.tool_call_id == calls[0].id
        assert result[2].tool_result.is_error is True
        assert result[2].tool_result.content == _DANGLING_CONTENT

    def test_multiple_dangling_same_assistant(self) -> None:
        msg, calls = _assistant_with_calls("a", "b", "c")
        msgs = [msg]
        result = patch_dangling_tool_calls(msgs)
        assert len(result) == 4
        for i, tc in enumerate(calls):
            assert result[i + 1].tool_result.tool_call_id == tc.id
            assert result[i + 1].tool_result.is_error is True

    def test_partial_tail_missing(self) -> None:
        msg, calls = _assistant_with_calls("a", "b", "c")
        msgs = [msg, _tool_result(calls[0])]
        result = patch_dangling_tool_calls(msgs)
        assert len(result) == 4
        assert result[1].tool_result.tool_call_id == calls[0].id
        assert result[1].tool_result.is_error is False
        assert result[2].tool_result.tool_call_id == calls[1].id
        assert result[2].tool_result.is_error is True
        assert result[3].tool_result.tool_call_id == calls[2].id
        assert result[3].tool_result.is_error is True

    def test_synthetic_after_existing_results(self) -> None:
        msg, calls = _assistant_with_calls("a", "b", "c")
        msgs = [msg, _tool_result(calls[0]), _tool_result(calls[1])]
        result = patch_dangling_tool_calls(msgs)
        assert result[1].tool_result.is_error is False
        assert result[2].tool_result.is_error is False
        assert result[3].tool_result.is_error is True
        assert result[3].tool_result.tool_call_id == calls[2].id

    def test_synthetic_position_with_trailing_messages(self) -> None:
        msg, calls = _assistant_with_calls("bash")
        msgs = [Message.user("hi"), msg, Message.user("next")]
        result = patch_dangling_tool_calls(msgs)
        assert result[0].role == Role.USER
        assert result[1].role == Role.ASSISTANT
        assert result[2].role == Role.TOOL
        assert result[3].role == Role.USER


# -- Orphaned tool_results --


class TestOrphaned:
    def test_remove_orphaned(self) -> None:
        orphan = Message.tool(tool_call_id="nonexistent_id", content="stale")
        msgs = [Message.user("hi"), orphan]
        result = patch_dangling_tool_calls(msgs)
        assert len(result) == 1
        assert result[0].role == Role.USER

    def test_orphan_adjacent_to_assistant(self) -> None:
        msg, calls = _assistant_with_calls("bash")
        orphan = Message.tool(tool_call_id="ghost", content="stale")
        msgs = [msg, orphan]
        result = patch_dangling_tool_calls(msgs)
        assert len(result) == 2
        assert result[1].tool_result.tool_call_id == calls[0].id
        assert result[1].tool_result.is_error is True


# -- Immutability --


class TestImmutability:
    def test_does_not_mutate_input(self) -> None:
        msg, _calls = _assistant_with_calls("bash")
        msgs = [msg]
        original_len = len(msgs)
        patch_dangling_tool_calls(msgs)
        assert len(msgs) == original_len


# -- Message integrity (CRITICAL) --


class TestMessageIntegrity:
    def test_full_conversation_integrity(self) -> None:
        sys_msg = Message.system("You are helpful")
        user1 = Message.user("hello")
        ast1 = Message.assistant("hi there")
        ast2, calls2 = _assistant_with_calls("search", "read")
        tr2a = _tool_result(calls2[0], "found it")
        tr2b = _tool_result(calls2[1], "file contents")
        user2 = Message.user("now do X")
        ast3, calls3 = _assistant_with_calls("bash", "write")

        msgs = [sys_msg, user1, ast1, ast2, tr2a, tr2b, user2, ast3]
        result = patch_dangling_tool_calls(msgs)

        assert result[0] is sys_msg
        assert result[1] is user1
        assert result[2] is ast1
        assert result[3] is ast2
        assert result[4] is tr2a
        assert result[5] is tr2b
        assert result[6] is user2
        assert result[7] is ast3
        assert result[8].tool_result.tool_call_id == calls3[0].id
        assert result[9].tool_result.tool_call_id == calls3[1].id
        assert len(result) == 10

    def test_multiple_rounds_only_patch_broken(self) -> None:
        ast1, calls1 = _assistant_with_calls("a", "b")
        tr1a = _tool_result(calls1[0])
        tr1b = _tool_result(calls1[1])
        ast2, calls2 = _assistant_with_calls("c")

        msgs = [ast1, tr1a, tr1b, ast2]
        result = patch_dangling_tool_calls(msgs)

        assert result[0] is ast1
        assert result[1] is tr1a
        assert result[2] is tr1b
        assert result[3] is ast2
        assert result[4].tool_result.tool_call_id == calls2[0].id
        assert len(result) == 5

    def test_non_tool_messages_never_lost(self) -> None:
        sys_msg = Message.system("sys")
        user1 = Message.user("u1")
        ast_plain = Message.assistant("plain response")
        user2 = Message.user("u2")
        ast_broken, _calls = _assistant_with_calls("bash")
        user3 = Message.user("u3")

        msgs = [sys_msg, user1, ast_plain, user2, ast_broken, user3]
        result = patch_dangling_tool_calls(msgs)

        non_tool = [m for m in result if m.role != Role.TOOL]
        assert non_tool == [sys_msg, user1, ast_plain, user2, ast_broken, user3]

    def test_existing_results_order_preserved(self) -> None:
        ast, calls = _assistant_with_calls("a", "b", "c", "d")
        tr_a = _tool_result(calls[0], "result_a")
        tr_b = _tool_result(calls[1], "result_b")

        msgs = [ast, tr_a, tr_b]
        result = patch_dangling_tool_calls(msgs)

        assert result[1] is tr_a
        assert result[2] is tr_b
        assert result[3].tool_result.tool_call_id == calls[2].id
        assert result[4].tool_result.tool_call_id == calls[3].id

    def test_normal_conversation_passthrough(self) -> None:
        ast, calls = _assistant_with_calls("bash")
        msgs = [
            Message.system("sys"),
            Message.user("hi"),
            ast,
            _tool_result(calls[0]),
            Message.assistant("done"),
        ]
        result = patch_dangling_tool_calls(msgs)
        assert result is msgs


# -- strip_last_tool_run_attachments --


class TestStripLastToolRunAttachments:
    def test_empty_list(self) -> None:
        msgs: list[Message] = []
        assert strip_last_tool_run_attachments(msgs) == 0

    def test_user_with_attachments_blocks_strip(self) -> None:
        """USER message with attachments is user-side; return 0 to defer to
        REPL rollback even if an earlier TOOL also carries attachments."""
        ast, calls = _assistant_with_calls("web_fetch")
        msgs = [
            Message.user("hi"),
            ast,
            _tool_with_attachments(calls[0], "ok", [_att("old.png")]),
            Message.assistant("done"),
            Message.user("look @x.pdf", attachments=[_att("x.pdf", "application/pdf")]),
        ]
        assert strip_last_tool_run_attachments(msgs) == 0
        # Tool attachments must remain untouched
        assert msgs[2].tool_result.attachments is not None
        # User attachments must remain untouched (REPL rollback handles them)
        assert msgs[4].attachments is not None

    def test_skips_trailing_non_attachment_messages_to_find_tool(self) -> None:
        """Trailing ASSISTANT / SYSTEM / USER-without-attachments must not
        block recovery — scan back to the most recent attachment-bearing
        TOOL run."""
        ast, calls = _assistant_with_calls("web_fetch")
        msgs = [
            ast,
            _tool_with_attachments(calls[0], "ok", [_att()]),
            Message.assistant("explained"),
        ]
        assert strip_last_tool_run_attachments(msgs) == 1
        assert msgs[1].tool_result.attachments is None
        assert msgs[1].tool_result.content.endswith(_STRIPPED_NOTE)

    def test_skips_trailing_background_system_to_find_tool(self) -> None:
        """Background-task SYSTEM notifications appended by
        ``_collect_background_results`` must not break tool-side recovery."""
        ast, calls = _assistant_with_calls("web_fetch")
        msgs = [
            Message.user("hi"),
            ast,
            _tool_with_attachments(calls[0], "ok", [_att()]),
            Message.system(
                "[Background Task Notification] done",
                metadata={"is_background_result": True},
            ),
        ]
        assert strip_last_tool_run_attachments(msgs) == 1
        assert msgs[2].tool_result.attachments is None

    def test_single_tool_run_with_attachments(self) -> None:
        ast, calls = _assistant_with_calls("web_fetch")
        tool = _tool_with_attachments(
            calls[0], "Downloaded PDF (108KB)", [_att("a.pdf", "application/pdf", 108)],
        )
        msgs = [Message.user("fetch"), ast, tool]
        n = strip_last_tool_run_attachments(msgs)
        assert n == 1
        assert msgs[2].tool_result.attachments is None
        assert msgs[2].tool_result.content.endswith(_STRIPPED_NOTE)
        # Original content preserved
        assert "Downloaded PDF (108KB)" in msgs[2].tool_result.content

    def test_parallel_tool_run_aggregates(self) -> None:
        ast, calls = _assistant_with_calls("a", "b", "c")
        msgs = [
            Message.user("hi"),
            ast,
            _tool_with_attachments(calls[0], "r0", [_att("x.png"), _att("y.png")]),
            _tool_with_attachments(calls[1], "r1", [_att("z.pdf", "application/pdf")]),
            Message.tool(tool_call_id=calls[2].id, content="r2"),
        ]
        n = strip_last_tool_run_attachments(msgs)
        assert n == 3
        assert msgs[2].tool_result.attachments is None
        assert msgs[3].tool_result.attachments is None
        # tool without attachments stays untouched
        assert msgs[4].tool_result.content == "r2"
        assert msgs[2].tool_result.content.endswith(_STRIPPED_NOTE)
        assert msgs[3].tool_result.content.endswith(_STRIPPED_NOTE)

    def test_only_strips_latest_run(self) -> None:
        ast1, calls1 = _assistant_with_calls("a")
        ast2, calls2 = _assistant_with_calls("b")
        old_tool = _tool_with_attachments(calls1[0], "old", [_att("old.png")])
        new_tool = _tool_with_attachments(calls2[0], "new", [_att("new.png")])
        msgs = [Message.user("hi"), ast1, old_tool, ast2, new_tool]
        n = strip_last_tool_run_attachments(msgs)
        assert n == 1
        # Old run untouched
        assert msgs[2].tool_result.attachments is not None
        assert msgs[2].tool_result.content == "old"
        # New run stripped
        assert msgs[4].tool_result.attachments is None
        assert msgs[4].tool_result.content.endswith(_STRIPPED_NOTE)

    def test_empty_content_gets_note_without_leading_blank(self) -> None:
        ast, calls = _assistant_with_calls("a")
        tool = _tool_with_attachments(calls[0], "", [_att()])
        msgs = [ast, tool]
        n = strip_last_tool_run_attachments(msgs)
        assert n == 1
        assert msgs[1].tool_result.content == _STRIPPED_NOTE

    def test_idempotent_second_strip_returns_zero(self) -> None:
        ast, calls = _assistant_with_calls("a")
        tool = _tool_with_attachments(calls[0], "ok", [_att()])
        msgs = [ast, tool]
        assert strip_last_tool_run_attachments(msgs) == 1
        assert strip_last_tool_run_attachments(msgs) == 0
