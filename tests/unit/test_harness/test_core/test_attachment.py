"""Tests for Attachment, ToolOutput, and session round-trip with attachments."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_harness.core.message import Attachment, Message, Role, ToolCall, ToolOutput, ToolResult
from agent_harness.session.base import SessionState


def _att(digest: str = "d" * 64) -> Attachment:
    return Attachment(digest=digest, mime="image/png", size=4, filename="x.png")


def test_attachment_frozen_blocks_assignment() -> None:
    att = _att()
    with pytest.raises(ValidationError):
        att.mime = "image/jpeg"  # type: ignore[misc]


def test_attachment_hashable_same_content_same_hash() -> None:
    a = _att()
    b = _att()
    assert hash(a) == hash(b)
    assert a == b


def test_tool_output_round_trip() -> None:
    att = _att()
    out = ToolOutput(content="Fetched", attachments=[att])
    dumped = out.model_dump()
    restored = ToolOutput.model_validate(dumped)
    assert restored.content == "Fetched"
    assert restored.attachments == [att]


def test_tool_output_str_only() -> None:
    out = ToolOutput(content="plain")
    assert out.attachments is None


def test_tool_result_carries_attachments() -> None:
    att = _att()
    tr = ToolResult(tool_call_id="t1", content="ok", attachments=[att])
    assert tr.attachments == [att]


def test_message_carries_attachments() -> None:
    att = _att()
    m = Message(role=Role.USER, content="see this", attachments=[att])
    assert m.attachments == [att]


def test_message_tool_factory_threads_attachments() -> None:
    att = _att()
    m = Message.tool(tool_call_id="t1", content="ok", attachments=[att])
    assert m.role == Role.TOOL
    assert m.tool_result is not None
    assert m.tool_result.attachments == [att]


def test_session_state_round_trip_with_attachments() -> None:
    att = _att()
    state = SessionState(
        session_id="s1",
        messages=[
            Message(role=Role.USER, content="hi", attachments=[att]),
            Message(role=Role.ASSISTANT, content="hello"),
        ],
    )
    blob = state.model_dump_json()
    restored = SessionState.model_validate_json(blob)
    assert restored.messages[0].attachments == [att]
    assert restored.messages[1].attachments is None


def test_session_state_old_json_without_attachments_compatible() -> None:
    old = (
        '{"session_id":"s1","messages":['
        '{"role":"user","content":"hi"},'
        '{"role":"assistant","content":"hello"}'
        ']}'
    )
    state = SessionState.model_validate_json(old)
    assert state.messages[0].attachments is None
    assert state.messages[1].attachments is None


def test_attachment_dict_independent_from_tool_call() -> None:
    tc = ToolCall(name="x", arguments={})
    assert not hasattr(tc, "attachments")
