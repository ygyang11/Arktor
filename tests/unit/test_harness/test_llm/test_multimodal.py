"""Tests for BaseLLM multimodal template methods + per-provider block overrides."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest

from agent_harness.core.message import Attachment, Message, Role, ToolCall, ToolResult
from agent_harness.llm.anthropic_provider import AnthropicProvider
from agent_harness.llm.base import BaseLLM
from agent_harness.llm.openai_provider import OpenAIProvider
from agent_harness.utils import blob as blob_module


@pytest.fixture(autouse=True)
def _isolate_blob_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(blob_module, "_BLOB_DIR", tmp_path / "blobs")


def _att(data: bytes, mime: str, filename: str | None = None) -> Attachment:
    return blob_module.make_attachment(data, mime, filename)


def _openai() -> OpenAIProvider:
    p = OpenAIProvider.__new__(OpenAIProvider)
    return p


def _anthropic() -> AnthropicProvider:
    p = AnthropicProvider.__new__(AnthropicProvider)
    return p


class TestExpandToolResultMedia:
    def test_passthrough_when_no_media(self) -> None:
        p = _openai()
        msgs = [
            Message(role=Role.USER, content="hi"),
            Message(role=Role.ASSISTANT, content="ok"),
            Message.tool(tool_call_id="t1", content="result"),
            Message(role=Role.ASSISTANT, content="done"),
        ]
        out = p._expand_tool_result_media(msgs)
        assert out == msgs

    def test_inserts_single_synthetic_user_for_run(self) -> None:
        p = _openai()
        a = _att(b"img1", "image/png", "a.png")
        b = _att(b"img2", "image/png", "b.png")
        msgs = [
            Message(role=Role.ASSISTANT, content="calling"),
            Message.tool(tool_call_id="t1", content="ok1", attachments=[a]),
            Message.tool(tool_call_id="t2", content="ok2", attachments=[b]),
            Message(role=Role.ASSISTANT, content="next"),
        ]
        out = p._expand_tool_result_media(msgs)
        assert len(out) == 5
        assert out[0] is msgs[0]
        assert out[1] is msgs[1]
        assert out[2] is msgs[2]
        synthetic = out[3]
        assert synthetic.role == Role.USER
        assert synthetic.content == p._TOOL_MEDIA_PREAMBLE
        assert synthetic.attachments == [a, b]
        assert out[4] is msgs[3]

    def test_no_synthetic_when_run_has_no_attachments(self) -> None:
        p = _openai()
        msgs = [
            Message.tool(tool_call_id="t1", content="ok"),
            Message.tool(tool_call_id="t2", content="ok"),
        ]
        out = p._expand_tool_result_media(msgs)
        assert len(out) == 2
        assert out == msgs

    def test_single_tool_with_media(self) -> None:
        p = _openai()
        a = _att(b"pdf", "application/pdf", "doc.pdf")
        msgs = [
            Message.tool(tool_call_id="t1", content="ok", attachments=[a]),
        ]
        out = p._expand_tool_result_media(msgs)
        assert len(out) == 2
        assert out[1].role == Role.USER
        assert out[1].attachments == [a]


class TestRenderUserContentOpenAI:
    def test_text_only_when_no_attachments_not_called(self) -> None:
        # Render only invoked when msg.attachments — directly call here.
        p = _openai()
        msg = Message(role=Role.USER, content="hi", attachments=[])
        parts = p._render_user_content(msg)
        assert parts == [{"type": "text", "text": "hi"}]

    def test_image_block(self) -> None:
        p = _openai()
        data = b"raw-png-bytes"
        a = _att(data, "image/png", "x.png")
        msg = Message(role=Role.USER, content="see", attachments=[a])
        parts = p._render_user_content(msg)
        assert parts[0] == {"type": "text", "text": "see"}
        img = parts[1]
        assert img["type"] == "image_url"
        b64 = base64.b64encode(data).decode()
        assert img["image_url"] == {"url": f"data:image/png;base64,{b64}"}

    def test_pdf_block(self) -> None:
        p = _openai()
        data = b"%PDF-bytes"
        a = _att(data, "application/pdf", "doc.pdf")
        msg = Message(role=Role.USER, attachments=[a])
        parts = p._render_user_content(msg)
        assert len(parts) == 1
        b64 = base64.b64encode(data).decode()
        assert parts[0] == {
            "type": "file",
            "file": {
                "filename": "doc.pdf",
                "file_data": f"data:application/pdf;base64,{b64}",
            },
        }

    def test_pdf_default_filename(self) -> None:
        p = _openai()
        a = _att(b"x", "application/pdf")
        msg = Message(role=Role.USER, attachments=[a])
        parts = p._render_user_content(msg)
        assert parts[0]["file"]["filename"] == "document.pdf"

    def test_non_media_mime_skipped(self) -> None:
        p = _openai()
        a = _att(b"svg", "image/svg+xml", "x.svg")
        msg = Message(role=Role.USER, content="hi", attachments=[a])
        parts = p._render_user_content(msg)
        assert parts == [{"type": "text", "text": "hi"}]


class TestRenderUserContentAnthropic:
    def test_image_block(self) -> None:
        p = _anthropic()
        data = b"jpg-bytes"
        a = _att(data, "image/jpeg")
        msg = Message(role=Role.USER, attachments=[a])
        parts = p._render_user_content(msg)
        b64 = base64.b64encode(data).decode()
        assert parts[0] == {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        }

    def test_pdf_block(self) -> None:
        p = _anthropic()
        data = b"%PDF"
        a = _att(data, "application/pdf", "r.pdf")
        msg = Message(role=Role.USER, content="here", attachments=[a])
        parts = p._render_user_content(msg)
        b64 = base64.b64encode(data).decode()
        assert parts[0] == {"type": "text", "text": "here"}
        assert parts[1] == {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": b64,
            },
        }


class TestBaseLLMHooks:
    def test_default_text_block(self) -> None:
        class _LLM(BaseLLM):
            def __init__(self) -> None:
                pass

            async def generate(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
                raise NotImplementedError

            async def stream(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
                raise NotImplementedError

        llm = _LLM()
        assert llm._text_block("hi") == {"type": "text", "text": "hi"}

    def test_default_image_pdf_blocks_raise(self) -> None:
        class _LLM(BaseLLM):
            def __init__(self) -> None:
                pass

            async def generate(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
                raise NotImplementedError

            async def stream(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
                raise NotImplementedError

        llm = _LLM()
        a = _att(b"x", "image/png")
        with pytest.raises(NotImplementedError):
            llm._image_block(a, "b64")
        with pytest.raises(NotImplementedError):
            llm._pdf_block(a, "b64")


# -- Media rejection → LLMUnsupportedContentError mapping --


import openai as _openai_sdk  # noqa: E402
import anthropic as _anthropic_sdk  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from agent_harness.core.errors import (  # noqa: E402
    LLMContextLengthError,
    LLMError,
    LLMUnsupportedContentError,
)


class _FakeOpenAIBadRequest(_openai_sdk.BadRequestError):
    def __init__(self, msg: str) -> None:
        Exception.__init__(self, msg)


class _FakeAnthropicBadRequest(_anthropic_sdk.BadRequestError):
    def __init__(self, msg: str) -> None:
        Exception.__init__(self, msg)


def _wire_openai(p: OpenAIProvider, raiser: Any) -> None:
    p._additive_semantics = False
    p._strip_reasoning_details = False

    async def fake_create(**req: Any) -> Any:
        return raiser()

    p._build_request = lambda *a, **k: {"messages": []}  # type: ignore[method-assign]
    p._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    p._parse_response = lambda r: r  # type: ignore[method-assign]


def _wire_anthropic(p: AnthropicProvider, raiser: Any) -> None:
    async def fake_create(**req: Any) -> Any:
        return raiser()

    p._build_request = lambda *a, **k: {"messages": []}  # type: ignore[method-assign]
    p._client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    p._parse_response = lambda r: r  # type: ignore[method-assign]


class TestOpenAIMediaRejection:
    async def test_media_phrase_raises_unsupported_content(self) -> None:
        p = _openai()
        msg = (
            "the message at position 5 with role 'user' contains an "
            "invalid part type: file"
        )

        def raiser() -> Any:
            raise _FakeOpenAIBadRequest(msg)

        _wire_openai(p, raiser)
        with pytest.raises(LLMUnsupportedContentError):
            await p.generate([])

    async def test_deepseek_unknown_variant_image_url(self) -> None:
        p = _openai()
        msg = "messages[3]: unknown variant `image_url`, expected `text`"

        def raiser() -> Any:
            raise _FakeOpenAIBadRequest(msg)

        _wire_openai(p, raiser)
        with pytest.raises(LLMUnsupportedContentError):
            await p.generate([])

    async def test_context_length_still_raises_context_error(self) -> None:
        p = _openai()

        def raiser() -> Any:
            raise _FakeOpenAIBadRequest("maximum context length exceeded")

        _wire_openai(p, raiser)
        with pytest.raises(LLMContextLengthError):
            await p.generate([])

    async def test_unrelated_400_still_llm_error(self) -> None:
        p = _openai()

        def raiser() -> Any:
            raise _FakeOpenAIBadRequest("some unrelated 400")

        _wire_openai(p, raiser)
        with pytest.raises(LLMError) as ei:
            await p.generate([])
        assert not isinstance(ei.value, LLMUnsupportedContentError)


class TestAnthropicMediaRejection:
    async def test_media_phrase_raises_unsupported_content(self) -> None:
        p = _anthropic()

        def raiser() -> Any:
            raise _FakeAnthropicBadRequest("Could not process image")

        _wire_anthropic(p, raiser)
        with pytest.raises(LLMUnsupportedContentError):
            await p.generate([])

    async def test_context_length_still_raises_context_error(self) -> None:
        p = _anthropic()

        def raiser() -> Any:
            raise _FakeAnthropicBadRequest("prompt too long for context")

        _wire_anthropic(p, raiser)
        with pytest.raises(LLMContextLengthError):
            await p.generate([])

    async def test_unrelated_400_still_llm_error(self) -> None:
        p = _anthropic()

        def raiser() -> Any:
            raise _FakeAnthropicBadRequest("invalid argument: temperature out of range")

        _wire_anthropic(p, raiser)
        with pytest.raises(LLMError) as ei:
            await p.generate([])
        assert not isinstance(ei.value, LLMUnsupportedContentError)
