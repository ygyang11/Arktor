"""Unit tests for document_parser.backends.paddleocr.materialize_jsonl."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

from agent_app.tools.document_parser.backends.paddleocr import materialize_jsonl
from agent_app.tools.document_parser.errors import (
    DocumentBackendError,
    DocumentErrorClass,
)

paddleocr_mod = sys.modules["agent_app.tools.document_parser.backends.paddleocr"]


def _make_jsonl(pages: list[dict[str, object]]) -> str:
    lines = []
    for p in pages:
        lines.append(json.dumps({
            "logId": "x", "errorCode": 0, "errorMsg": "",
            "result": {"layoutParsingResults": [p]},
        }))
    return "\n".join(lines)


class TestMaterializeJsonl:
    async def test_text_extraction(self, tmp_path: Path) -> None:
        jsonl = _make_jsonl([
            {"prunedResult": {"page_count": 1, "width": 100, "height": 200},
             "markdown": {"text": "page-1 body", "images": {}}},
            {"prunedResult": {"page_count": 1},
             "markdown": {"text": "page-2 body", "images": {}}},
        ])
        out = await materialize_jsonl(jsonl, tmp_path, "paddleocr-vl-1.5", "PaddleOCR-VL-1.5")
        md = (tmp_path / "content.md").read_text()
        assert "page-1 body" in md
        assert "page-2 body" in md
        assert out.page_count == 2
        assert out.image_count == 0

    async def test_image_url_fetched_and_rewritten(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        url = "https://paddleocr-result.bj.bcebos.com/abc.jpg"
        jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00fake-jpeg"

        async def _fake_get_bytes(ctx: object, u: str, *, timeout_s: int) -> bytes:
            assert u == url
            return jpeg_bytes

        monkeypatch.setattr(paddleocr_mod, "get_bytes", _fake_get_bytes)

        jsonl = _make_jsonl([
            {"prunedResult": {"page_count": 1},
             "markdown": {
                 "text": "see imgs/img_in_image_box_0.jpg here",
                 "images": {"imgs/img_in_image_box_0.jpg": url},
             }},
            {"prunedResult": {"page_count": 1},
             "markdown": {
                 "text": "ref imgs/img_in_image_box_0.jpg again",
                 "images": {"imgs/img_in_image_box_0.jpg": url},
             }},
        ])
        out = await materialize_jsonl(jsonl, tmp_path, "n", "m")

        md = (tmp_path / "content.md").read_text()
        assert "images/img_in_image_box_0.jpg" in md
        assert "imgs/img_in_image_box_0.jpg" not in md
        assert out.image_count == 1
        files = list((tmp_path / "images").iterdir())
        assert len(files) == 1
        # Verify the fetched bytes were written, NOT a base64-decoded garbage
        assert files[0].read_bytes() == jpeg_bytes

    async def test_base64_fallback_still_works(self, tmp_path: Path) -> None:
        jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00"
        b64 = base64.b64encode(jpeg_bytes).decode()
        jsonl = _make_jsonl([
            {"prunedResult": {"page_count": 1},
             "markdown": {
                 "text": "see imgs/x.jpg",
                 "images": {"imgs/x.jpg": b64},
             }},
        ])
        out = await materialize_jsonl(jsonl, tmp_path, "n", "m")
        assert out.image_count == 1
        files = list((tmp_path / "images").iterdir())
        assert files[0].read_bytes() == jpeg_bytes

    async def test_failed_url_fetch_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def _fake_get_bytes(ctx: object, u: str, *, timeout_s: int) -> bytes:
            raise DocumentBackendError(
                DocumentErrorClass.BACKEND_TRANSIENT_ERROR, 502, "bad gateway",
            )

        monkeypatch.setattr(paddleocr_mod, "get_bytes", _fake_get_bytes)

        jsonl = _make_jsonl([
            {"prunedResult": {"page_count": 1},
             "markdown": {
                 "text": "x",
                 "images": {"imgs/x.jpg": "https://bcebos.com/x.jpg"},
             }},
        ])
        out = await materialize_jsonl(jsonl, tmp_path, "n", "m")
        # image silently dropped; rest of document still materializes
        assert out.image_count == 0
        assert (tmp_path / "content.md").exists()
        assert not (tmp_path / "images").exists()

    async def test_failed_image_keeps_original_ref_in_content_md(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A URL that fails to download must NOT have its reference rewritten
        to `images/...` in content.md — that would dangle into a missing
        local file. The original `imgs/...` ref stays put so the broken
        link is at least visually obvious."""
        url_ok = "https://bcebos.com/ok.jpg"
        url_fail = "https://bcebos.com/fail.jpg"

        async def _fake_get_bytes(ctx: object, u: str, *, timeout_s: int) -> bytes:
            if u == url_ok:
                return b"\xff\xd8\xff\xe0jfif"
            raise DocumentBackendError(
                DocumentErrorClass.BACKEND_TRANSIENT_ERROR, 502, "bad gateway",
            )

        monkeypatch.setattr(paddleocr_mod, "get_bytes", _fake_get_bytes)

        jsonl = _make_jsonl([
            {"prunedResult": {"page_count": 1},
             "markdown": {
                 "text": "see imgs/ok.jpg and imgs/fail.jpg",
                 "images": {
                     "imgs/ok.jpg": url_ok,
                     "imgs/fail.jpg": url_fail,
                 },
             }},
        ])
        out = await materialize_jsonl(jsonl, tmp_path, "n", "m")
        assert out.image_count == 1
        md = (tmp_path / "content.md").read_text()
        assert "images/ok.jpg" in md
        # failed one keeps its original ref — NOT rewritten to a dangling
        # `images/fail.jpg` that does not exist on disk
        assert "imgs/fail.jpg" in md
        assert "images/fail.jpg" not in md
        assert (tmp_path / "images" / "ok.jpg").exists()
        assert not (tmp_path / "images" / "fail.jpg").exists()

    async def test_layout_pages(self, tmp_path: Path) -> None:
        jsonl = _make_jsonl([
            {"prunedResult": {"page_count": 1, "parsing_res_list": [
                {"block_label": "title", "block_content": "hello",
                 "block_bbox": [0, 0, 10, 10]}]},
             "markdown": {"text": "hello"}},
        ])
        await materialize_jsonl(jsonl, tmp_path, "n", "m")
        layout = json.loads((tmp_path / "layout.json").read_text())
        assert isinstance(layout["pages"], list)
        assert layout["pages"][0]["page_count"] == 1
        assert layout["pages"][0]["parsing_res_list"][0]["block_label"] == "title"

    async def test_empty_jsonl_writes_empty_artifacts(self, tmp_path: Path) -> None:
        out = await materialize_jsonl("", tmp_path, "n", "m")
        assert (tmp_path / "content.md").read_text() == ""
        layout = json.loads((tmp_path / "layout.json").read_text())
        assert layout == {"pages": []}
        assert out.page_count is None
        assert out.image_count == 0

    async def test_unparseable_lines_skipped(self, tmp_path: Path) -> None:
        valid = json.dumps({
            "result": {"layoutParsingResults": [
                {"prunedResult": {"page_count": 1},
                 "markdown": {"text": "ok"}}
            ]}
        })
        jsonl = f"garbage\n{valid}\n"
        out = await materialize_jsonl(jsonl, tmp_path, "n", "m")
        assert "ok" in (tmp_path / "content.md").read_text()
        assert out.page_count == 1

    async def test_base64_decode_failure_skipped(self, tmp_path: Path) -> None:
        jsonl = _make_jsonl([
            {"prunedResult": {"page_count": 1},
             "markdown": {
                 "text": "no rewrite",
                 "images": {"imgs/bad.jpg": "@@@not-base64@@@"},
             }},
        ])
        out = await materialize_jsonl(jsonl, tmp_path, "n", "m")
        assert out.image_count == 0

    async def test_disk_full_translates_to_io_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original = Path.write_text

        def _raise(
            self: Path, data: str, encoding: str | None = None, errors: str | None = None,
            newline: str | None = None,
        ) -> int:
            if self.name == "content.md":
                raise OSError(28, "No space left on device")
            return original(self, data, encoding=encoding, errors=errors, newline=newline)

        monkeypatch.setattr(Path, "write_text", _raise)
        with pytest.raises(DocumentBackendError) as ei:
            await materialize_jsonl("", tmp_path, "n", "m")
        assert ei.value.error_class is DocumentErrorClass.IO_ERROR
