"""Unit tests for document_parser.backends.paddleocr.materialize_jsonl."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from agent_app.tools.document_parser.backends.paddleocr import materialize_jsonl
from agent_app.tools.document_parser.errors import (
    DocumentBackendError,
    DocumentErrorClass,
)


def _make_jsonl(pages: list[dict[str, object]]) -> str:
    lines = []
    for p in pages:
        lines.append(json.dumps({
            "logId": "x", "errorCode": 0, "errorMsg": "",
            "result": {"layoutParsingResults": [p]},
        }))
    return "\n".join(lines)


class TestMaterializeJsonl:
    def test_text_extraction(self, tmp_path: Path) -> None:
        jsonl = _make_jsonl([
            {"prunedResult": {"page_count": 1, "width": 100, "height": 200},
             "markdown": {"text": "page-1 body", "images": {}}},
            {"prunedResult": {"page_count": 1},
             "markdown": {"text": "page-2 body", "images": {}}},
        ])
        out = materialize_jsonl(jsonl, tmp_path, "paddleocr-vl-1.5", "PaddleOCR-VL-1.5")
        md = (tmp_path / "content.md").read_text()
        assert "page-1 body" in md
        assert "page-2 body" in md
        assert out.page_count == 2
        assert out.image_count == 0

    def test_image_dedup_and_rewrite(self, tmp_path: Path) -> None:
        b64 = base64.b64encode(b"\x89PNG").decode()
        jsonl = _make_jsonl([
            {"prunedResult": {"page_count": 1},
             "markdown": {
                 "text": "see imgs/img_in_image_box_0.jpg here",
                 "images": {"imgs/img_in_image_box_0.jpg": b64},
             }},
            {"prunedResult": {"page_count": 1},
             "markdown": {
                 "text": "ref imgs/img_in_image_box_0.jpg again",
                 "images": {"imgs/img_in_image_box_0.jpg": b64},
             }},
        ])
        out = materialize_jsonl(jsonl, tmp_path, "n", "m")
        md = (tmp_path / "content.md").read_text()
        assert "images/img_in_image_box_0.jpg" in md
        assert "imgs/img_in_image_box_0.jpg" not in md
        assert out.image_count == 1
        files = list((tmp_path / "images").iterdir())
        assert len(files) == 1

    def test_layout_pages(self, tmp_path: Path) -> None:
        jsonl = _make_jsonl([
            {"prunedResult": {"page_count": 1, "parsing_res_list": [
                {"block_label": "title", "block_content": "hello",
                 "block_bbox": [0, 0, 10, 10]}]},
             "markdown": {"text": "hello"}},
        ])
        materialize_jsonl(jsonl, tmp_path, "n", "m")
        layout = json.loads((tmp_path / "layout.json").read_text())
        assert isinstance(layout["pages"], list)
        assert layout["pages"][0]["page_count"] == 1
        assert layout["pages"][0]["parsing_res_list"][0]["block_label"] == "title"

    def test_empty_jsonl_writes_empty_artifacts(self, tmp_path: Path) -> None:
        out = materialize_jsonl("", tmp_path, "n", "m")
        assert (tmp_path / "content.md").read_text() == ""
        layout = json.loads((tmp_path / "layout.json").read_text())
        assert layout == {"pages": []}
        assert out.page_count is None
        assert out.image_count == 0

    def test_unparseable_lines_skipped(self, tmp_path: Path) -> None:
        valid = json.dumps({
            "result": {"layoutParsingResults": [
                {"prunedResult": {"page_count": 1},
                 "markdown": {"text": "ok"}}
            ]}
        })
        jsonl = f"garbage\n{valid}\n"
        out = materialize_jsonl(jsonl, tmp_path, "n", "m")
        assert "ok" in (tmp_path / "content.md").read_text()
        assert out.page_count == 1

    def test_base64_decode_failure_skipped(self, tmp_path: Path) -> None:
        jsonl = _make_jsonl([
            {"prunedResult": {"page_count": 1},
             "markdown": {
                 "text": "no rewrite",
                 "images": {"imgs/bad.jpg": "@@@not-base64@@@"},
             }},
        ])
        out = materialize_jsonl(jsonl, tmp_path, "n", "m")
        assert out.image_count == 0

    def test_disk_full_translates_to_io_error(
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
            materialize_jsonl("", tmp_path, "n", "m")
        assert ei.value.error_class is DocumentErrorClass.IO_ERROR
