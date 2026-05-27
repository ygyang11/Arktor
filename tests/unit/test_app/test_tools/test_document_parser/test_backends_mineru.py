"""Unit tests for document_parser.backends.mineru."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from agent_app.tools.document_parser.backends import mineru as mineru_mod
from agent_app.tools.document_parser.backends.mineru import (
    MinerULightweightBackend,
    MinerUOptions,
    MinerUV4Backend,
)


def _make_zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


class TestV4Body:
    def test_batch_body_uses_files_array(self) -> None:
        b = MinerUV4Backend("k", opts=MinerUOptions(model="mineru-vlm"))
        body = b._batch_body([{"url": "https://x/y.pdf", "is_ocr": False}])
        assert "files" in body
        assert "file_names" not in body
        assert body["model_version"] == "vlm"
        assert body["files"] == [{"url": "https://x/y.pdf", "is_ocr": False}]


class TestFetchAndUnpack:
    async def test_extracts_full_md_and_layout_and_images(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        layout_data = json.dumps({"pdf_info": [{"page_no": 0}, {"page_no": 1}]}).encode()
        zip_bytes = _make_zip({
            "full.md": b"markdown body",
            "layout.json": layout_data,
            "images/img_a.jpg": b"\xff\xd8\xff",
            "images/img_b.jpg": b"\xff\xd8\xff",
            "abc_origin.pdf": b"%PDF-1.4 should be skipped",
        })

        async def _fake_get_bytes(
            ctx: object, url: str, *, timeout_s: int,
        ) -> bytes:
            return zip_bytes

        monkeypatch.setattr(mineru_mod, "get_bytes", _fake_get_bytes)

        b = MinerUV4Backend("k", opts=MinerUOptions(model="mineru-vlm"))
        import aiohttp
        async with aiohttp.ClientSession() as s:
            outcome = await b._fetch_and_unpack("https://x/z.zip", tmp_path)
        assert (tmp_path / "content.md").read_bytes() == b"markdown body"
        assert json.loads((tmp_path / "layout.json").read_bytes())["pdf_info"][0]["page_no"] == 0
        assert outcome.page_count == 2
        assert outcome.image_count == 2
        assert outcome.backend_name == "mineru-vlm"
        assert outcome.backend_model == "vlm"
        assert not (tmp_path / "abc_origin.pdf").exists()

    async def test_zip_slip_filtered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        zip_bytes = _make_zip({
            "../escape.txt": b"bad",
            "full.md": b"ok",
        })

        async def _fake_get_bytes(
            ctx: object, url: str, *, timeout_s: int,
        ) -> bytes:
            return zip_bytes

        monkeypatch.setattr(mineru_mod, "get_bytes", _fake_get_bytes)

        b = MinerUV4Backend("k", opts=MinerUOptions(model="mineru-vlm"))
        import aiohttp
        async with aiohttp.ClientSession() as s:
            await b._fetch_and_unpack("https://x/z.zip", tmp_path)
        assert not (tmp_path.parent / "escape.txt").exists()


class TestLightweight:
    async def test_url_parse_writes_content_and_layout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Drive _poll_and_materialize directly by mocking get_envelope + get_text
        calls: dict[str, Any] = {"poll": 0}

        async def _fake_get_envelope(
            ctx: object, url: str, *,
            headers: dict[str, str] | None = None, timeout_s: int | None = None,
        ) -> dict[str, Any]:
            calls["poll"] += 1
            return {
                "code": 0,
                "data": {"state": "done", "markdown_url": "https://x/md"},
            }

        async def _fake_get_text(
            ctx: object, url: str, *, timeout_s: int,
        ) -> str:
            return "# Hello"

        async def _fake_sleep(_t: float) -> None:
            return None

        monkeypatch.setattr(mineru_mod, "get_envelope", _fake_get_envelope)
        monkeypatch.setattr(mineru_mod, "get_text", _fake_get_text)
        monkeypatch.setattr(mineru_mod.asyncio, "sleep", _fake_sleep)

        b = MinerULightweightBackend()
        outcome = await b._poll_and_materialize("task-1", tmp_path)
        assert (tmp_path / "content.md").read_text() == "# Hello"
        layout = json.loads((tmp_path / "layout.json").read_text())
        assert layout == {"pages": []}
        assert outcome.page_count is None
        assert outcome.image_count == 0
        assert outcome.backend_name == "mineru-lightweight"
