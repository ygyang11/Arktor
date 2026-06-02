"""Unit tests for document_parser.storage."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from agent_app.tools.document_parser import storage
from agent_app.tools.document_parser.storage import (
    _MIME_TO_SUFFIX,
    Kind,
    TargetInspection,
    already_parsed,
    detect_artifacts,
    format_cached,
    format_no_viable,
    format_success,
    hash_source,
    inspect_target,
    is_local_path,
    make_slug,
    session_documents_root,
    write_manifest,
)


class TestMimeSuffixMap:
    def test_pdf_and_images(self) -> None:
        assert _MIME_TO_SUFFIX["application/pdf"] == ".pdf"
        assert _MIME_TO_SUFFIX["image/png"] == ".png"
        assert _MIME_TO_SUFFIX["image/jpeg"] == ".jpg"
        assert _MIME_TO_SUFFIX["image/gif"] == ".gif"
        assert _MIME_TO_SUFFIX["image/webp"] == ".webp"


class TestIsLocalPath:
    def test_local(self) -> None:
        assert is_local_path("/abs/path")
        assert is_local_path("./rel")
        assert is_local_path("file.pdf")

    def test_url(self) -> None:
        assert not is_local_path("http://x.com/a.pdf")
        assert not is_local_path("https://x.com/a.pdf")


class TestInspectLocal:
    async def test_pdf(self, tmp_path: Path) -> None:
        p = tmp_path / "doc.pdf"
        p.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        insp = await inspect_target(str(p))
        assert insp.is_local is True
        assert insp.kind == "pdf"
        assert insp.mime == "application/pdf"
        assert insp.name == "doc.pdf"
        assert insp.size_bytes is not None and insp.size_bytes > 0

    async def test_image(self, tmp_path: Path) -> None:
        p = tmp_path / "x.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")
        insp = await inspect_target(str(p))
        assert insp.is_local is True
        assert insp.kind == "image"
        assert insp.pages == 1
        assert insp.mime == "image/png"

    async def test_unknown(self, tmp_path: Path) -> None:
        p = tmp_path / "x.bin"
        p.write_bytes(b"\x00\x01")
        insp = await inspect_target(str(p))
        assert insp.kind == "unknown"
        assert insp.pages is None

    async def test_missing(self) -> None:
        with pytest.raises(FileNotFoundError):
            await inspect_target("/non/existent/path.pdf")


class TestInspectRemote:
    async def test_head_pdf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _head(
            url: str, *, headers: dict[str, str] | None = None, timeout: int = 30,
        ) -> tuple[int, Mapping[str, str]]:
            return 200, {
                "Content-Type": "application/pdf",
                "Content-Length": "12345",
            }
        monkeypatch.setattr(
            "agent_harness.utils.http_retry.http_head_with_retry", _head,
        )
        insp = await inspect_target("https://arxiv.org/pdf/2401.14200")
        assert insp.is_local is False
        assert insp.kind == "pdf"
        assert insp.mime == "application/pdf"
        assert insp.size_bytes == 12345
        assert insp.name == "2401.14200"

    async def test_head_image(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _head(
            url: str, *, headers: dict[str, str] | None = None, timeout: int = 30,
        ) -> tuple[int, Mapping[str, str]]:
            return 200, {"Content-Type": "image/png"}
        monkeypatch.setattr(
            "agent_harness.utils.http_retry.http_head_with_retry", _head,
        )
        insp = await inspect_target("https://example.com/cover.png")
        assert insp.kind == "image"
        assert insp.pages == 1
        assert insp.mime == "image/png"

    async def test_head_fails_falls_back_to_suffix(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def _head(
            url: str, *, headers: dict[str, str] | None = None, timeout: int = 30,
        ) -> tuple[int, Mapping[str, str]]:
            raise RuntimeError("boom")
        monkeypatch.setattr(
            "agent_harness.utils.http_retry.http_head_with_retry", _head,
        )
        insp = await inspect_target("https://example.com/doc.pdf")
        assert insp.kind == "pdf"
        assert insp.mime is None
        assert insp.size_bytes is None

    async def test_head_sends_user_agent(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Wikimedia and other anti-bot CDNs 403 aiohttp's default UA on HEAD.
        Inspect must self-identify so kind/mime/size detection works for
        those sources."""
        captured: dict[str, Any] = {}

        async def _head(
            url: str, *, headers: dict[str, str] | None = None, timeout: int = 30,
        ) -> tuple[int, Mapping[str, str]]:
            captured["headers"] = headers
            return 200, {"Content-Type": "image/jpeg"}

        monkeypatch.setattr(
            "agent_harness.utils.http_retry.http_head_with_retry", _head,
        )
        await inspect_target("https://upload.wikimedia.org/x.jpg")
        hdrs = captured.get("headers") or {}
        ua = hdrs.get("User-Agent") or hdrs.get("user-agent")
        assert ua and "arktor" in ua


class TestSlug:
    def test_basic(self) -> None:
        s = make_slug(source="https://x.com/a.pdf", content_hash="deadbeefcafebabe")
        assert s.startswith("a_")
        assert s.endswith("_deadbeef")

    def test_with_hint(self) -> None:
        s = make_slug(
            source="https://x.com/a.pdf", content_hash="deadbeef",
            suggested="arxiv-2401.14200",
        )
        assert s.startswith("arxiv-2401.14200_")

    def test_unsafe_chars_sanitized(self) -> None:
        s = make_slug(
            source="https://x.com/a.pdf", content_hash="deadbeef",
            suggested="a b/c?d",
        )
        assert "/" not in s
        assert "?" not in s
        assert " " not in s


class TestHashSource:
    def test_local_streamed(self, tmp_path: Path) -> None:
        p = tmp_path / "x.txt"
        p.write_bytes(b"hello world")
        h = hash_source(str(p), is_local=True)
        assert len(h) == 64

    def test_remote_url(self) -> None:
        h1 = hash_source("https://x.com/a.pdf", is_local=False)
        h2 = hash_source("https://x.com/a.pdf", is_local=False)
        assert h1 == h2


class TestAlreadyParsed:
    def test_partial_is_false(self, tmp_path: Path) -> None:
        (tmp_path / "content.md").write_text("x")
        assert not already_parsed(tmp_path)

    def test_full_is_true(self, tmp_path: Path) -> None:
        (tmp_path / "content.md").write_text("x")
        (tmp_path / "manifest.json").write_text("{}")
        assert already_parsed(tmp_path)

    def test_corrupt_manifest_is_false(self, tmp_path: Path) -> None:
        (tmp_path / "content.md").write_text("x")
        (tmp_path / "manifest.json").write_text("{not valid json")
        assert not already_parsed(tmp_path)

    def test_non_dict_manifest_is_false(self, tmp_path: Path) -> None:
        (tmp_path / "content.md").write_text("x")
        (tmp_path / "manifest.json").write_text('"a string, not an object"')
        assert not already_parsed(tmp_path)


class TestWriteManifest:
    def test_full_round_trip(self, tmp_path: Path) -> None:
        (tmp_path / "content.md").write_text("body")
        (tmp_path / "layout.json").write_text('{"pages": []}')
        path = write_manifest(
            tmp_path,
            slug="a_deadbeef",
            source={"target": "https://x.com/a.pdf", "name": "a.pdf", "origin": "remote_url"},
            backend={"name": "mineru-vlm", "model": "vlm"},
            size_bytes=12345,
            mime="application/pdf",
            kind="pdf",
            page_count=10,
            image_count=3,
            content_md_tokens=2000,
            content_md_lines=50,
            successful_tier_elapsed_ms=4321,
            fallback_chain=[],
            skipped_tiers=[],
        )
        data = json.loads(path.read_text())
        assert data["mime"] == "application/pdf"
        assert data["kind"] == "pdf"
        assert data["size_bytes"] == 12345
        assert data["stats"]["page_count"] == 10
        assert data["stats"]["successful_tier_elapsed_ms"] == 4321
        assert "content_md" in data["artifacts"]
        assert "layout_json" in data["artifacts"]


class TestDetectArtifacts:
    def test_empty(self, tmp_path: Path) -> None:
        assert detect_artifacts(tmp_path) == {}

    def test_with_images_dir_only_when_non_empty(self, tmp_path: Path) -> None:
        (tmp_path / "content.md").write_text("x")
        (tmp_path / "images").mkdir()
        a = detect_artifacts(tmp_path)
        assert "images_dir" not in a
        (tmp_path / "images" / "img_1.png").write_bytes(b"x")
        a = detect_artifacts(tmp_path)
        assert a.get("images_dir") == "images/"


class TestFormatSuccess:
    def test_pdf_with_pages_and_size(self, tmp_path: Path) -> None:
        out = format_success(
            slug_dir=tmp_path,
            source="https://x/a.pdf",
            name="a.pdf",
            kind="pdf",
            page_count=80,
            size_mb=23.1,
            backend_name="mineru-vlm",
            backend_model="vlm",
            content_md_tokens=120_000,
            content_md_lines=502,
            image_count=320,
        )
        assert "Document parsed and saved." in out
        assert "format : pdf (80 pages, 23.1 MB)" in out
        assert "backend: mineru-vlm (vlm)" in out
        assert "(~120.0k tokens, 502 lines)" in out
        assert "320 figures" in out
        assert "manifest.json" in out

    def test_image_no_pages(self, tmp_path: Path) -> None:
        out = format_success(
            slug_dir=tmp_path, source="x", name="cover.png", kind="image",
            page_count=1, size_mb=0.5,
            backend_name="paddleocr-vl-1.5", backend_model="PaddleOCR-VL-1.5",
            content_md_tokens=100, content_md_lines=5, image_count=0,
        )
        assert "format : image" in out
        assert "figures" not in out

    def test_unknown_uses_suffix(self, tmp_path: Path) -> None:
        out = format_success(
            slug_dir=tmp_path, source="x", name="oddfile.docx", kind="unknown",
            page_count=None, size_mb=None,
            backend_name="b", backend_model="m",
            content_md_tokens=0, content_md_lines=0, image_count=0,
        )
        assert "format : docx" in out


class TestFormatNoViable:
    def test_chain_groups_by_tier(self) -> None:
        chain = [
            {"tier": "paddleocr-vl-1.5", "mode": "url",
             "error_class": "INVALID_INPUT", "error_message": "code 10004"},
            {"tier": "paddleocr-vl-1.5", "mode": "local",
             "error_class": "BACKEND_READ_FAILED", "error_message": "encrypted pdf"},
            {"tier": "mineru-vlm", "mode": "local",
             "error_class": "BACKEND_READ_FAILED", "error_message": "encrypted pdf"},
        ]
        out = format_no_viable(skipped=[], chain=chain)
        assert "Tried:" in out
        assert "Error: document parsing failed." in out
        assert "1." in out
        assert "2." in out

    def test_skipped_preflight(self) -> None:
        out = format_no_viable(
            skipped=[{"tier": "mineru-lightweight", "reason": "size>10MB(url)"}],
            chain=[],
        )
        assert "Skipped (preflight)" in out
        assert "mineru-lightweight" in out
        assert "Tried:" not in out

    def test_unattempted_section(self) -> None:
        chain = [{"tier": "paddleocr-vl-1.5", "mode": "url",
                  "error_class": "INVALID_INPUT", "error_message": "x"}]
        out = format_no_viable(
            skipped=[], chain=chain, unattempted=["paddleocr-vl", "mineru-vlm"],
        )
        assert "Skipped (aborted)" in out
        assert "paddleocr-vl" in out


class TestFormatCached:
    def test_reads_from_manifest_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # populate manifest only — do not let inspect_target run
        manifest: dict[str, Any] = {
            "slug": "a_deadbeef",
            "source": {"target": "https://x/a.pdf", "name": "a.pdf", "origin": "remote_url"},
            "size_bytes": 23 * 1024 * 1024,
            "mime": "application/pdf",
            "kind": "pdf",
            "backend": {"name": "mineru-vlm", "model": "vlm"},
            "parsed_at": "2026-05-27T00:00:00",
            "artifacts": {"content_md": "content.md", "layout_json": "layout.json"},
            "stats": {
                "page_count": 80, "image_count": 320,
                "content_md_tokens": 120_000, "content_md_lines": 502,
            },
            "fallback_chain": [],
            "skipped_tiers": [],
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

        def _boom(_url: str) -> TargetInspection:
            raise AssertionError("format_cached must not call inspect_target")

        monkeypatch.setattr(storage, "inspect_target", _boom)

        out = format_cached(tmp_path, "https://x/a.pdf")
        assert "format : pdf (80 pages, 23.0 MB)" in out
        assert "backend: mineru-vlm (vlm)" in out
        assert "320 figures" in out


class TestSessionDocsRoot:
    def test_anonymous_when_no_sid(self) -> None:
        p = session_documents_root(None)
        assert p.parts[-2:] == ("anonymous", "documents")

    def test_sid_path(self) -> None:
        p = session_documents_root("S1")
        assert "S1" in p.parts
