"""Unit tests for document_parser tool + parse_document service."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import sys

from agent_app.tools.document_parser.document_parser import (
    DocumentParserTool,
    _build_pipeline,
    _make_downloader,
    parse_document,
)

dp_mod = sys.modules["agent_app.tools.document_parser.document_parser"]
from agent_app.tools.document_parser.errors import (
    DocumentBackendError,
    DocumentErrorClass,
)
from agent_app.tools.document_parser.storage import TargetInspection
from agent_harness.core.config import DocumentParserConfig
from agent_harness.core.errors import HttpResponseTooLargeError


class TestBuildPipeline:
    def test_paddleocr_only(self) -> None:
        cfg = DocumentParserConfig(provider="paddleocr", paddleocr_api_key="x")
        tiers = _build_pipeline(cfg)
        names = [t.name for t in tiers]
        assert names == ["paddleocr-vl-1.5", "paddleocr-vl"]

    def test_mineru_only(self) -> None:
        cfg = DocumentParserConfig(provider="mineru", mineru_api_key="x")
        tiers = _build_pipeline(cfg)
        names = [t.name for t in tiers]
        assert names == ["mineru-vlm", "mineru-lightweight"]

    def test_auto_returns_four_tiers(self) -> None:
        cfg = DocumentParserConfig(provider="auto")
        tiers = _build_pipeline(cfg)
        assert len(tiers) == 4
        names = [t.name for t in tiers]
        assert names[0] == "paddleocr-vl-1.5"
        assert names[1] == "paddleocr-vl"
        assert names[2] == "mineru-vlm"
        assert names[3] == "mineru-lightweight"


class TestDocumentParserTool:
    def test_schema(self) -> None:
        t = DocumentParserTool()
        sch = t.get_schema()
        assert sch.name == "document_parser"
        assert "target" in sch.parameters["properties"]
        assert sch.parameters["required"] == ["target"]

    def test_session_aware_structural(self) -> None:
        from agent_harness.tool.base import SessionAware
        t = DocumentParserTool()
        assert isinstance(t, SessionAware)
        t.bind_session("S1")
        assert t._session_id == "S1"

    async def test_execute_empty_returns_error(self) -> None:
        t = DocumentParserTool()
        out = await t.execute(target="")
        assert out.startswith("Error:")

    async def test_execute_local_missing(self) -> None:
        t = DocumentParserTool()
        out = await t.execute(target="/tmp/__no_such_file__.pdf")
        assert "not found" in out


class TestParseDocumentCacheHit:
    async def test_returns_cached_without_parsing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Force session_documents_root to point at tmp_path
        def _root(_sid: str | None) -> Path:
            return tmp_path

        monkeypatch.setattr(dp_mod, "session_documents_root", _root)

        # Pre-seed cached artifacts at expected slug location
        async def _fake_inspect(_target: str) -> TargetInspection:
            return TargetInspection(
                is_local=False, size_bytes=1024, size_mb=0.001,
                pages=1, name="a.pdf", mime="application/pdf", kind="pdf",
            )

        monkeypatch.setattr(dp_mod, "inspect_target", _fake_inspect)

        slug = "arxiv-2401_3661d44d"
        cache_dir = tmp_path / slug
        cache_dir.mkdir(parents=True)
        (cache_dir / "content.md").write_text("body")
        (cache_dir / "manifest.json").write_text(json.dumps({
            "slug": slug,
            "source": {"target": "https://x/a.pdf", "name": "a.pdf", "origin": "remote_url"},
            "size_bytes": 1024,
            "mime": "application/pdf",
            "kind": "pdf",
            "backend": {"name": "paddleocr-vl-1.5", "model": "PaddleOCR-VL-1.5"},
            "stats": {
                "page_count": 1, "image_count": 0,
                "content_md_tokens": 4, "content_md_lines": 1,
            },
            "artifacts": {"content_md": "content.md"},
            "fallback_chain": [], "skipped_tiers": [],
        }))

        def _fake_make_slug(*, source: str, content_hash: str, suggested: str | None = None) -> str:
            return slug

        monkeypatch.setattr(dp_mod, "make_slug", _fake_make_slug)

        # Ensure run_pipeline is never reached
        async def _boom(*a: object, **kw: object) -> object:
            raise AssertionError("run_pipeline must not be called on cache hit")

        monkeypatch.setattr(dp_mod, "run_pipeline", _boom)

        out = await parse_document(
            target="https://x/a.pdf",
            session_id=None,
            slug_hint=None,
        )
        assert "Document parsed and saved." in out
        assert "paddleocr-vl-1.5" in out


class TestMakeDownloader:
    async def test_too_large(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _fail(*a: object, **kw: object) -> tuple[int, bytes]:
            raise HttpResponseTooLargeError(limit=1)

        monkeypatch.setattr(dp_mod, "http_get_bytes_with_retry", _fail)
        dl = _make_downloader("https://x/a.pdf", mime="application/pdf")
        with pytest.raises(DocumentBackendError) as ei:
            await dl()
        assert ei.value.error_class is DocumentErrorClass.FILE_TOO_LARGE

    async def test_http_4xx_download_failed(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def _fake(*a: object, **kw: object) -> tuple[int, bytes]:
            return 404, b""

        monkeypatch.setattr(dp_mod, "http_get_bytes_with_retry", _fake)
        dl = _make_downloader("https://x/a.pdf", mime="application/pdf")
        with pytest.raises(DocumentBackendError) as ei:
            await dl()
        assert ei.value.error_class is DocumentErrorClass.DOWNLOAD_FAILED

    async def test_suffix_from_mime_for_extensionless_url(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        async def _fake(*a: object, **kw: object) -> tuple[int, bytes]:
            return 200, b"%PDF-1.4"

        monkeypatch.setattr(dp_mod, "http_get_bytes_with_retry", _fake)
        dl = _make_downloader(
            "https://arxiv.org/pdf/2401.14200", mime="application/pdf",
        )
        p = await dl()
        assert p.suffix == ".pdf"
        p.unlink(missing_ok=True)

    async def test_io_error_translation(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def _fake(*a: object, **kw: object) -> tuple[int, bytes]:
            return 200, b"data"

        monkeypatch.setattr(dp_mod, "http_get_bytes_with_retry", _fake)

        original_fdopen = dp_mod.os.fdopen

        class _BrokenFile:
            def __enter__(self) -> _BrokenFile:
                return self

            def __exit__(self, *a: object) -> None:
                pass

            def write(self, _b: bytes) -> int:
                raise OSError(28, "No space left on device")

        def _broken_fdopen(_fd: int, _mode: str) -> _BrokenFile:
            return _BrokenFile()

        monkeypatch.setattr(dp_mod.os, "fdopen", _broken_fdopen)

        dl = _make_downloader("https://x/a.pdf", mime="application/pdf")
        with pytest.raises(DocumentBackendError) as ei:
            await dl()
        assert ei.value.error_class is DocumentErrorClass.IO_ERROR

        monkeypatch.setattr(dp_mod.os, "fdopen", original_fdopen)


class TestFinalizeManifestTolerant:
    async def test_manifest_write_failure_does_not_break_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from agent_app.tools.document_parser.backends import DocumentBackendOutcome
        from agent_app.tools.document_parser.document_parser import _finalize
        from agent_app.tools.document_parser.pipeline import PipelineSuccess

        (tmp_path / "content.md").write_text("x")

        def _raise_manifest(*a: object, **kw: object) -> object:
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(dp_mod, "write_manifest", _raise_manifest)
        success = PipelineSuccess(
            outcome=DocumentBackendOutcome("mineru-vlm", "vlm", 5, 0),
            fallback_chain=[], skipped_tiers=[],
            successful_tier_elapsed_ms=1234,
        )
        insp = TargetInspection(
            is_local=False, size_bytes=100, size_mb=0.01, pages=5,
            name="a.pdf", mime="application/pdf", kind="pdf",
        )
        out = _finalize(success, "https://x/a.pdf", insp, tmp_path, "slug")
        assert "Document parsed and saved." in out
