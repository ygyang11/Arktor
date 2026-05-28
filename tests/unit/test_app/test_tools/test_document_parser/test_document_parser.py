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


class TestSlugLock:
    """Concurrent parses of the same (session_id, slug) must serialize so
    the second caller hits the cache instead of re-running the pipeline.
    Different slugs (even in the same session) must NOT block each other."""

    async def test_get_slug_lock_keys(self) -> None:
        from agent_app.tools.document_parser.document_parser import (
            _get_slug_lock, _slug_locks,
        )

        _slug_locks.clear()
        lock_xa = _get_slug_lock("X", "slug_a")
        lock_xb = _get_slug_lock("X", "slug_b")
        lock_ya = _get_slug_lock("Y", "slug_a")
        lock_anon_a = _get_slug_lock(None, "slug_a")
        lock_anon_a_again = _get_slug_lock(None, "slug_a")

        assert lock_xa is not lock_xb        # same session, different slug
        assert lock_xa is not lock_ya        # different session, same slug
        assert lock_xa is not lock_anon_a    # session vs anonymous
        assert lock_anon_a is lock_anon_a_again  # cached, identity preserved

    async def test_anon_string_session_id_does_not_collide_with_none(
        self,
    ) -> None:
        """session_id="_anon" is a legal string by SAFE_ID_PATTERN; it must
        get its own lock and not be conflated with the None (anonymous)
        bucket. Tuple key (vs the previous "_anon" sentinel string) is
        what guarantees this."""
        from agent_app.tools.document_parser.document_parser import (
            _get_slug_lock, _slug_locks,
        )
        _slug_locks.clear()
        lock_none = _get_slug_lock(None, "slug_a")
        lock_anon_str = _get_slug_lock("_anon", "slug_a")
        assert lock_none is not lock_anon_str

    async def test_concurrent_same_slug_serializes_second_hits_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two concurrent parses of the same target: first runs pipeline,
        second finds cache populated and skips pipeline entirely."""
        import asyncio
        from agent_app.tools.document_parser.backends import DocumentBackendOutcome
        from agent_app.tools.document_parser.document_parser import _slug_locks
        from agent_app.tools.document_parser.pipeline import PipelineSuccess
        _slug_locks.clear()

        monkeypatch.setattr(dp_mod, "session_documents_root", lambda _s: tmp_path)

        async def _fake_inspect(_target: str) -> TargetInspection:
            return TargetInspection(
                is_local=False, size_bytes=1024, size_mb=0.001,
                pages=1, name="a.pdf", mime="application/pdf", kind="pdf",
            )
        monkeypatch.setattr(dp_mod, "inspect_target", _fake_inspect)

        slug = "shared_slug"
        monkeypatch.setattr(
            dp_mod, "make_slug",
            lambda *, source, content_hash, suggested=None: slug,
        )

        pipeline_calls = 0
        ready = asyncio.Event()

        async def _slow_pipeline(*a: object, **kw: object) -> PipelineSuccess:
            nonlocal pipeline_calls
            pipeline_calls += 1
            # Block until both callers have entered (or queued for) parse_document
            await ready.wait()
            dest_dir = tmp_path / slug
            (dest_dir / "content.md").write_text("body")
            return PipelineSuccess(
                outcome=DocumentBackendOutcome("paddleocr-vl-1.5", "p", 1, 0),
                fallback_chain=[], skipped_tiers=[],
                successful_tier_elapsed_ms=10,
            )
        monkeypatch.setattr(dp_mod, "run_pipeline", _slow_pipeline)

        t1 = asyncio.create_task(parse_document(
            target="https://x/a.pdf", session_id="X", slug_hint=None,
        ))
        t2 = asyncio.create_task(parse_document(
            target="https://x/a.pdf", session_id="X", slug_hint=None,
        ))
        # Give both tasks a chance to enter parse_document and contend for lock
        await asyncio.sleep(0.05)
        ready.set()
        await t1
        await t2

        assert pipeline_calls == 1, (
            "Second concurrent call must hit cache, not re-run pipeline"
        )

    async def test_concurrent_different_slugs_run_in_parallel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Different slugs must NOT block each other — lock granularity
        is per-(session, slug), not per-session."""
        import asyncio
        from agent_app.tools.document_parser.backends import DocumentBackendOutcome
        from agent_app.tools.document_parser.document_parser import _slug_locks
        from agent_app.tools.document_parser.pipeline import PipelineSuccess
        _slug_locks.clear()

        monkeypatch.setattr(dp_mod, "session_documents_root", lambda _s: tmp_path)

        async def _fake_inspect(_target: str) -> TargetInspection:
            return TargetInspection(
                is_local=False, size_bytes=1024, size_mb=0.001,
                pages=1, name="a.pdf", mime="application/pdf", kind="pdf",
            )
        monkeypatch.setattr(dp_mod, "inspect_target", _fake_inspect)

        # Different slug per target
        monkeypatch.setattr(
            dp_mod, "make_slug",
            lambda *, source, content_hash, suggested=None: f"slug_{source[-1]}",
        )

        in_flight = 0
        max_in_flight = 0

        async def _track_pipeline(*a: object, **kw: object) -> PipelineSuccess:
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.05)  # simulate work
            in_flight -= 1
            dest_dir = a[4]  # 5th positional arg is dest_dir
            (dest_dir / "content.md").write_text("body")
            return PipelineSuccess(
                outcome=DocumentBackendOutcome("paddleocr-vl-1.5", "p", 1, 0),
                fallback_chain=[], skipped_tiers=[],
                successful_tier_elapsed_ms=10,
            )
        monkeypatch.setattr(dp_mod, "run_pipeline", _track_pipeline)

        t1 = asyncio.create_task(parse_document(
            target="https://x/a", session_id="X", slug_hint=None,
        ))
        t2 = asyncio.create_task(parse_document(
            target="https://x/b", session_id="X", slug_hint=None,
        ))
        await asyncio.gather(t1, t2)
        assert max_in_flight == 2, (
            "Different slugs must parse concurrently (not serialized)"
        )


class TestParseDocumentFailureCleanup:
    async def test_failed_parse_removes_empty_dest_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """All-tiers-fail must not leave the empty `<sid>/<slug>/` dir
        behind. Cumulative failures across repeated stress tests showed
        these accumulating as cruft."""
        from agent_app.tools.document_parser.errors import (
            NoViableDocumentBackend,
        )

        monkeypatch.setattr(dp_mod, "session_documents_root", lambda _s: tmp_path)

        async def _fake_inspect(_target: str) -> TargetInspection:
            return TargetInspection(
                is_local=False, size_bytes=1024, size_mb=0.001,
                pages=1, name="a.pdf", mime="application/pdf", kind="pdf",
            )
        monkeypatch.setattr(dp_mod, "inspect_target", _fake_inspect)

        slug = "boom_deadbeef"
        monkeypatch.setattr(
            dp_mod, "make_slug",
            lambda *, source, content_hash, suggested=None: slug,
        )

        async def _raise_pipeline(*a: object, **kw: object) -> object:
            raise NoViableDocumentBackend(
                skipped=[],
                fallback_chain=[],
                unattempted=[],
            )
        monkeypatch.setattr(dp_mod, "run_pipeline", _raise_pipeline)

        out = await parse_document(
            target="https://x/fail.pdf",
            session_id=None,
            slug_hint=None,
        )
        assert out.startswith("Error: document parsing failed.")
        assert not (tmp_path / slug).exists()

    async def test_failed_parse_with_partial_artifacts_keeps_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If a tier did write something before another tier raised
        (corner case but possible across backends), the partial work
        must NOT be deleted. rmdir() refuses on non-empty dirs so this
        is implicit, but lock it in with a test."""
        from agent_app.tools.document_parser.errors import (
            NoViableDocumentBackend,
        )

        monkeypatch.setattr(dp_mod, "session_documents_root", lambda _s: tmp_path)

        async def _fake_inspect(_target: str) -> TargetInspection:
            return TargetInspection(
                is_local=False, size_bytes=1024, size_mb=0.001,
                pages=1, name="a.pdf", mime="application/pdf", kind="pdf",
            )
        monkeypatch.setattr(dp_mod, "inspect_target", _fake_inspect)

        slug = "partial_cafebabe"
        monkeypatch.setattr(
            dp_mod, "make_slug",
            lambda *, source, content_hash, suggested=None: slug,
        )

        async def _raise_with_partial(*a: object, **kw: object) -> object:
            # Simulate a backend having written a partial file before all
            # tiers eventually failed.
            (tmp_path / slug / "content.md").parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / slug / "content.md").write_text("partial")
            raise NoViableDocumentBackend(
                skipped=[], fallback_chain=[], unattempted=[],
            )
        monkeypatch.setattr(dp_mod, "run_pipeline", _raise_with_partial)

        await parse_document(
            target="https://x/partial.pdf",
            session_id=None,
            slug_hint=None,
        )
        assert (tmp_path / slug / "content.md").exists()


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

    async def test_sends_user_agent_header(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """aiohttp's default User-Agent is rejected (403) by Wikipedia and
        likely other anti-bot CDNs. The downloader must self-identify so
        localize() can actually fetch image/PDF URLs from those sources."""
        captured: dict[str, Any] = {}

        async def _capture(*a: object, **kw: object) -> tuple[int, bytes]:
            captured.update(kw)
            return 200, b"%PDF-1.4"

        monkeypatch.setattr(dp_mod, "http_get_bytes_with_retry", _capture)
        dl = _make_downloader("https://x/a.pdf", mime="application/pdf")
        p = await dl()
        p.unlink(missing_ok=True)

        hdrs = captured.get("headers") or {}
        ua = hdrs.get("User-Agent") or hdrs.get("user-agent")
        assert ua, "downloader must set a User-Agent header"
        assert "Mozilla" not in ua, (
            "browser-impersonating UA breaks IMF (403); use a custom UA"
        )

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
