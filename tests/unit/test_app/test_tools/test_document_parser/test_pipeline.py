"""Unit tests for document_parser.pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import aiohttp
import pytest

from agent_app.tools.document_parser.backends import (
    DocumentBackend,
    DocumentBackendOutcome,
)
from agent_app.tools.document_parser.errors import (
    DocumentBackendError,
    DocumentErrorClass,
    NoViableDocumentBackend,
)
from agent_app.tools.document_parser.pipeline import preflight, run_pipeline
from agent_app.tools.document_parser.storage import TargetInspection


class _FakeBackend(DocumentBackend):
    def __init__(
        self,
        name: str,
        *,
        needs_key: str | None = None,
        max_mb_local: float | None = None,
        max_mb_url: float | None = None,
        max_pages: int | None = None,
        url_outcome: DocumentBackendOutcome | DocumentBackendError | None = None,
        local_outcome: DocumentBackendOutcome | DocumentBackendError | None = None,
    ) -> None:
        self.name = name
        self.model = name
        self.needs_key = needs_key
        self.max_mb_local = max_mb_local
        self.max_mb_url = max_mb_url
        self.max_pages = max_pages
        self._url = url_outcome
        self._local = local_outcome
        self.url_calls = 0
        self.local_calls = 0

    async def parse_local(
        self, session: aiohttp.ClientSession, file_path: Path, dest_dir: Path,
    ) -> DocumentBackendOutcome:
        self.local_calls += 1
        if isinstance(self._local, DocumentBackendError):
            raise self._local
        if self._local is None:
            raise DocumentBackendError(
                DocumentErrorClass.BACKEND_READ_FAILED, None, f"{self.name} local unset",
            )
        return self._local

    async def parse_url(
        self, url: str, dest_dir: Path,
    ) -> DocumentBackendOutcome:
        self.url_calls += 1
        if isinstance(self._url, DocumentBackendError):
            raise self._url
        if self._url is None:
            raise DocumentBackendError(
                DocumentErrorClass.BACKEND_READ_FAILED, None, f"{self.name} url unset",
            )
        return self._url


def _insp(
    *, is_local: bool = False, size_mb: float | None = None,
    pages: int | None = None,
) -> TargetInspection:
    return TargetInspection(
        is_local=is_local,
        size_bytes=int(size_mb * 1024 * 1024) if size_mb else None,
        size_mb=size_mb,
        pages=pages,
        name="x.pdf",
        mime="application/pdf",
        kind="pdf",
    )


class TestPreflight:
    def test_drop_when_key_missing(self) -> None:
        b = _FakeBackend("paddleocr-vl-1.5", needs_key="paddleocr")
        plan, skipped = preflight([b], _insp(), {})
        assert plan == []
        assert skipped == [{"tier": "paddleocr-vl-1.5", "reason": "no_api_key:paddleocr"}]

    def test_size_filter_uses_local_vs_url(self) -> None:
        b = _FakeBackend("paddleocr-vl-1.5", max_mb_local=50.0, max_mb_url=200.0)
        _, skipped = preflight([b], _insp(is_local=True, size_mb=100.0), {})
        assert skipped[0]["reason"] == "size>50.0MB(local)"
        _, skipped = preflight([b], _insp(is_local=False, size_mb=250.0), {})
        assert skipped[0]["reason"] == "size>200.0MB(url)"

    def test_pages_filter(self) -> None:
        b = _FakeBackend("paddleocr-vl", max_pages=10)
        _, skipped = preflight([b], _insp(pages=20), {})
        assert skipped[0]["reason"] == "pages>10"

    def test_all_filtered_raises_no_viable(self) -> None:
        b = _FakeBackend("p", needs_key="paddleocr")
        plan, skipped = preflight([b], _insp(), {})
        assert plan == []
        assert skipped


async def _noop_download() -> Path:
    raise AssertionError("downloader must not be invoked here")


class TestRunPipelineHappy:
    async def test_url_first_tier_succeeds(self, tmp_path: Path) -> None:
        outcome = DocumentBackendOutcome("paddleocr-vl-1.5", "PaddleOCR-VL-1.5", 10, 0)
        b1 = _FakeBackend("paddleocr-vl-1.5", url_outcome=outcome)
        b2 = _FakeBackend("mineru-vlm")
        async with aiohttp.ClientSession() as s:
            success = await run_pipeline(
                s, [b1, b2], "https://x/a.pdf", _insp(), tmp_path, {},
                download=_noop_download,
            )
        assert success.outcome.backend_name == "paddleocr-vl-1.5"
        assert b1.url_calls == 1
        assert b2.url_calls == 0

    async def test_local_first_tier_succeeds(self, tmp_path: Path) -> None:
        local = tmp_path / "doc.pdf"
        local.write_bytes(b"%PDF-1.4")
        outcome = DocumentBackendOutcome("mineru-vlm", "vlm", 5, 0)
        b1 = _FakeBackend("mineru-vlm", local_outcome=outcome)
        async with aiohttp.ClientSession() as s:
            success = await run_pipeline(
                s, [b1], str(local), _insp(is_local=True, size_mb=0.1, pages=5),
                tmp_path, {}, download=_noop_download,
            )
        assert success.outcome.backend_name == "mineru-vlm"
        assert b1.local_calls == 1


class TestRunPipelineFallback:
    async def test_fallback_to_next_tier_on_fallback_class(
        self, tmp_path: Path,
    ) -> None:
        outcome = DocumentBackendOutcome("mineru-vlm", "vlm", 8, 0)
        b1 = _FakeBackend(
            "paddleocr-vl-1.5",
            url_outcome=DocumentBackendError(
                DocumentErrorClass.AUTH_FAILED, 401, "bad token",
            ),
        )
        b2 = _FakeBackend("mineru-vlm", url_outcome=outcome)
        async with aiohttp.ClientSession() as s:
            success = await run_pipeline(
                s, [b1, b2], "https://x/a.pdf", _insp(), tmp_path, {},
                download=_noop_download,
            )
        assert success.outcome.backend_name == "mineru-vlm"
        assert b1.url_calls == 1
        assert b2.url_calls == 1
        assert len(success.fallback_chain) == 1


class TestRunPipelineShortCircuit:
    async def test_invalid_input_short_circuits(self, tmp_path: Path) -> None:
        b1 = _FakeBackend(
            "paddleocr-vl-1.5",
            url_outcome=DocumentBackendError(
                DocumentErrorClass.INVALID_INPUT, None, "bad target",
            ),
        )
        b2 = _FakeBackend(
            "mineru-vlm",
            url_outcome=DocumentBackendOutcome("mineru-vlm", "vlm", 1, 0),
        )
        async with aiohttp.ClientSession() as s:
            with pytest.raises(NoViableDocumentBackend) as ei:
                await run_pipeline(
                    s, [b1, b2], "https://x/a.pdf", _insp(), tmp_path, {},
                    download=_noop_download,
                )
        assert b2.url_calls == 0
        assert "mineru-vlm" in ei.value.unattempted

    async def test_io_error_short_circuits(self, tmp_path: Path) -> None:
        b1 = _FakeBackend(
            "paddleocr-vl-1.5",
            url_outcome=DocumentBackendError(
                DocumentErrorClass.IO_ERROR, None, "disk full",
            ),
        )
        b2 = _FakeBackend("mineru-vlm")
        async with aiohttp.ClientSession() as s:
            with pytest.raises(NoViableDocumentBackend) as ei:
                await run_pipeline(
                    s, [b1, b2], "https://x/a.pdf", _insp(), tmp_path, {},
                    download=_noop_download,
                )
        assert b2.url_calls == 0
        assert ei.value.unattempted == ["mineru-vlm"]


class TestRunPipelineUrlToLocal:
    async def test_url_fetch_failed_localizes(self, tmp_path: Path) -> None:
        outcome = DocumentBackendOutcome("paddleocr-vl-1.5", "p", 1, 0)
        b1 = _FakeBackend(
            "paddleocr-vl-1.5",
            url_outcome=DocumentBackendError(
                DocumentErrorClass.BACKEND_FETCH_FAILED, 10002, "url not recognized",
            ),
            local_outcome=outcome,
        )
        downloaded: list[int] = []

        local_file = tmp_path / "downloaded.pdf"
        local_file.write_bytes(b"%PDF-1.4")

        async def _download() -> Path:
            downloaded.append(1)
            return local_file

        async with aiohttp.ClientSession() as s:
            success = await run_pipeline(
                s, [b1], "https://x/a.pdf", _insp(), tmp_path, {},
                download=_download,
            )
        assert downloaded == [1]
        assert b1.url_calls == 1
        assert b1.local_calls == 1
        assert success.outcome.backend_name == "paddleocr-vl-1.5"


class TestLocalizeDownloadFailedFallsThrough:
    """A DOWNLOAD_FAILED in localize must NOT short-circuit the pipeline —
    the next tier has its own URL fetcher (different transport, region,
    auth) and might still succeed. Regression guard for the LSE 302 case
    where our client-side download couldn't follow the redirect but the
    server-side fetcher of the fallback tier could.
    """

    async def test_localize_download_failed_lets_next_tier_try_url(
        self, tmp_path: Path,
    ) -> None:
        outcome = DocumentBackendOutcome("mineru-vlm", "vlm", 18, 0)
        # paddle url fails 10004 → triggers localize → localize 302/DOWNLOAD_FAILED
        # → next tier (mineru) should still get a chance at URL
        b1 = _FakeBackend(
            "paddleocr-vl-1.5",
            url_outcome=DocumentBackendError(
                DocumentErrorClass.UNSUPPORTED_BY_TIER, 10004, "fmt",
            ),
        )
        b2 = _FakeBackend("mineru-vlm", url_outcome=outcome)

        async def _fail_download() -> Path:
            raise DocumentBackendError(
                DocumentErrorClass.DOWNLOAD_FAILED, 302,
                "HTTP 302 while downloading URL for localization",
            )

        async with aiohttp.ClientSession() as s:
            success = await run_pipeline(
                s, [b1, b2], "https://x/a.pdf", _insp(), tmp_path, {},
                download=_fail_download,
            )
        assert success.outcome.backend_name == "mineru-vlm"
        # chain must show paddle url 10004 + localize DOWNLOAD_FAILED
        classes = {a.error_class for a in success.fallback_chain}
        assert "UNSUPPORTED_BY_TIER" in classes
        assert "DOWNLOAD_FAILED" in classes

    async def test_localize_failure_cached_across_tiers(
        self, tmp_path: Path,
    ) -> None:
        """A 2-tier pipeline where both want to localize the same URL.
        The second tier must NOT re-attempt the download — it should
        see the cached failure and abort localize immediately."""
        b1 = _FakeBackend(
            "paddleocr-vl-1.5",
            url_outcome=DocumentBackendError(
                DocumentErrorClass.UNSUPPORTED_BY_TIER, 10004, "fmt",
            ),
        )
        b2 = _FakeBackend(
            "paddleocr-vl",
            url_outcome=DocumentBackendError(
                DocumentErrorClass.UNSUPPORTED_BY_TIER, 10004, "fmt",
            ),
        )
        b3 = _FakeBackend(
            "mineru-vlm",
            url_outcome=DocumentBackendOutcome("mineru-vlm", "vlm", 1, 0),
        )

        download_attempts: list[int] = []

        async def _fail_download() -> Path:
            download_attempts.append(1)
            raise DocumentBackendError(
                DocumentErrorClass.DOWNLOAD_FAILED, 302, "redirect",
            )

        async with aiohttp.ClientSession() as s:
            success = await run_pipeline(
                s, [b1, b2, b3], "https://x/a.pdf", _insp(), tmp_path, {},
                download=_fail_download,
            )
        assert success.outcome.backend_name == "mineru-vlm"
        # The download was attempted ONCE despite two tiers needing localize
        assert len(download_attempts) == 1


class TestRunPipelineNaturalExhaustion:
    async def test_no_unattempted_when_all_tried(self, tmp_path: Path) -> None:
        err = DocumentBackendError(
            DocumentErrorClass.AUTH_FAILED, 401, "bad token",
        )
        b1 = _FakeBackend("paddleocr-vl-1.5", url_outcome=err)
        b2 = _FakeBackend("mineru-vlm", url_outcome=err)
        async with aiohttp.ClientSession() as s:
            with pytest.raises(NoViableDocumentBackend) as ei:
                await run_pipeline(
                    s, [b1, b2], "https://x/a.pdf", _insp(), tmp_path, {},
                    download=_noop_download,
                )
        assert ei.value.unattempted == []
        assert len(ei.value.fallback_chain) == 2


class TestRunPipelineElapsedMs:
    async def test_failed_attempts_record_nonnegative_elapsed(
        self, tmp_path: Path,
    ) -> None:
        err = DocumentBackendError(
            DocumentErrorClass.AUTH_FAILED, 401, "bad token",
        )
        b1 = _FakeBackend("paddleocr-vl-1.5", url_outcome=err)
        b2 = _FakeBackend(
            "mineru-vlm",
            url_outcome=DocumentBackendOutcome("mineru-vlm", "vlm", 1, 0),
        )
        async with aiohttp.ClientSession() as s:
            success = await run_pipeline(
                s, [b1, b2], "https://x/a.pdf", _insp(), tmp_path, {},
                download=_noop_download,
            )
        assert len(success.fallback_chain) == 1
        a = success.fallback_chain[0]
        assert a.tier == "paddleocr-vl-1.5"
        assert a.elapsed_ms >= 0

    async def test_localize_failure_records_elapsed(
        self, tmp_path: Path,
    ) -> None:
        b1 = _FakeBackend(
            "paddleocr-vl-1.5",
            url_outcome=DocumentBackendError(
                DocumentErrorClass.BACKEND_FETCH_FAILED, 10002, "url unrecognized",
            ),
        )

        async def _failing_download() -> Path:
            raise DocumentBackendError(
                DocumentErrorClass.DOWNLOAD_FAILED, None, "net broke",
            )

        async with aiohttp.ClientSession() as s:
            with pytest.raises(NoViableDocumentBackend) as ei:
                await run_pipeline(
                    s, [b1], "https://x/a.pdf", _insp(), tmp_path, {},
                    download=_failing_download,
                )
        modes = [a.mode for a in ei.value.fallback_chain]
        assert "localize" in modes
        for a in ei.value.fallback_chain:
            assert a.elapsed_ms >= 0
