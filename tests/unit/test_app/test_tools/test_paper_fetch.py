"""Tests for the paper_fetch builtin tool."""
from __future__ import annotations

import json
import sys

import pytest

from agent_app.tools.paper.paper_fetch import (
    _ArxivIdCheck,
    _fetch_arxiv_metadata,
    _fetch_full_content,
    _format_metadata,
    paper_fetch,
)


class TestFormatMetadata:
    def test_full_arxiv_metadata(self) -> None:
        paper = {
            "title": "Attention Is All You Need",
            "arxiv_id": "1706.03762",
            "authors": ["Vaswani", "Shazeer", "Parmar"],
            "published": "2017-06-12",
            "abstract": "The dominant sequence transduction models...",
            "categories": ["cs.CL", "cs.AI"],
            "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
            "abs_url": "https://arxiv.org/abs/1706.03762",
        }
        result = _format_metadata(paper)
        assert "# Attention Is All You Need" in result
        assert "1706.03762" in result
        assert "Vaswani" in result
        assert "## Abstract" in result
        assert "paper_fetch" in result

    def test_full_s2_metadata(self) -> None:
        paper = {
            "title": "Test Paper",
            "s2_id": "abc123",
            "doi": "10.1234/xxx",
            "authors": ["Alice"],
            "year": 2023,
            "venue": "NeurIPS",
            "citation_count": 50,
            "reference_count": 30,
            "fields_of_study": ["Computer Science"],
            "publication_types": ["Conference"],
            "abstract": "We propose...",
            "tldr": "A short summary.",
            "pdf_url": "https://example.com/paper.pdf",
        }
        result = _format_metadata(paper)
        assert "# Test Paper" in result
        assert "NeurIPS" in result
        assert "Citations" in result
        assert "References" in result
        assert "Fields of Study" in result
        assert "## TL;DR" in result

    def test_minimal_metadata(self) -> None:
        paper = {"title": "Test"}
        result = _format_metadata(paper)
        assert "# Test" in result

    def test_metadata_actionable_guidance(self) -> None:
        paper = {"title": "X", "arxiv_id": "2301.07041"}
        result = _format_metadata(paper)
        assert 'mode="full"' in result
        assert "paper_fetch" in result

    def test_metadata_no_guidance_without_id(self) -> None:
        paper = {"title": "X"}
        result = _format_metadata(paper)
        assert "paper_fetch" not in result

    def test_publication_date_preferred_over_year(self) -> None:
        paper = {"title": "X", "publication_date": "2023-06-15", "year": 2023}
        result = _format_metadata(paper)
        assert "2023-06-15" in result
        assert "Year" not in result


class TestPaperFetchTool:
    async def test_empty_id(self) -> None:
        result = await paper_fetch.execute(paper_id="")
        assert "Error" in result

    async def test_unknown_mode(self) -> None:
        result = await paper_fetch.execute(paper_id="test", mode="unknown")
        assert "Error" in result
        assert "metadata" in result

    async def test_unknown_source(self) -> None:
        result = await paper_fetch.execute(
            paper_id="test", mode="metadata", source="unknown"
        )
        assert "Error" in result

    async def test_unknown_source_full_mode(self) -> None:
        result = await paper_fetch.execute(
            paper_id="test", mode="full", source="unknown"
        )
        assert "Error" in result
        assert "semantic_scholar" in result

    def test_schema_params(self) -> None:
        schema = paper_fetch.get_schema()
        assert schema.name == "paper_fetch"
        assert (
            schema.description
            == "Fetch a specific paper by ID, returning full text or metadata."
        )
        props = schema.parameters["properties"]
        assert "paper_id" in props
        assert "mode" in props
        assert "source" in props
        assert props["mode"]["enum"] == ["metadata", "full"]
        assert props["source"]["enum"] == ["arxiv", "semantic_scholar"]


class _FakeTimeout:
    def __init__(self, total: int) -> None:
        self.total = total


class _FakeResponse:
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self._body = body
        self.charset = "utf-8"

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    async def text(self) -> str:
        return self._body

    async def read(self) -> bytes:
        return self._body.encode("utf-8")


class _FakeSession:
    def __init__(
        self,
        statuses: list[int],
        calls: list[int],
        bodies: list[str] | None = None,
    ) -> None:
        self._statuses = statuses
        self._calls = calls
        self._bodies = bodies or []

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def request(
        self,
        method: str,
        url: str,
        **kwargs: object,
    ) -> _FakeResponse:
        _ = (method, url, kwargs)
        self._calls.append(1)
        idx = min(len(self._calls) - 1, len(self._statuses) - 1)
        status = self._statuses[idx]
        body = self._bodies[min(idx, len(self._bodies) - 1)] if self._bodies else f"status={status}"
        return _FakeResponse(status, body)


class _FakeAiohttpModule:
    class ClientError(Exception):
        pass

    ClientTimeout = _FakeTimeout

    def __init__(
        self,
        statuses: list[int],
        calls: list[int],
        bodies: list[str] | None = None,
    ) -> None:
        self._statuses = statuses
        self._calls = calls
        self._bodies = bodies

    def ClientSession(self) -> _FakeSession:  # noqa: N802
        return _FakeSession(self._statuses, self._calls, self._bodies)


class TestFetchArxivMetadata:
    @pytest.mark.asyncio
    async def test_fetch_xml_exception_returns_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paper_fetch_module = sys.modules[_fetch_arxiv_metadata.__module__]
        paper_search_module = sys.modules["agent_app.tools.paper.paper_search"]

        async def _fail_fetch_xml(url: str) -> object:
            _ = url
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(paper_search_module, "_fetch_xml", _fail_fetch_xml)
        result = await paper_fetch_module._fetch_arxiv_metadata("2301.07041")
        assert result.startswith("Error: arXiv request failed:")

    @pytest.mark.asyncio
    async def test_fetch_xml_exception_does_not_duplicate_prefix(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paper_fetch_module = sys.modules[_fetch_arxiv_metadata.__module__]
        paper_search_module = sys.modules["agent_app.tools.paper.paper_search"]

        async def _fail_fetch_xml(url: str) -> object:
            _ = url
            raise RuntimeError("arXiv request failed: simulated failure")

        monkeypatch.setattr(paper_search_module, "_fetch_xml", _fail_fetch_xml)
        result = await paper_fetch_module._fetch_arxiv_metadata("2301.07041")
        assert result == "Error: arXiv request failed: simulated failure"


class TestPaperFetchSemanticScholarParsing:
    @pytest.mark.asyncio
    async def test_metadata_invalid_json_returns_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paper_fetch_module = sys.modules[_fetch_arxiv_metadata.__module__]
        call_markers: list[int] = []
        fake_aiohttp = _FakeAiohttpModule([200], call_markers, bodies=["not-json"])
        monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
        monkeypatch.setattr(
            paper_fetch_module,
            "_CFG",
            paper_fetch_module.PaperFetchConfig(max_full_tokens=15_000, html_fetch_timeout=30),
        )

        result = await paper_fetch.execute(
            paper_id="DOI:10.1000/test",
            mode="metadata",
            source="semantic_scholar",
        )
        assert result.startswith("Error: failed to parse Semantic Scholar response:")

    @pytest.mark.asyncio
    async def test_metadata_non_object_json_returns_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paper_fetch_module = sys.modules[_fetch_arxiv_metadata.__module__]
        call_markers: list[int] = []
        fake_aiohttp = _FakeAiohttpModule(
            [200], call_markers, bodies=[json.dumps([1, 2, 3])]
        )
        monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
        monkeypatch.setattr(
            paper_fetch_module,
            "_CFG",
            paper_fetch_module.PaperFetchConfig(max_full_tokens=15_000, html_fetch_timeout=30),
        )

        result = await paper_fetch.execute(
            paper_id="DOI:10.1000/test",
            mode="metadata",
            source="semantic_scholar",
        )
        assert result == "Error: unexpected Semantic Scholar response format"


class TestPaperFetchFullPathRetries:
    @pytest.mark.asyncio
    async def test_try_arxiv_html_returns_none_on_retry_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paper_fetch_module = sys.modules[_fetch_arxiv_metadata.__module__]

        async def _fake_http_get_with_retry(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: int = 30,
            retry: object = None,
        ) -> tuple[int, str]:
            _ = (headers, timeout, retry)
            assert "arxiv.org/html/" in url
            raise RuntimeError("transient failure")

        monkeypatch.setattr(paper_fetch_module, "http_get_with_retry", _fake_http_get_with_retry)
        result = await paper_fetch_module._try_arxiv_html("2301.07041")
        assert result is None

    @pytest.mark.asyncio
    async def test_try_unpaywall_uses_retry_result_and_parses_json(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paper_fetch_module = sys.modules[_fetch_arxiv_metadata.__module__]

        async def _fake_http_get_with_retry(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: int = 30,
            retry: object = None,
        ) -> tuple[int, str]:
            _ = (headers, timeout, retry)
            assert "api.unpaywall.org" in url
            return 200, json.dumps({"best_oa_location": {"url_for_pdf": "https://oa.example/p.pdf"}})

        monkeypatch.setattr(paper_fetch_module, "http_get_with_retry", _fake_http_get_with_retry)
        result = await paper_fetch_module._try_unpaywall("10.1000/test")
        assert result == "https://oa.example/p.pdf"

    @pytest.mark.asyncio
    async def test_try_unpaywall_invalid_json_returns_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paper_fetch_module = sys.modules[_fetch_arxiv_metadata.__module__]

        async def _fake_http_get_with_retry(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: int = 30,
            retry: object = None,
        ) -> tuple[int, str]:
            _ = (url, headers, timeout, retry)
            return 200, "not-json"

        monkeypatch.setattr(paper_fetch_module, "http_get_with_retry", _fake_http_get_with_retry)
        result = await paper_fetch_module._try_unpaywall("10.1000/test")
        assert result.startswith("Error: failed to parse Unpaywall response for DOI 10.1000/test:")


class TestFetchFullArxivPrecheck:
    @pytest.mark.asyncio
    async def test_html_hit_skips_id_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = sys.modules[_fetch_full_content.__module__]
        called = {"id_check": False}

        async def _html(arxiv_id: str) -> str:
            return "full body " * 100

        async def _idcheck(clean_id: str) -> _ArxivIdCheck:
            called["id_check"] = True
            return _ArxivIdCheck.PRESENT

        monkeypatch.setattr(mod, "_try_arxiv_html", _html)
        monkeypatch.setattr(mod, "_arxiv_id_check", _idcheck)
        out = await _fetch_full_content("2301.07041", "arxiv", None)
        assert "full body" in out
        assert called["id_check"] is False

    @pytest.mark.asyncio
    async def test_missing_fast_fails_before_pdf(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = sys.modules[_fetch_full_content.__module__]
        called = {"pdf": False}

        async def _html(arxiv_id: str) -> None:
            return None

        async def _idcheck(clean_id: str) -> _ArxivIdCheck:
            return _ArxivIdCheck.MISSING

        async def _pdf(pdf_url: str, paper_id: str = "") -> str:
            called["pdf"] = True
            return "should not run"

        monkeypatch.setattr(mod, "_try_arxiv_html", _html)
        monkeypatch.setattr(mod, "_arxiv_id_check", _idcheck)
        monkeypatch.setattr(mod, "_fetch_via_pdf_parser", _pdf)
        out = await _fetch_full_content("9999.99999", "arxiv", None)
        assert out == "Error: no arXiv paper found for ID: 9999.99999"
        assert called["pdf"] is False

    @pytest.mark.asyncio
    async def test_unknown_still_tries_pdf(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = sys.modules[_fetch_full_content.__module__]
        called = {"pdf": False}

        async def _html(arxiv_id: str) -> None:
            return None

        async def _idcheck(clean_id: str) -> _ArxivIdCheck:
            return _ArxivIdCheck.UNKNOWN

        async def _pdf(pdf_url: str, paper_id: str = "") -> str:
            called["pdf"] = True
            return "pdf body text"

        monkeypatch.setattr(mod, "_try_arxiv_html", _html)
        monkeypatch.setattr(mod, "_arxiv_id_check", _idcheck)
        monkeypatch.setattr(mod, "_fetch_via_pdf_parser", _pdf)
        out = await _fetch_full_content("2301.07041", "arxiv", None)
        assert called["pdf"] is True
        assert out == "pdf body text"

    @pytest.mark.asyncio
    async def test_pdf_failure_returned_verbatim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = sys.modules[_fetch_full_content.__module__]

        async def _html(arxiv_id: str) -> None:
            return None

        async def _idcheck(clean_id: str) -> _ArxivIdCheck:
            return _ArxivIdCheck.PRESENT

        async def _pdf(pdf_url: str, paper_id: str = "") -> str:
            return "Error: could not retrieve full text — this paper's PDF (x) failed. Reason: boom"

        monkeypatch.setattr(mod, "_try_arxiv_html", _html)
        monkeypatch.setattr(mod, "_arxiv_id_check", _idcheck)
        monkeypatch.setattr(mod, "_fetch_via_pdf_parser", _pdf)
        out = await _fetch_full_content("2301.07041", "arxiv", None)
        assert out == "Error: could not retrieve full text — this paper's PDF (x) failed. Reason: boom"


class TestFetchViaPdfParserMessage:
    @pytest.mark.asyncio
    async def test_backend_reason_surfaced_without_double_error_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agent_app.tools.paper.paper_fetch import _fetch_via_pdf_parser

        pdf_mod = sys.modules["agent_app.tools.pdf_parser"]

        async def _exec(url: str) -> str:
            _ = url
            return "Error: PDF parsing failed: 文件格式不支持"

        monkeypatch.setattr(pdf_mod.pdf_parser, "execute", _exec)
        out = await _fetch_via_pdf_parser(
            "https://arxiv.org/pdf/2006.16668.pdf", paper_id="2006.16668"
        )
        assert out.startswith(
            "Error: could not retrieve full text — it is extracted from "
            "this paper's PDF (https://arxiv.org/pdf/2006.16668.pdf), "
            "which failed."
        )
        # 格式不支持 -> document-unparseable bucket, scoped to this paper.
        assert "could not be parsed" in out
        assert "unavailable for this specific paper" in out
        # No internal jargon / backend identity / raw body leaks.
        assert "文件格式不支持" not in out
        assert "PDF parsing failed" not in out
        assert "MinerU" not in out and "PaddleOCR" not in out
        assert "Error: Error:" not in out
        assert "Underlying error:" not in out
        assert 'paper_fetch(mode="metadata")' in out

    @pytest.mark.asyncio
    async def test_success_truncates_and_no_error_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agent_app.tools.paper.paper_fetch import _fetch_via_pdf_parser

        pdf_mod = sys.modules["agent_app.tools.pdf_parser"]

        async def _exec(url: str) -> str:
            _ = url
            return "full paper body text"

        monkeypatch.setattr(pdf_mod.pdf_parser, "execute", _exec)
        out = await _fetch_via_pdf_parser("https://arxiv.org/pdf/x.pdf")
        assert out == "full paper body text"
        assert not out.startswith("Error:")


class TestArxivIdCheck:
    @pytest.mark.asyncio
    async def test_empty_entries_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import xml.etree.ElementTree as ET

        from agent_app.tools.paper.paper_fetch import _arxiv_id_check

        ps = sys.modules["agent_app.tools.paper.paper_search"]

        async def _xml(url: str) -> ET.Element:
            return ET.fromstring('<feed xmlns="http://www.w3.org/2005/Atom"></feed>')

        monkeypatch.setattr(ps, "_fetch_xml", _xml)
        assert await _arxiv_id_check("9999.99999") is _ArxivIdCheck.MISSING

    @pytest.mark.asyncio
    async def test_entry_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import xml.etree.ElementTree as ET

        from agent_app.tools.paper.paper_fetch import _arxiv_id_check

        ps = sys.modules["agent_app.tools.paper.paper_search"]

        async def _xml(url: str) -> ET.Element:
            return ET.fromstring(
                '<feed xmlns="http://www.w3.org/2005/Atom"><entry/></feed>'
            )

        monkeypatch.setattr(ps, "_fetch_xml", _xml)
        assert await _arxiv_id_check("1706.03762") is _ArxivIdCheck.PRESENT

    @pytest.mark.asyncio
    async def test_exception_is_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agent_app.tools.paper.paper_fetch import _arxiv_id_check

        ps = sys.modules["agent_app.tools.paper.paper_search"]

        async def _xml(url: str) -> object:
            raise RuntimeError("arXiv API returned HTTP 429")

        monkeypatch.setattr(ps, "_fetch_xml", _xml)
        assert await _arxiv_id_check("2301.07041") is _ArxivIdCheck.UNKNOWN
