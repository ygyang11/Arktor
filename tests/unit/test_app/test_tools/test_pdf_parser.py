"""Tests for pdf_parser tool."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_harness.core.config import PdfConfig
from agent_app.tools.pdf_parser import (
    _CFG,
    _PDF_EXECUTOR_TIMEOUT,
    _download_mineru_markdown,
    _download_paddleocr_markdown,
    _get_json_with_retry,
    _is_local_file,
    _post_json_with_retry,
    _read_local_file,
    pdf_parser,
)


class TestPdfTimeoutConfig:
    def test_poll_budget_is_300s_wall(self) -> None:
        # PaddleOCR official sample interval = 5s; MinerU sample = 300s.
        assert _CFG.poll_interval == 5.0
        assert _CFG.max_poll_attempts == 60
        assert _CFG.max_poll_attempts * _CFG.poll_interval == 300.0

    def test_pdf_parser_executor_timeout_exceeds_poll_budget(self) -> None:
        assert pdf_parser.executor_timeout == _PDF_EXECUTOR_TIMEOUT
        assert _PDF_EXECUTOR_TIMEOUT > 300.0
        assert pdf_parser.executor_timeout > 30  # not the global default

    def test_paper_fetch_full_timeout_covers_pdf_poll_budget(self) -> None:
        from agent_app.tools.paper.paper_fetch import (
            _FULL_EXECUTOR_TIMEOUT,
            paper_fetch,
        )

        assert paper_fetch.executor_timeout == _FULL_EXECUTOR_TIMEOUT
        assert paper_fetch.executor_timeout >= 360.0
        assert (
            paper_fetch.executor_timeout
            >= _CFG.max_poll_attempts * _CFG.poll_interval
        )


class TestPdfParserValidation:
    @pytest.mark.asyncio
    async def test_empty_url_returns_error(self) -> None:
        result = await pdf_parser.execute(url="")
        assert result.startswith("Error:")
        assert "empty" in result

    @pytest.mark.asyncio
    async def test_whitespace_url_returns_error(self) -> None:
        result = await pdf_parser.execute(url="   ")
        assert result.startswith("Error:")
        assert "empty" in result

    @pytest.mark.asyncio
    async def test_unknown_provider_returns_error(self) -> None:
        fake_cfg = PdfConfig(provider="unknown")
        with patch(
            "agent_app.tools.pdf_parser.resolve_pdf_config",
            return_value=fake_cfg,
        ):
            result = await pdf_parser.execute(url="https://example.com/doc.pdf")
        assert "Unknown PDF provider" in result

    @pytest.mark.asyncio
    async def test_paddleocr_no_api_key_returns_error(self) -> None:
        fake_cfg = PdfConfig(provider="paddleocr", paddleocr_api_key=None)
        with patch(
            "agent_app.tools.pdf_parser.resolve_pdf_config",
            return_value=fake_cfg,
        ):
            result = await pdf_parser.execute(url="https://example.com/doc.pdf")
        assert "PADDLEOCR_API_KEY not set" in result


class TestLocalFileDetection:
    def test_url_is_not_local(self) -> None:
        assert not _is_local_file("https://example.com/doc.pdf")
        assert not _is_local_file("http://example.com/doc.pdf")

    def test_path_is_local(self) -> None:
        assert _is_local_file("/tmp/doc.pdf")
        assert _is_local_file("./doc.pdf")
        assert _is_local_file("doc.pdf")
        assert _is_local_file("~/docs/paper.pdf")


class TestReadLocalFile:
    async def test_read_existing_pdf(self, tmp_path: Path) -> None:
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")
        name, data = await _read_local_file(str(pdf_file))
        assert name == "test.pdf"
        assert data == b"%PDF-1.4 fake content"

    async def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            await _read_local_file("/nonexistent/path/doc.pdf")

    async def test_non_pdf_rejected(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "doc.txt"
        txt_file.write_text("not a pdf")
        with pytest.raises(ValueError, match="Expected a PDF"):
            await _read_local_file(str(txt_file))


class TestLocalFileParsing:
    async def test_local_file_not_found_returns_error(self) -> None:
        fake_cfg = PdfConfig(provider="mineru", mineru_api_key="key")
        with patch("agent_app.tools.pdf_parser.resolve_pdf_config", return_value=fake_cfg):
            result = await pdf_parser.execute(url="/nonexistent/doc.pdf")
        assert result.startswith("Error:")
        assert "not found" in result.lower()

    async def test_local_non_pdf_returns_error(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "doc.txt"
        txt_file.write_text("not a pdf")
        fake_cfg = PdfConfig(provider="mineru", mineru_api_key="key")
        with patch("agent_app.tools.pdf_parser.resolve_pdf_config", return_value=fake_cfg):
            result = await pdf_parser.execute(url=str(txt_file))
        assert result.startswith("Error:")
        assert "PDF" in result


class TestPdfConfigDefaults:
    def test_default_provider_is_mineru(self) -> None:
        cfg = PdfConfig()
        assert cfg.provider == "mineru"

    def test_blank_api_key_becomes_none(self) -> None:
        cfg = PdfConfig(mineru_api_key="  ")
        assert cfg.mineru_api_key is None

    def test_blank_paddleocr_key_becomes_none(self) -> None:
        cfg = PdfConfig(paddleocr_api_key="")
        assert cfg.paddleocr_api_key is None


class TestPdfParserRetryWiring:
    @pytest.mark.asyncio
    async def test_get_json_with_retry_uses_retry_policy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pdf_parser_module = sys.modules[_get_json_with_retry.__module__]
        captured: dict[str, object] = {}

        async def _fake_http_get_with_retry(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: int = 30,
            retry: object,
        ) -> tuple[int, str]:
            captured["url"] = url
            captured["headers"] = headers
            captured["timeout"] = timeout
            captured["retry"] = retry
            return 200, '{"ok": true}'

        monkeypatch.setattr(pdf_parser_module, "http_get_with_retry", _fake_http_get_with_retry)
        status, data, body = await _get_json_with_retry(
            "https://example.com/status",
            headers={"x-test": "1"},
            timeout=9,
        )

        assert status == 200
        assert data == {"ok": True}
        assert body == '{"ok": true}'
        assert captured["url"] == "https://example.com/status"
        assert captured["headers"] == {"x-test": "1"}
        assert captured["timeout"] == 9
        retry = captured["retry"]
        assert isinstance(retry, pdf_parser_module.HttpRetryConfig)
        assert retry.max_attempts == pdf_parser_module._CFG.request_max_attempts
        assert retry.base_delay == pdf_parser_module._CFG.request_base_delay

    @pytest.mark.asyncio
    async def test_post_json_with_retry_uses_retry_policy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pdf_parser_module = sys.modules[_get_json_with_retry.__module__]
        captured: dict[str, object] = {}

        async def _fake_http_post_json_with_retry(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            json_body: object | None = None,
            timeout: int = 30,
            retry: object,
        ) -> tuple[int, str]:
            captured["url"] = url
            captured["headers"] = headers
            captured["json_body"] = json_body
            captured["timeout"] = timeout
            captured["retry"] = retry
            return 200, '{"task_id": "t1"}'

        monkeypatch.setattr(
            pdf_parser_module,
            "http_post_json_with_retry",
            _fake_http_post_json_with_retry,
        )
        status, data, body = await _post_json_with_retry(
            "https://example.com/submit",
            headers={"Authorization": "bearer k"},
            json_body={"url": "https://example.com/a.pdf"},
            timeout=15,
        )

        assert status == 200
        assert data == {"task_id": "t1"}
        assert body == '{"task_id": "t1"}'
        assert captured["url"] == "https://example.com/submit"
        assert captured["headers"] == {"Authorization": "bearer k"}
        assert captured["json_body"] == {"url": "https://example.com/a.pdf"}
        assert captured["timeout"] == 15
        retry = captured["retry"]
        assert isinstance(retry, pdf_parser_module.HttpRetryConfig)
        assert retry.max_attempts == pdf_parser_module._CFG.request_max_attempts
        assert retry.base_delay == pdf_parser_module._CFG.request_base_delay

    @pytest.mark.asyncio
    async def test_download_mineru_markdown_uses_retry_policy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pdf_parser_module = sys.modules[_get_json_with_retry.__module__]
        captured: dict[str, object] = {}

        async def _fake_http_get_bytes_with_retry(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: int = 30,
            retry: object,
        ) -> tuple[int, bytes]:
            captured["url"] = url
            captured["headers"] = headers
            captured["timeout"] = timeout
            captured["retry"] = retry
            return 503, b"service unavailable"

        monkeypatch.setattr(
            pdf_parser_module,
            "http_get_bytes_with_retry",
            _fake_http_get_bytes_with_retry,
        )
        result = await _download_mineru_markdown("https://example.com/result.zip")

        assert result == "Error: failed to download PDF parsing result (HTTP 503)"
        assert captured["url"] == "https://example.com/result.zip"
        assert captured["headers"] is None
        assert captured["timeout"] == 30
        retry = captured["retry"]
        assert isinstance(retry, pdf_parser_module.HttpRetryConfig)
        assert retry.max_attempts == pdf_parser_module._CFG.request_max_attempts
        assert retry.base_delay == pdf_parser_module._CFG.request_base_delay

    @pytest.mark.asyncio
    async def test_download_paddleocr_markdown_uses_retry_policy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pdf_parser_module = sys.modules[_get_json_with_retry.__module__]
        captured: dict[str, object] = {}
        jsonl = '{"result":{"layoutParsingResults":[{"markdown":{"text":"Page 1"}}]}}\n'

        async def _fake_http_get_with_retry(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: int = 30,
            retry: object,
        ) -> tuple[int, str]:
            captured["url"] = url
            captured["headers"] = headers
            captured["timeout"] = timeout
            captured["retry"] = retry
            return 200, jsonl

        monkeypatch.setattr(pdf_parser_module, "http_get_with_retry", _fake_http_get_with_retry)
        result = await _download_paddleocr_markdown("https://example.com/out.jsonl")

        assert result == "Page 1"
        assert captured["url"] == "https://example.com/out.jsonl"
        assert captured["headers"] is None
        assert captured["timeout"] == 30
        retry = captured["retry"]
        assert isinstance(retry, pdf_parser_module.HttpRetryConfig)
        assert retry.max_attempts == pdf_parser_module._CFG.request_max_attempts
        assert retry.base_delay == pdf_parser_module._CFG.request_base_delay
