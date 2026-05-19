"""Tests for web_fetch tool."""
from __future__ import annotations

import sys

import pytest

from agent_harness.core.errors import HttpResponseTooLargeError
from agent_harness.utils.http_retry import HttpRetryConfig, HttpTextResponse
from agent_app.tools.web.web_fetch import (
    _CFG,
    _extract_from_html,
    _extract_text_from_html,
    _format_response,
    _is_binary_content_type,
    _is_pdf,
    _reject_internal_host,
    _validate_url,
    web_fetch,
)


def _noop_host(host: str) -> None:
    return None


class TestWebFetchValidation:
    @pytest.mark.asyncio
    async def test_empty_url_returns_error(self) -> None:
        result = await web_fetch.execute(url="")
        assert result.startswith("Error:")
        assert "empty" in result

    @pytest.mark.asyncio
    async def test_file_scheme_is_blocked(self) -> None:
        result = await web_fetch.execute(url="file:///etc/passwd")
        assert result.startswith("Error:")
        assert "unsupported URL scheme" in result

    @pytest.mark.asyncio
    async def test_ftp_scheme_is_blocked(self) -> None:
        result = await web_fetch.execute(url="ftp://example.com/file")
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_missing_host_returns_error(self) -> None:
        result = await web_fetch.execute(url="http://")
        assert result.startswith("Error:")
        assert "missing host" in result

    @pytest.mark.asyncio
    async def test_invalid_timeout_returns_error(self) -> None:
        result = await web_fetch.execute(url="https://example.com", timeout=0)
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_no_scheme_returns_error(self) -> None:
        result = await web_fetch.execute(url="not-a-url")
        assert result.startswith("Error:")

    def test_executor_timeout_covers_worst_case_timeout(self) -> None:
        assert web_fetch.executor_timeout is not None
        assert (
            web_fetch.executor_timeout
            >= _CFG.max_timeout * _CFG.retry_max_attempts
        )


class TestWebFetchExecution:
    @pytest.fixture(autouse=True)
    def _bypass_ssrf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = sys.modules["agent_app.tools.web.web_fetch"]
        monkeypatch.setattr(module, "_reject_internal_host", _noop_host)

    @pytest.mark.asyncio
    async def test_html_response_uses_retry_helper(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        module = sys.modules["agent_app.tools.web.web_fetch"]
        captured: dict[str, object] = {}

        async def _fake_http_get_text_with_retry(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: int = 30,
            retry: HttpRetryConfig,
            max_bytes: int | None = None,
            allow_redirects: bool = True,
        ) -> HttpTextResponse:
            captured["url"] = url
            captured["headers"] = headers
            captured["timeout"] = timeout
            captured["retry"] = retry
            return HttpTextResponse(
                status=200,
                headers={"Content-Type": "text/html; charset=utf-8"},
                body="<html><body><h1>Hello</h1><p>world</p></body></html>",
            )

        monkeypatch.setattr(module, "http_get_text_with_retry", _fake_http_get_text_with_retry)

        result = await web_fetch.execute(url="https://example.com", timeout=9)
        assert "world" in result
        assert "<html>" not in result and "<p>" not in result
        assert captured["url"] == "https://example.com"
        assert captured["timeout"] == 9
        assert captured["headers"] == {"User-Agent": module._CFG.user_agent}
        assert isinstance(captured["retry"], HttpRetryConfig)

    @pytest.mark.asyncio
    async def test_timeout_from_retry_helper_is_mapped(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        module = sys.modules["agent_app.tools.web.web_fetch"]

        async def _fake_http_get_text_with_retry(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: int = 30,
            retry: HttpRetryConfig,
            max_bytes: int | None = None,
            allow_redirects: bool = True,
        ) -> HttpTextResponse:
            _ = (url, headers, timeout, retry, max_bytes, allow_redirects)
            raise TimeoutError

        monkeypatch.setattr(module, "http_get_text_with_retry", _fake_http_get_text_with_retry)

        result = await web_fetch.execute(url="https://example.com", timeout=7)
        assert result == "Error: request timed out"

    @pytest.mark.asyncio
    async def test_pdf_content_type_still_short_circuits(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        module = sys.modules["agent_app.tools.web.web_fetch"]

        async def _fake_http_get_text_with_retry(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: int = 30,
            retry: HttpRetryConfig,
            max_bytes: int | None = None,
            allow_redirects: bool = True,
        ) -> HttpTextResponse:
            _ = (url, headers, timeout, retry, max_bytes, allow_redirects)
            return HttpTextResponse(
                status=200,
                headers={"Content-Type": "application/pdf"},
                body="ignored",
            )

        monkeypatch.setattr(module, "http_get_text_with_retry", _fake_http_get_text_with_retry)

        result = await web_fetch.execute(url="https://example.com/file.pdf")
        assert "URL is a PDF document" in result


class TestTextExtractor:
    def test_extracts_visible_text(self) -> None:
        html = "<html><body><h1>Title</h1><p>Hello world</p></body></html>"
        result = _extract_text_from_html(html)
        assert "Title" in result
        assert "Hello world" in result

    def test_strips_script_and_style(self) -> None:
        html = "<script>alert(1)</script><style>.x{}</style><p>visible</p>"
        result = _extract_text_from_html(html)
        assert "alert" not in result
        assert ".x" not in result
        assert "visible" in result

    def test_strips_noscript(self) -> None:
        html = "<noscript>hidden</noscript><p>shown</p>"
        result = _extract_text_from_html(html)
        assert "hidden" not in result
        assert "shown" in result

    def test_collapses_blank_lines(self) -> None:
        html = "<p>a</p><br><br><br><p>b</p>"
        result = _extract_text_from_html(html)
        assert "\n\n\n" not in result
        assert "a" in result
        assert "b" in result

    def test_empty_html(self) -> None:
        assert _extract_text_from_html("") == ""

    def test_nested_skip_tags(self) -> None:
        html = "<script><script>inner</script></script><p>ok</p>"
        result = _extract_text_from_html(html)
        assert "inner" not in result
        assert "ok" in result


class TestFormatResponse:
    def test_json_is_pretty_printed(self) -> None:
        result = _format_response('{"key":"value"}', "application/json")
        assert '"key": "value"' in result

    def test_invalid_json_returns_raw(self) -> None:
        result = _format_response("{broken", "application/json")
        assert result == "{broken"

    def test_html_is_extracted(self) -> None:
        result = _format_response(
            "<html><body><p>Hello</p></body></html>",
            "text/html; charset=utf-8",
        )
        assert "<html>" not in result
        assert "Hello" in result

    def test_plain_text_is_passthrough(self) -> None:
        assert _format_response("raw content", "text/plain") == "raw content"

    def test_empty_content_type_is_passthrough(self) -> None:
        assert _format_response("some data", "") == "some data"


class TestTrafilaturaExtraction:
    def test_uses_trafilatura_when_it_returns_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import trafilatura

        monkeypatch.setattr(
            trafilatura, "extract", lambda *a, **k: "# Title\n\nclean body"
        )
        out = _extract_from_html("<html><nav>junk</nav><p>x</p></html>")
        assert out == "# Title\n\nclean body"
        assert "junk" not in out

    def test_falls_back_when_trafilatura_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import trafilatura

        monkeypatch.setattr(trafilatura, "extract", lambda *a, **k: None)
        out = _extract_from_html("<html><body><p>fallback text</p></body></html>")
        assert "fallback text" in out

    def test_falls_back_when_trafilatura_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import trafilatura

        def _boom(*a: object, **k: object) -> str:
            raise RuntimeError("trafilatura blew up")

        monkeypatch.setattr(trafilatura, "extract", _boom)
        out = _extract_from_html("<html><body><p>still here</p></body></html>")
        assert "still here" in out

    def test_thin_extraction_appends_note(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import trafilatura

        monkeypatch.setattr(trafilatura, "extract", lambda *a, **k: "x")
        big_html = "<html>" + "<div>spa shell</div>" * 1000 + "</html>"
        out = _format_response(big_html, "text/html")
        assert "may be JS-rendered" in out

    def test_no_note_when_extraction_is_substantial(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import trafilatura

        monkeypatch.setattr(trafilatura, "extract", lambda *a, **k: "y" * 600)
        out = _format_response("<html>" + "z" * 6000 + "</html>", "text/html")
        assert "may be JS-rendered" not in out


class TestMaxBytesWiring:
    @pytest.fixture(autouse=True)
    def _bypass_ssrf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = sys.modules["agent_app.tools.web.web_fetch"]
        monkeypatch.setattr(module, "_reject_internal_host", _noop_host)

    @pytest.mark.asyncio
    async def test_passes_max_response_bytes_to_helper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = sys.modules["agent_app.tools.web.web_fetch"]
        captured: dict[str, object] = {}

        async def _fake(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: int = 30,
            retry: HttpRetryConfig,
            max_bytes: int | None = None,
            allow_redirects: bool = True,
        ) -> HttpTextResponse:
            captured["max_bytes"] = max_bytes
            return HttpTextResponse(status=200, headers={}, body="ok")

        monkeypatch.setattr(module, "http_get_text_with_retry", _fake)
        await web_fetch.execute(url="https://example.com")
        assert captured["max_bytes"] == _CFG.max_response_bytes

    @pytest.mark.asyncio
    async def test_too_large_surfaces_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = sys.modules["agent_app.tools.web.web_fetch"]

        async def _fake(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: int = 30,
            retry: HttpRetryConfig,
            max_bytes: int | None = None,
            allow_redirects: bool = True,
        ) -> HttpTextResponse:
            _ = (url, headers, timeout, retry, max_bytes, allow_redirects)
            raise HttpResponseTooLargeError(limit=5 * 1024 * 1024)

        monkeypatch.setattr(module, "http_get_text_with_retry", _fake)
        result = await web_fetch.execute(url="https://example.com/huge")
        assert result.startswith("Error:")
        assert "exceeds web_fetch's 5 MB limit" in result
        assert not result.startswith("Error: [")


class TestCloudflareUaFlip:
    @pytest.fixture(autouse=True)
    def _bypass_ssrf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = sys.modules["agent_app.tools.web.web_fetch"]
        monkeypatch.setattr(module, "_reject_internal_host", _noop_host)

    @pytest.mark.asyncio
    async def test_cf_challenge_triggers_honest_ua_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = sys.modules["agent_app.tools.web.web_fetch"]
        uas: list[str | None] = []

        async def _fake(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: int = 30,
            retry: HttpRetryConfig,
            max_bytes: int | None = None,
            allow_redirects: bool = True,
        ) -> HttpTextResponse:
            _ = (url, timeout, retry, max_bytes)
            uas.append((headers or {}).get("User-Agent"))
            if len(uas) == 1:
                return HttpTextResponse(
                    status=403,
                    headers={"cf-mitigated": "challenge"},
                    body="blocked",
                )
            return HttpTextResponse(
                status=200,
                headers={"Content-Type": "text/plain"},
                body="real content",
            )

        monkeypatch.setattr(module, "http_get_text_with_retry", _fake)
        result = await web_fetch.execute(url="https://example.com")
        assert "real content" in result
        assert len(uas) == 2
        assert uas[0] == module._CFG.user_agent
        assert uas[1] == f"agent-harness/{module._HARNESS_VERSION}"

    @pytest.mark.asyncio
    async def test_plain_403_does_not_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = sys.modules["agent_app.tools.web.web_fetch"]
        calls: list[int] = []

        async def _fake(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: int = 30,
            retry: HttpRetryConfig,
            max_bytes: int | None = None,
            allow_redirects: bool = True,
        ) -> HttpTextResponse:
            _ = (url, headers, timeout, retry, max_bytes, allow_redirects)
            calls.append(1)
            return HttpTextResponse(status=403, headers={}, body="forbidden")

        monkeypatch.setattr(module, "http_get_text_with_retry", _fake)
        result = await web_fetch.execute(url="https://example.com")
        assert result == "Error: HTTP 403 for https://example.com"
        assert len(calls) == 1


class TestSsrfGuard:
    @pytest.mark.parametrize(
        "host",
        ["localhost", "metadata.google.internal", "LocalHost"],
    )
    def test_blacklisted_hosts_blocked(self, host: str) -> None:
        with pytest.raises(ValueError, match="internal/private host blocked"):
            _reject_internal_host(host)

    @pytest.mark.parametrize(
        "ip",
        ["127.0.0.1", "10.0.0.1", "192.168.1.1", "172.16.0.1", "169.254.169.254"],
    )
    def test_private_and_metadata_ips_blocked(self, ip: str) -> None:
        with pytest.raises(ValueError, match="internal/private host blocked"):
            _reject_internal_host(ip)

    def test_public_ip_passes(self) -> None:
        _reject_internal_host("1.1.1.1")  # no raise

    def test_public_hostname_passes_without_resolution(self) -> None:
        _reject_internal_host("nonexistent.invalid")  # no raise

    def test_hostname_resolving_private_is_not_blocked(self) -> None:
        _reject_internal_host("sneaky.example.com")  # no raise

    @pytest.mark.asyncio
    async def test_web_fetch_blocks_localhost(self) -> None:
        result = await web_fetch.execute(url="http://localhost:8080/admin")
        assert result.startswith("Error:")
        assert "internal/private host blocked" in result

    @pytest.mark.asyncio
    async def test_web_fetch_blocks_metadata_ip(self) -> None:
        result = await web_fetch.execute(
            url="http://169.254.169.254/latest/meta-data/"
        )
        assert result.startswith("Error:")
        assert "internal/private host blocked" in result

    @pytest.mark.asyncio
    async def test_validate_url_still_checks_scheme(self) -> None:
        with pytest.raises(ValueError, match="unsupported URL scheme"):
            await _validate_url("ftp://example.com/x")

    @pytest.mark.asyncio
    async def test_redirect_to_private_is_blocked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = sys.modules["agent_app.tools.web.web_fetch"]

        async def _fake(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: int = 30,
            retry: HttpRetryConfig,
            max_bytes: int | None = None,
            allow_redirects: bool = True,
        ) -> HttpTextResponse:
            _ = (headers, timeout, retry, max_bytes)
            assert allow_redirects is False
            return HttpTextResponse(
                status=302,
                headers={"Location": "http://169.254.169.254/latest/meta-data/"},
                body="",
            )

        monkeypatch.setattr(module, "http_get_text_with_retry", _fake)
        result = await web_fetch.execute(url="https://public.example.com/r")
        assert result.startswith("Error:")
        assert "internal/private host blocked" in result

    @pytest.mark.asyncio
    async def test_too_many_redirects(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = sys.modules["agent_app.tools.web.web_fetch"]
        monkeypatch.setattr(module, "_reject_internal_host", _noop_host)

        async def _fake(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: int = 30,
            retry: HttpRetryConfig,
            max_bytes: int | None = None,
            allow_redirects: bool = True,
        ) -> HttpTextResponse:
            _ = (url, headers, timeout, retry, max_bytes, allow_redirects)
            return HttpTextResponse(
                status=302,
                headers={"Location": "https://public.example.com/next"},
                body="",
            )

        monkeypatch.setattr(module, "http_get_text_with_retry", _fake)
        result = await web_fetch.execute(url="https://public.example.com/start")
        assert result.startswith("Error: too many redirects")


class TestBinaryContentTypeDetection:
    def test_pdf_is_not_binary(self) -> None:
        assert _is_binary_content_type("application/pdf") is False

    def test_zip_is_binary(self) -> None:
        assert _is_binary_content_type("application/zip") is True

    def test_octet_stream_is_binary(self) -> None:
        assert _is_binary_content_type("application/octet-stream") is True

    def test_image_png_is_binary(self) -> None:
        assert _is_binary_content_type("image/png") is True

    def test_video_mp4_is_binary(self) -> None:
        assert _is_binary_content_type("video/mp4") is True

    def test_html_is_not_binary(self) -> None:
        assert _is_binary_content_type("text/html; charset=utf-8") is False

    def test_json_is_not_binary(self) -> None:
        assert _is_binary_content_type("application/json") is False

    def test_plain_text_is_not_binary(self) -> None:
        assert _is_binary_content_type("text/plain") is False

    def test_empty_is_not_binary(self) -> None:
        assert _is_binary_content_type("") is False


class TestPdfDetection:
    def test_pdf_content_type(self) -> None:
        assert _is_pdf("application/pdf", "https://example.com/report") is True

    def test_pdf_content_type_with_charset(self) -> None:
        assert _is_pdf("application/pdf; charset=utf-8", "https://example.com/r") is True

    def test_pdf_url_suffix(self) -> None:
        assert _is_pdf("application/octet-stream", "https://example.com/report.pdf") is True

    def test_pdf_url_suffix_case_insensitive(self) -> None:
        assert _is_pdf("application/octet-stream", "https://example.com/Report.PDF") is True

    def test_pdf_url_suffix_with_query_params(self) -> None:
        assert _is_pdf("", "https://cdn.example.com/file.pdf?token=abc") is True

    def test_pdf_empty_content_type_with_pdf_suffix(self) -> None:
        assert _is_pdf("", "https://example.com/doc.pdf") is True

    def test_html_not_pdf(self) -> None:
        assert _is_pdf("text/html", "https://example.com/page") is False

    def test_octet_stream_without_pdf_suffix(self) -> None:
        assert _is_pdf("application/octet-stream", "https://example.com/data.bin") is False

    def test_json_not_pdf(self) -> None:
        assert _is_pdf("application/json", "https://api.example.com/data") is False
