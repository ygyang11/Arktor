"""Tests for HTTP retry helpers."""
from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from agent_harness.core.errors import HttpResponseTooLargeError
from agent_harness.utils import host_throttler, http_retry as http_retry_module
from agent_harness.utils.http_retry import (
    HttpRetryConfig,
    HttpTextResponse,
    _backoff_seconds,
    _decode_body,
    _parse_retry_after,
    _read_capped,
)


class TestDecodeBody:
    def test_valid_charset_strict(self) -> None:
        assert _decode_body("héllo".encode("latin-1"), "latin-1") == "héllo"

    def test_none_charset_defaults_utf8(self) -> None:
        assert _decode_body("café".encode(), None) == "café"

    def test_bogus_charset_falls_back_utf8_replace(self) -> None:
        # Unknown charset -> LookupError -> utf-8/replace fallback, no crash.
        out = _decode_body("café".encode(), "not-a-real-charset-xyz")
        assert "caf" in out

    def test_undecodable_bytes_replace_not_crash(self) -> None:
        out = _decode_body(b"\xff\xfe\x00bad", "utf-8")
        assert isinstance(out, str)  # replaced, no UnicodeDecodeError


@pytest.fixture(autouse=True)
def _reset_throttler() -> None:
    host_throttler._STATES.clear()


class _FakeTimeout:
    def __init__(self, total: int) -> None:
        self.total = total


class _FakeContent:
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def iter_chunked(self, n: int) -> AsyncIterator[bytes]:
        for i in range(0, len(self._body), n):
            yield self._body[i : i + n]


class _FakeResponse:
    def __init__(self, status: int, body: str, headers: dict[str, str]) -> None:
        self.status = status
        self._body = body
        self.headers = headers
        self.charset = "utf-8"
        self.content = _FakeContent(body.encode())

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    async def read(self) -> bytes:
        return self._body.encode()

    async def text(self) -> str:
        return self._body


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def request(self, method: str, url: str, **kwargs: object) -> _FakeResponse:
        _ = (method, url, kwargs)
        return self._response


class _FakeAiohttpModule:
    class ClientError(Exception):
        pass

    ClientTimeout = _FakeTimeout

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    def ClientSession(self) -> _FakeSession:  # noqa: N802
        return _FakeSession(self._response)


class TestHttpTextRetry:
    @pytest.mark.asyncio
    async def test_http_get_text_with_retry_returns_headers_and_body(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setitem(
            sys.modules,
            "aiohttp",
            _FakeAiohttpModule(
                _FakeResponse(
                    status=200,
                    body="hello",
                    headers={"Content-Type": "text/plain", "X-Test": "1"},
                )
            ),
        )

        response = await http_retry_module.http_get_text_with_retry("https://example.com")
        assert response == HttpTextResponse(
            status=200,
            headers={"Content-Type": "text/plain", "X-Test": "1"},
            body="hello",
        )

    def test_parse_retry_after_delta_seconds(self) -> None:
        assert _parse_retry_after("120") == 120.0
        assert _parse_retry_after("0") == 0.0
        assert _parse_retry_after("-5") == 0.0
        assert _parse_retry_after("3.7") == 3.7

    def test_parse_retry_after_http_date_future(self) -> None:
        future = datetime.now(tz=UTC) + timedelta(seconds=45)
        value = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
        parsed = _parse_retry_after(value)
        assert parsed is not None
        assert 40.0 <= parsed <= 50.0

    def test_parse_retry_after_http_date_past_clamps_to_zero(self) -> None:
        assert _parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0

    def test_parse_retry_after_invalid_returns_none(self) -> None:
        assert _parse_retry_after("") is None
        assert _parse_retry_after("garbage") is None

    def test_backoff_seconds_prefers_retry_after_header(self) -> None:
        retry = HttpRetryConfig(max_attempts=3, base_delay=1.0, max_sleep=60.0)
        delay = _backoff_seconds({"Retry-After": "15"}, attempt=0, retry=retry)
        assert delay == 15.0

    def test_backoff_seconds_caps_retry_after_at_max_sleep(self) -> None:
        retry = HttpRetryConfig(max_attempts=3, base_delay=1.0, max_sleep=10.0)
        delay = _backoff_seconds({"Retry-After": "120"}, attempt=0, retry=retry)
        assert delay == 10.0

    def test_backoff_seconds_falls_back_to_exponential_when_no_header(self) -> None:
        retry = HttpRetryConfig(max_attempts=3, base_delay=2.0, max_sleep=60.0)
        assert _backoff_seconds({}, attempt=0, retry=retry) == 2.0
        assert _backoff_seconds({}, attempt=1, retry=retry) == 4.0
        assert _backoff_seconds({}, attempt=2, retry=retry) == 8.0

    def test_backoff_seconds_caps_exponential_at_max_sleep(self) -> None:
        retry = HttpRetryConfig(max_attempts=10, base_delay=2.0, max_sleep=5.0)
        assert _backoff_seconds({}, attempt=4, retry=retry) == 5.0

    def test_backoff_seconds_falls_back_when_retry_after_unparseable(self) -> None:
        retry = HttpRetryConfig(max_attempts=3, base_delay=2.0, max_sleep=60.0)
        delay = _backoff_seconds({"Retry-After": "garbage"}, attempt=1, retry=retry)
        assert delay == 4.0

    def test_backoff_seconds_case_insensitive_header_lookup(self) -> None:
        retry = HttpRetryConfig(max_attempts=3, base_delay=1.0, max_sleep=60.0)
        for header_name in ("retry-after", "RETRY-AFTER", "Retry-after", "rETrY-AfTeR"):
            delay = _backoff_seconds({header_name: "7"}, attempt=0, retry=retry)
            assert delay == 7.0, f"failed for header name {header_name!r}"

    @pytest.mark.asyncio
    async def test_http_get_with_retry_keeps_legacy_tuple_shape(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def _fake_http_get_text_with_retry(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: int = 30,
            retry: object = None,
            max_bytes: int | None = None,
        ) -> HttpTextResponse:
            _ = (url, headers, timeout, retry, max_bytes)
            return HttpTextResponse(status=204, headers={"X-Test": "1"}, body="")

        monkeypatch.setattr(
            http_retry_module,
            "http_get_text_with_retry",
            _fake_http_get_text_with_retry,
        )

        status, body = await http_retry_module.http_get_with_retry("https://example.com")
        assert (status, body) == (204, "")


class _CappedResp:
    def __init__(self, body: bytes) -> None:
        self.content = _FakeContent(body)
        self._read_called = False

    async def read(self) -> bytes:
        self._read_called = True
        return b"".join([c async for c in self.content.iter_chunked(64 * 1024)])


class TestReadCapped:
    @pytest.mark.asyncio
    async def test_max_bytes_none_keeps_full_read(self) -> None:
        resp = _CappedResp(b"x" * 5000)
        out = await _read_capped(resp, None)  # type: ignore[arg-type]
        assert out == b"x" * 5000
        assert resp._read_called is True

    @pytest.mark.asyncio
    async def test_aborts_when_exceeding_max_bytes(self) -> None:
        # 300 KB body, cap at 100 KB: must raise before fully consuming.
        consumed = 0

        class _Counting:
            async def iter_chunked(self, n: int) -> AsyncIterator[bytes]:
                nonlocal consumed
                for _ in range(300):
                    consumed += 1024
                    yield b"y" * 1024

        resp = type("R", (), {"content": _Counting()})()
        with pytest.raises(HttpResponseTooLargeError) as ei:
            await _read_capped(resp, 100 * 1024)  # type: ignore[arg-type]
        assert ei.value.limit == 100 * 1024
        # Stopped near the cap, did not drain all 300 KB.
        assert consumed <= 100 * 1024 + 1024

    @pytest.mark.asyncio
    async def test_web_fetch_too_large_propagates_out_of_retry_loop(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        big = _FakeResponse(status=200, body="z" * 200_000, headers={})
        monkeypatch.setitem(sys.modules, "aiohttp", _FakeAiohttpModule(big))
        with pytest.raises(HttpResponseTooLargeError):
            await http_retry_module.http_get_text_with_retry(
                "https://example.com", max_bytes=1024
            )
