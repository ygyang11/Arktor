"""HTTP retry helpers for async requests."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Mapping

from agent_harness.core.errors import HttpResponseTooLargeError
from agent_harness.utils.host_throttler import throttle

if TYPE_CHECKING:
    import aiohttp

logger = logging.getLogger(__name__)

_READ_CHUNK = 64 * 1024


@dataclass(frozen=True)
class HttpRetryConfig:
    """Retry policy for HTTP requests."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_sleep: float = 30.0


@dataclass(frozen=True)
class HttpTextResponse:
    """HTTP text response payload with headers preserved."""

    status: int
    headers: Mapping[str, str]
    body: str


DEFAULT_HTTP_RETRY = HttpRetryConfig()


def _is_retryable_status(status: int) -> bool:
    return status == 429 or 500 <= status < 600


def _parse_retry_after(value: str) -> float | None:
    """Parse RFC 7231 Retry-After: delta-seconds or HTTP-date. None if neither."""
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return max(0.0, (dt - datetime.now(tz=UTC)).total_seconds())
    except (TypeError, ValueError):
        return None


def _backoff_seconds(
    headers: Mapping[str, str], attempt: int, retry: HttpRetryConfig,
) -> float:
    """Resolve sleep between attempts: prefer Retry-After (capped), fall back to exp backoff."""
    raw = next(
        (v for k, v in headers.items() if k.lower() == "retry-after"),
        None,
    )
    delay: float | None = _parse_retry_after(raw) if raw else None
    if delay is None:
        delay = retry.base_delay * (2 ** attempt)
    return min(delay, retry.max_sleep)


def _decode_body(raw: bytes, charset: str | None) -> str:
    """Decode with the declared charset; fall back to utf-8/replace.

    Mirrors aiohttp's resilience: a bogus/unknown charset header
    (LookupError) or undecodable bytes (UnicodeDecodeError) degrade to
    utf-8 with replacement instead of crashing, while a valid charset is
    honored strictly (no needless replacement).
    """
    try:
        return raw.decode(charset or "utf-8")
    except (LookupError, UnicodeDecodeError):
        return raw.decode("utf-8", errors="replace")


async def _read_capped(resp: aiohttp.ClientResponse, max_bytes: int | None) -> bytes:
    """Stream the body, aborting once max_bytes is exceeded.

    Raises HttpResponseTooLargeError before the oversized body is fully
    downloaded or decoded. max_bytes=None preserves the full-read behavior.
    """
    if max_bytes is None:
        return await resp.read()
    chunks: list[bytes] = []
    total = 0
    async for chunk in resp.content.iter_chunked(_READ_CHUNK):
        total += len(chunk)
        if total > max_bytes:
            raise HttpResponseTooLargeError(limit=max_bytes)
        chunks.append(chunk)
    return b"".join(chunks)


async def _request_with_retry(
    *,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    json_body: object | None = None,
    retry: HttpRetryConfig = DEFAULT_HTTP_RETRY,
) -> tuple[int, str]:
    import aiohttp  # noqa: PLC0415

    hdrs = dict(headers or {})
    last_exc: Exception | None = None
    last_status = 429
    last_body = "Rate limit exceeded after retries"
    last_headers: dict[str, str] = {}
    attempts = max(1, retry.max_attempts)

    for attempt in range(attempts):
        last_headers = {}
        try:
            async with throttle(url), aiohttp.ClientSession() as session:
                request_kwargs: dict[str, object] = {
                    "headers": hdrs,
                    "timeout": aiohttp.ClientTimeout(total=timeout),
                }
                if json_body is not None:
                    request_kwargs["json"] = json_body
                async with session.request(method, url, **request_kwargs) as resp:  # type: ignore[arg-type]
                    body = await resp.text()
                    if not _is_retryable_status(resp.status):
                        return resp.status, body
                    last_status = resp.status
                    last_body = body
                    last_headers = dict(getattr(resp, "headers", {}))
                    _log_retry_status(resp.status, attempt + 1, attempts)
        except (aiohttp.ClientError, TimeoutError) as exc:
            last_exc = exc
            _log_retry_exception(exc, attempt + 1, attempts)

        if attempt < (attempts - 1):
            await asyncio.sleep(_backoff_seconds(last_headers, attempt, retry))

    if last_exc:
        raise last_exc
    return last_status, last_body


async def _request_text_with_retry(
    *,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    json_body: object | None = None,
    retry: HttpRetryConfig = DEFAULT_HTTP_RETRY,
    max_bytes: int | None = None,
    allow_redirects: bool = True,
) -> HttpTextResponse:
    import aiohttp  # noqa: PLC0415

    hdrs = dict(headers or {})
    last_exc: Exception | None = None
    last_status = 429
    last_headers: dict[str, str] = {}
    last_body = "Rate limit exceeded after retries"
    attempts = max(1, retry.max_attempts)

    for attempt in range(attempts):
        last_headers = {}
        try:
            async with throttle(url), aiohttp.ClientSession() as session:
                request_kwargs: dict[str, object] = {
                    "headers": hdrs,
                    "timeout": aiohttp.ClientTimeout(total=timeout),
                    "allow_redirects": allow_redirects,
                }
                if json_body is not None:
                    request_kwargs["json"] = json_body
                async with session.request(method, url, **request_kwargs) as resp:  # type: ignore[arg-type]
                    raw = await _read_capped(resp, max_bytes)
                    body = _decode_body(raw, resp.charset)
                    response_headers = dict(getattr(resp, "headers", {}))
                    response = HttpTextResponse(
                        status=resp.status,
                        headers=response_headers,
                        body=body,
                    )
                    if not _is_retryable_status(resp.status):
                        return response
                    last_status = response.status
                    last_headers = dict(response.headers)
                    last_body = response.body
                    _log_retry_status(resp.status, attempt + 1, attempts)
        except (aiohttp.ClientError, TimeoutError) as exc:
            last_exc = exc
            _log_retry_exception(exc, attempt + 1, attempts)

        if attempt < (attempts - 1):
            await asyncio.sleep(_backoff_seconds(last_headers, attempt, retry))

    if last_exc:
        raise last_exc
    return HttpTextResponse(status=last_status, headers=last_headers, body=last_body)


async def _request_bytes_with_retry(
    *,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    json_body: object | None = None,
    retry: HttpRetryConfig = DEFAULT_HTTP_RETRY,
    max_bytes: int | None = None,
    allow_redirects: bool = False,
) -> tuple[int, bytes]:
    import aiohttp  # noqa: PLC0415

    hdrs = dict(headers or {})
    last_exc: Exception | None = None
    last_status = 429
    last_body = b""
    last_headers: dict[str, str] = {}
    attempts = max(1, retry.max_attempts)

    for attempt in range(attempts):
        last_headers = {}
        try:
            async with throttle(url), aiohttp.ClientSession() as session:
                request_kwargs: dict[str, object] = {
                    "headers": hdrs,
                    "timeout": aiohttp.ClientTimeout(total=timeout),
                    "allow_redirects": allow_redirects,
                }
                if json_body is not None:
                    request_kwargs["json"] = json_body
                async with session.request(method, url, **request_kwargs) as resp:  # type: ignore[arg-type]
                    body = await _read_capped(resp, max_bytes)
                    if not _is_retryable_status(resp.status):
                        return resp.status, body
                    last_status = resp.status
                    last_body = body
                    last_headers = dict(getattr(resp, "headers", {}))
                    _log_retry_status(resp.status, attempt + 1, attempts)
        except (aiohttp.ClientError, TimeoutError) as exc:
            last_exc = exc
            _log_retry_exception(exc, attempt + 1, attempts)

        if attempt < (attempts - 1):
            await asyncio.sleep(_backoff_seconds(last_headers, attempt, retry))

    if last_exc:
        raise last_exc
    return last_status, last_body


def _log_retry_status(status: int, attempt: int, attempts: int) -> None:
    # Debug-level: these logs fire from within the CLI's live-status window
    action = "retrying" if attempt < attempts else "no retries left"
    if status == 429:
        logger.debug("Rate limited (429), attempt %d/%d failed; %s", attempt, attempts, action)
        return
    logger.debug(
        "Server error (HTTP %d), attempt %d/%d failed; %s",
        status,
        attempt,
        attempts,
        action,
    )


def _log_retry_exception(exc: Exception, attempt: int, attempts: int) -> None:
    action = "retrying" if attempt < attempts else "no retries left"
    logger.debug(
        "Request failed (%s), attempt %d/%d failed; %s",
        exc,
        attempt,
        attempts,
        action,
    )


async def http_get_with_retry(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    retry: HttpRetryConfig = DEFAULT_HTTP_RETRY,
    max_bytes: int | None = None,
) -> tuple[int, str]:
    """GET with retries on 429/5xx and transient transport failures."""
    response = await http_get_text_with_retry(
        url,
        headers=headers,
        timeout=timeout,
        retry=retry,
        max_bytes=max_bytes,
    )
    return response.status, response.body


async def http_get_text_with_retry(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    retry: HttpRetryConfig = DEFAULT_HTTP_RETRY,
    max_bytes: int | None = None,
    allow_redirects: bool = True,
) -> HttpTextResponse:
    """GET text with retries on 429/5xx and transient transport failures."""
    return await _request_text_with_retry(
        method="GET",
        url=url,
        headers=headers,
        timeout=timeout,
        retry=retry,
        max_bytes=max_bytes,
        allow_redirects=allow_redirects,
    )


async def http_post_json_with_retry(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: object | None = None,
    timeout: int = 30,
    retry: HttpRetryConfig = DEFAULT_HTTP_RETRY,
) -> tuple[int, str]:
    """POST JSON with retries on 429/5xx and transient transport failures."""
    return await _request_with_retry(
        method="POST",
        url=url,
        headers=headers,
        timeout=timeout,
        json_body=json_body,
        retry=retry,
    )


async def http_get_bytes_with_retry(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    retry: HttpRetryConfig = DEFAULT_HTTP_RETRY,
    max_bytes: int | None = None,
    allow_redirects: bool = False,
) -> tuple[int, bytes]:
    """GET bytes with retries on 429/5xx and transient transport failures."""
    return await _request_bytes_with_retry(
        method="GET",
        url=url,
        headers=headers,
        timeout=timeout,
        retry=retry,
        max_bytes=max_bytes,
        allow_redirects=allow_redirects,
    )


async def http_head_with_retry(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    retry: HttpRetryConfig = DEFAULT_HTTP_RETRY,
    allow_redirects: bool = True,
) -> tuple[int, Mapping[str, str]]:
    """HEAD with retries on 429/5xx and transient transport failures."""
    response = await _request_text_with_retry(
        method="HEAD",
        url=url,
        headers=headers,
        timeout=timeout,
        retry=retry,
        max_bytes=None,
        allow_redirects=allow_redirects,
    )
    return response.status, response.headers
