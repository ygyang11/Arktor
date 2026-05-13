"""HTTP retry helpers for async requests."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HttpRetryConfig:
    """Retry policy for HTTP requests."""

    max_attempts: int = 3
    base_delay: float = 1.0


@dataclass(frozen=True)
class HttpTextResponse:
    """HTTP text response payload with headers preserved."""

    status: int
    headers: Mapping[str, str]
    body: str


DEFAULT_HTTP_RETRY = HttpRetryConfig()


def _is_retryable_status(status: int) -> bool:
    return status == 429 or 500 <= status < 600


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
    attempts = max(1, retry.max_attempts)

    for attempt in range(attempts):
        try:
            async with aiohttp.ClientSession() as session:
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
                    _log_retry_status(resp.status, attempt + 1, attempts)
        except (aiohttp.ClientError, TimeoutError) as exc:
            last_exc = exc
            _log_retry_exception(exc, attempt + 1, attempts)

        if attempt < (attempts - 1):
            await asyncio.sleep(retry.base_delay * (2**attempt))

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
) -> HttpTextResponse:
    import aiohttp  # noqa: PLC0415

    hdrs = dict(headers or {})
    last_exc: Exception | None = None
    last_status = 429
    last_headers: dict[str, str] = {}
    last_body = "Rate limit exceeded after retries"
    attempts = max(1, retry.max_attempts)

    for attempt in range(attempts):
        try:
            async with aiohttp.ClientSession() as session:
                request_kwargs: dict[str, object] = {
                    "headers": hdrs,
                    "timeout": aiohttp.ClientTimeout(total=timeout),
                }
                if json_body is not None:
                    request_kwargs["json"] = json_body
                async with session.request(method, url, **request_kwargs) as resp:  # type: ignore[arg-type]
                    body = await resp.text()
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
            await asyncio.sleep(retry.base_delay * (2**attempt))

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
) -> tuple[int, bytes]:
    import aiohttp  # noqa: PLC0415

    hdrs = dict(headers or {})
    last_exc: Exception | None = None
    last_status = 429
    last_body = b""
    attempts = max(1, retry.max_attempts)

    for attempt in range(attempts):
        try:
            async with aiohttp.ClientSession() as session:
                request_kwargs: dict[str, object] = {
                    "headers": hdrs,
                    "timeout": aiohttp.ClientTimeout(total=timeout),
                }
                if json_body is not None:
                    request_kwargs["json"] = json_body
                async with session.request(method, url, **request_kwargs) as resp:  # type: ignore[arg-type]
                    body = await resp.read()
                    if not _is_retryable_status(resp.status):
                        return resp.status, body
                    last_status = resp.status
                    last_body = body
                    _log_retry_status(resp.status, attempt + 1, attempts)
        except (aiohttp.ClientError, TimeoutError) as exc:
            last_exc = exc
            _log_retry_exception(exc, attempt + 1, attempts)

        if attempt < (attempts - 1):
            await asyncio.sleep(retry.base_delay * (2**attempt))

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
) -> tuple[int, str]:
    """GET with retries on 429/5xx and transient transport failures."""
    response = await http_get_text_with_retry(
        url,
        headers=headers,
        timeout=timeout,
        retry=retry,
    )
    return response.status, response.body


async def http_get_text_with_retry(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    retry: HttpRetryConfig = DEFAULT_HTTP_RETRY,
) -> HttpTextResponse:
    """GET text with retries on 429/5xx and transient transport failures."""
    return await _request_text_with_retry(
        method="GET",
        url=url,
        headers=headers,
        timeout=timeout,
        retry=retry,
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
) -> tuple[int, bytes]:
    """GET bytes with retries on 429/5xx and transient transport failures."""
    return await _request_bytes_with_retry(
        method="GET",
        url=url,
        headers=headers,
        timeout=timeout,
        retry=retry,
    )
