"""Backend ABC + dataclasses + shared HTTP plumbing."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import aiohttp
from yarl import URL

from agent_app.tools.document_parser.errors import (
    DocumentBackendError,
    DocumentErrorClass,
)
from agent_harness.utils.http_retry import (
    HttpRetryConfig,
    http_get_bytes_with_retry,
    http_get_with_retry,
    http_post_json_with_retry,
)

_NO_CT = frozenset({"Content-Type"})


@dataclass
class DocumentBackendOutcome:
    backend_name: str
    backend_model: str
    page_count: int | None
    image_count: int


@dataclass(frozen=True)
class MinerUOptions:
    model: Literal["mineru-vlm", "mineru-pipeline"] = "mineru-vlm"
    enable_formula: bool = True
    enable_table: bool = True
    language: str = "ch"
    extra_formats: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PaddleOptions:
    model: Literal["PaddleOCR-VL-1.5", "PaddleOCR-VL"] = "PaddleOCR-VL-1.5"
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_chart_recognition: bool = False


class DocumentBackend(ABC):
    name: str
    model: str
    max_mb_local: float | None
    max_mb_url: float | None
    max_pages: int | None
    needs_key: str | None

    @abstractmethod
    async def parse_local(
        self, session: aiohttp.ClientSession, file_path: Path, dest_dir: Path,
    ) -> DocumentBackendOutcome: ...

    @abstractmethod
    async def parse_url(
        self, url: str, dest_dir: Path,
    ) -> DocumentBackendOutcome: ...


@dataclass(frozen=True)
class BackendHTTPContext:
    classify: Callable[[Any], DocumentErrorClass]
    retry: HttpRetryConfig
    request_timeout_s: int
    upload_timeout_s: int
    download_timeout_s: int
    backend_label: str


def decode_envelope(
    status: int, body: str, *, ctx: BackendHTTPContext,
) -> dict[str, Any]:
    if status != 200:
        raise DocumentBackendError(
            ctx.classify(status), status,
            f"HTTP {status} from {ctx.backend_label}: {body[:200]}",
        )
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as e:
        raise DocumentBackendError(
            DocumentErrorClass.UNKNOWN, status,
            f"non-JSON response from {ctx.backend_label}: {body[:200]}",
        ) from e
    if not isinstance(parsed, dict):
        raise DocumentBackendError(
            DocumentErrorClass.UNKNOWN, status,
            f"unexpected non-object response from {ctx.backend_label}",
        )
    return parsed


def raise_for_code(result: dict[str, Any], *, ctx: BackendHTTPContext) -> None:
    code = result.get("code")
    if code == 0:
        return
    msg_obj = result.get("msg") or result.get("message")
    msg = msg_obj if isinstance(msg_obj, str) else "backend returned error code without details"
    raise DocumentBackendError(ctx.classify(code), code, msg)


def require_data(
    result: dict[str, Any], *, ctx: BackendHTTPContext,
) -> dict[str, Any]:
    data = result.get("data")
    if not isinstance(data, dict):
        raise DocumentBackendError(
            DocumentErrorClass.UNKNOWN, None,
            f"{ctx.backend_label} response missing or malformed 'data' object",
        )
    return data


def require_str(
    data: dict[str, Any], key: str, *, ctx: BackendHTTPContext,
) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise DocumentBackendError(
            DocumentErrorClass.UNKNOWN, None,
            f"{ctx.backend_label} response missing field {key!r}",
        )
    return value


async def post_envelope(
    ctx: BackendHTTPContext, url: str, *,
    headers: dict[str, str], json_body: object,
    timeout_s: int | None = None,
) -> dict[str, Any]:
    try:
        status, body = await http_post_json_with_retry(
            url, headers=headers, json_body=json_body,
            timeout=timeout_s or ctx.request_timeout_s, retry=ctx.retry,
        )
    except aiohttp.ClientError as e:
        raise DocumentBackendError(
            DocumentErrorClass.BACKEND_TRANSIENT_ERROR, None,
            f"network error during {ctx.backend_label} POST: {type(e).__name__}",
        ) from e
    except TimeoutError as e:
        raise DocumentBackendError(
            DocumentErrorClass.TIMEOUT, None,
            f"request timeout during {ctx.backend_label} POST",
        ) from e
    return decode_envelope(status, body, ctx=ctx)


async def get_envelope(
    ctx: BackendHTTPContext, url: str, *,
    headers: dict[str, str] | None = None, timeout_s: int | None = None,
) -> dict[str, Any]:
    try:
        status, body = await http_get_with_retry(
            url, headers=headers,
            timeout=timeout_s or ctx.request_timeout_s, retry=ctx.retry,
        )
    except aiohttp.ClientError as e:
        raise DocumentBackendError(
            DocumentErrorClass.BACKEND_TRANSIENT_ERROR, None,
            f"network error during {ctx.backend_label} GET: {type(e).__name__}",
        ) from e
    except TimeoutError as e:
        raise DocumentBackendError(
            DocumentErrorClass.TIMEOUT, None,
            f"request timeout during {ctx.backend_label} GET",
        ) from e
    return decode_envelope(status, body, ctx=ctx)


async def get_text(
    ctx: BackendHTTPContext, url: str, *, timeout_s: int,
) -> str:
    try:
        status, body = await http_get_with_retry(
            url, timeout=timeout_s, retry=ctx.retry,
        )
    except aiohttp.ClientError as e:
        raise DocumentBackendError(
            DocumentErrorClass.BACKEND_TRANSIENT_ERROR, None,
            f"network error downloading {ctx.backend_label} text: {type(e).__name__}",
        ) from e
    except TimeoutError as e:
        raise DocumentBackendError(
            DocumentErrorClass.TIMEOUT, None,
            f"request timeout downloading {ctx.backend_label} text",
        ) from e
    if status != 200:
        cls = (
            DocumentErrorClass.BACKEND_TRANSIENT_ERROR if status >= 500
            else DocumentErrorClass.BACKEND_READ_FAILED
        )
        raise DocumentBackendError(
            cls, status,
            f"failed to download {ctx.backend_label} text resource (HTTP {status})",
        )
    return body


async def get_bytes(
    ctx: BackendHTTPContext, url: str, *, timeout_s: int,
) -> bytes:
    try:
        status, body = await http_get_bytes_with_retry(
            url, timeout=timeout_s, retry=ctx.retry,
        )
    except aiohttp.ClientError as e:
        raise DocumentBackendError(
            DocumentErrorClass.BACKEND_TRANSIENT_ERROR, None,
            f"network error downloading {ctx.backend_label} archive: {type(e).__name__}",
        ) from e
    except TimeoutError as e:
        raise DocumentBackendError(
            DocumentErrorClass.TIMEOUT, None,
            f"request timeout downloading {ctx.backend_label} archive",
        ) from e
    if status != 200:
        cls = (
            DocumentErrorClass.BACKEND_TRANSIENT_ERROR if status >= 500
            else DocumentErrorClass.BACKEND_READ_FAILED
        )
        raise DocumentBackendError(
            cls, status,
            f"failed to download {ctx.backend_label} archive (HTTP {status})",
        )
    return body


async def post_multipart(
    ctx: BackendHTTPContext, session: aiohttp.ClientSession,
    url: str, *, headers: dict[str, str], form: aiohttp.FormData,
    timeout_s: int | None = None,
) -> dict[str, Any]:
    try:
        async with session.post(
            url, headers=headers, data=form,
            timeout=aiohttp.ClientTimeout(total=timeout_s or ctx.upload_timeout_s),
        ) as r:
            status = r.status
            body = await r.text()
    except aiohttp.ClientError as e:
        raise DocumentBackendError(
            DocumentErrorClass.BACKEND_TRANSIENT_ERROR, None,
            f"network error during {ctx.backend_label} multipart POST: {type(e).__name__}",
        ) from e
    except TimeoutError as e:
        raise DocumentBackendError(
            DocumentErrorClass.TIMEOUT, None,
            f"request timeout during {ctx.backend_label} multipart POST",
        ) from e
    return decode_envelope(status, body, ctx=ctx)


async def put_upload(
    ctx: BackendHTTPContext, session: aiohttp.ClientSession,
    upload_url: str, data: bytes,
) -> None:
    try:
        async with session.put(
            URL(upload_url, encoded=True), data=data,
            skip_auto_headers=_NO_CT,
            timeout=aiohttp.ClientTimeout(total=ctx.upload_timeout_s),
        ) as r:
            status = r.status
    except aiohttp.ClientError as e:
        raise DocumentBackendError(
            DocumentErrorClass.BACKEND_TRANSIENT_ERROR, None,
            f"network error during file upload: {type(e).__name__}",
        ) from e
    except TimeoutError as e:
        raise DocumentBackendError(
            DocumentErrorClass.TIMEOUT, None,
            "request timeout during file upload",
        ) from e
    if status not in (200, 201):
        cls = (
            DocumentErrorClass.BACKEND_TRANSIENT_ERROR if status >= 500
            else DocumentErrorClass.BACKEND_READ_FAILED
        )
        raise DocumentBackendError(
            cls, status,
            f"failed to upload file to backend (HTTP {status})",
        )
