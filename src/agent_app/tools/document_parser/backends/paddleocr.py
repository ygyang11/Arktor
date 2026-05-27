"""PaddleOCR portal backend + JSONL materializer."""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from agent_app.tools.document_parser.backends.base import (
    BackendHTTPContext,
    DocumentBackend,
    DocumentBackendOutcome,
    PaddleOptions,
    get_bytes,
    get_envelope,
    get_text,
    post_envelope,
    post_multipart,
    raise_for_code,
    require_data,
    require_str,
)
from agent_app.tools.document_parser.errors import (
    DocumentBackendError,
    DocumentErrorClass,
    classify_paddle,
)
from agent_harness.utils.http_retry import HttpRetryConfig

logger = logging.getLogger(__name__)

PORTAL_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"

_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]")


@dataclass(frozen=True)
class _PaddleConfig:
    poll_interval: float = 5.0
    max_poll_attempts: int = 120
    request_max_attempts: int = 3
    request_base_delay: float = 1.0
    request_timeout_s: int = 20
    upload_timeout_s: int = 120
    download_timeout_s: int = 600
    image_fetch_concurrency: int = 8
    image_fetch_timeout_s: int = 30


_CFG = _PaddleConfig()

_HTTP_CTX = BackendHTTPContext(
    classify=classify_paddle,
    retry=HttpRetryConfig(
        max_attempts=_CFG.request_max_attempts,
        base_delay=_CFG.request_base_delay,
    ),
    request_timeout_s=_CFG.request_timeout_s,
    upload_timeout_s=_CFG.upload_timeout_s,
    download_timeout_s=_CFG.download_timeout_s,
    backend_label="PaddleOCR",
)


class PaddleBackend(DocumentBackend):
    needs_key: str | None = "paddleocr"
    max_mb_local: float | None = 50.0
    max_mb_url: float | None = 200.0

    _MAX_PAGES_BY_MODEL: dict[str, int] = {
        "PaddleOCR-VL-1.5": 100,
        "PaddleOCR-VL": 10,
    }

    def __init__(
        self, api_key: str, *,
        opts: PaddleOptions = PaddleOptions(),
    ) -> None:
        self._key = api_key
        self._opts = opts
        self.model = opts.model
        self.name = opts.model.lower()
        self.max_pages: int | None = self._MAX_PAGES_BY_MODEL.get(opts.model, 10)

    async def parse_local(
        self, session: aiohttp.ClientSession, file_path: Path, dest_dir: Path,
    ) -> DocumentBackendOutcome:
        job_id = await self._submit_local(session, file_path)
        return await self._collect(job_id, dest_dir)

    async def parse_url(
        self, url: str, dest_dir: Path,
    ) -> DocumentBackendOutcome:
        job_id = await self._submit_url(url)
        return await self._collect(job_id, dest_dir)

    async def _submit_local(
        self, session: aiohttp.ClientSession, file_path: Path,
    ) -> str:
        mime, _ = mimetypes.guess_type(str(file_path))
        form = aiohttp.FormData()
        form.add_field("model", self.model)
        form.add_field("optionalPayload", json.dumps(self._opt_payload()))
        form.add_field(
            "file", file_path.read_bytes(),
            filename=file_path.name,
            content_type=mime,
        )
        result = await post_multipart(
            _HTTP_CTX, session, PORTAL_URL,
            headers=self._auth(), form=form,
        )
        raise_for_code(result, ctx=_HTTP_CTX)
        return require_str(
            require_data(result, ctx=_HTTP_CTX), "jobId", ctx=_HTTP_CTX,
        )

    async def _submit_url(self, url: str) -> str:
        result = await post_envelope(
            _HTTP_CTX, PORTAL_URL,
            headers={**self._auth(), "Content-Type": "application/json"},
            json_body={
                "fileUrl": url, "model": self.model,
                "optionalPayload": self._opt_payload(),
            },
        )
        raise_for_code(result, ctx=_HTTP_CTX)
        return require_str(
            require_data(result, ctx=_HTTP_CTX), "jobId", ctx=_HTTP_CTX,
        )

    async def _collect(
        self, job_id: str, dest_dir: Path,
    ) -> DocumentBackendOutcome:
        json_url = await self._poll(job_id)
        jsonl = await get_text(_HTTP_CTX, json_url, timeout_s=_CFG.download_timeout_s)
        return await materialize_jsonl(jsonl, dest_dir, self.name, self.model)

    async def _poll(self, job_id: str) -> str:
        url = f"{PORTAL_URL}/{job_id}"
        headers = self._auth()
        for _ in range(_CFG.max_poll_attempts):
            await asyncio.sleep(_CFG.poll_interval)
            result = await get_envelope(_HTTP_CTX, url, headers=headers)
            raise_for_code(result, ctx=_HTTP_CTX)
            data = result.get("data")
            if not isinstance(data, dict):
                continue
            state = data.get("state")
            if state == "failed":
                err = data.get("errorMsg")
                raise DocumentBackendError(
                    DocumentErrorClass.BACKEND_READ_FAILED, None,
                    err if isinstance(err, str) and err else "backend task failed without details",
                )
            if state == "done":
                result_url = data.get("resultUrl")
                json_url = ""
                if isinstance(result_url, dict):
                    raw = result_url.get("jsonUrl")
                    if isinstance(raw, str):
                        json_url = raw
                if not json_url:
                    raise DocumentBackendError(
                        DocumentErrorClass.UNKNOWN, None,
                        "backend reported done but no result url",
                    )
                return json_url
        raise DocumentBackendError(
            DocumentErrorClass.TIMEOUT, None,
            f"polling timeout: parse did not complete within "
            f"{int(_CFG.max_poll_attempts * _CFG.poll_interval)}s",
        )

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"bearer {self._key}"}

    def _opt_payload(self) -> dict[str, bool]:
        return {
            "useDocOrientationClassify": self._opts.use_doc_orientation_classify,
            "useDocUnwarping": self._opts.use_doc_unwarping,
            "useChartRecognition": self._opts.use_chart_recognition,
        }


async def materialize_jsonl(
    jsonl: str, dest_dir: Path, backend_name: str, backend_model: str,
) -> DocumentBackendOutcome:
    raw_md_parts: list[str] = []
    layout_pages: list[dict[str, Any]] = []
    # orig_key (referenced inside markdown text) → safe local filename
    image_targets: dict[str, str] = {}
    # safe filename → raw value from JSONL (URL or base64 string)
    image_sources: dict[str, str] = {}

    for line in jsonl.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            page_obj = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("paddleocr: skipping unparseable jsonl line")
            continue
        if not isinstance(page_obj, dict):
            continue
        result_obj = page_obj.get("result")
        if not isinstance(result_obj, dict):
            continue
        lp_list = result_obj.get("layoutParsingResults")
        if not isinstance(lp_list, list):
            continue
        for lp in lp_list:
            if not isinstance(lp, dict):
                continue
            md_block = lp.get("markdown")
            md = md_block if isinstance(md_block, dict) else {}
            md_text_raw = md.get("text")
            md_text = md_text_raw if isinstance(md_text_raw, str) else ""
            images_block = md.get("images")
            if isinstance(images_block, dict):
                for orig_key, value in images_block.items():
                    if not isinstance(orig_key, str) or not isinstance(value, str):
                        continue
                    if orig_key in image_targets:
                        continue
                    safe = _UNSAFE_FILENAME.sub("_", Path(orig_key).name)[:128] or "img"
                    image_targets[orig_key] = safe
                    image_sources[safe] = value
            if md_text.strip():
                raw_md_parts.append(md_text)
            pruned = lp.get("prunedResult")
            if isinstance(pruned, dict):
                layout_pages.append(pruned)

    # Fetch / decode image bytes concurrently before any disk writes
    resolved_images = await _resolve_images(image_sources)
    successful_safes = set(resolved_images.keys())
    rewrites = [
        (orig_key, f"images/{safe}")
        for orig_key, safe in image_targets.items()
        if safe in successful_safes
    ]
    md_parts = [_apply_rewrites(t, rewrites).strip() for t in raw_md_parts]

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        if resolved_images:
            images_dir = dest_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            for safe, data in resolved_images.items():
                (images_dir / safe).write_bytes(data)

        content_md = "\n\n".join(md_parts)
        (dest_dir / "content.md").write_text(content_md, encoding="utf-8")
        (dest_dir / "layout.json").write_text(
            json.dumps({"pages": layout_pages}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        raise DocumentBackendError(
            DocumentErrorClass.IO_ERROR, None,
            f"I/O error materializing PaddleOCR artifacts to {dest_dir}: {e}",
        ) from e

    page_count = len(layout_pages) if layout_pages else None
    return DocumentBackendOutcome(
        backend_name=backend_name, backend_model=backend_model,
        page_count=page_count, image_count=len(resolved_images),
    )


def _apply_rewrites(text: str, rewrites: list[tuple[str, str]]) -> str:
    # Apply longest-key first to avoid one orig_key being a substring of
    # another rewriting partially through it. Stable order otherwise.
    for orig, repl in sorted(rewrites, key=lambda kv: -len(kv[0])):
        text = text.replace(orig, repl)
    return text


async def _resolve_images(sources: dict[str, str]) -> dict[str, bytes]:
    """Resolve `safe_filename → image bytes`, fetching URLs and decoding
    base64 in parallel. Failures (network / decode) drop that image from
    the result map; the document still materializes with the remaining
    successful images.

    PaddleOCR's portal returns CDN URLs in `markdown.images.*` for the
    chart / image / table crops it extracts. Earlier doc claimed base64
    payloads; in practice the values are signed BCEbos URLs. base64 is
    kept as a fallback in case future portal revisions revert.
    """
    if not sources:
        return {}

    sem = asyncio.Semaphore(_CFG.image_fetch_concurrency)

    async def _one(safe: str, value: str) -> tuple[str, bytes | None]:
        async with sem:
            return safe, await _resolve_one_image(value)

    results = await asyncio.gather(*(_one(s, v) for s, v in sources.items()))
    return {s: data for s, data in results if data is not None}


async def _resolve_one_image(value: str) -> bytes | None:
    if value.startswith(("http://", "https://")):
        try:
            return await get_bytes(
                _HTTP_CTX, value, timeout_s=_CFG.image_fetch_timeout_s,
            )
        except DocumentBackendError as e:
            logger.debug("paddleocr: image fetch failed (%s): %s", value[:80], e)
            return None
    try:
        return base64.b64decode(value, validate=True)
    except (TypeError, ValueError, binascii.Error):
        logger.debug("paddleocr: image value not URL nor base64; skipped")
        return None
