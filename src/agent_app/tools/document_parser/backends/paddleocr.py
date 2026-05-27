"""PaddleOCR portal backend + JSONL materializer."""
from __future__ import annotations

import asyncio
import base64
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
        return materialize_jsonl(jsonl, dest_dir, self.name, self.model)

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


def materialize_jsonl(
    jsonl: str, dest_dir: Path, backend_name: str, backend_model: str,
) -> DocumentBackendOutcome:
    md_parts: list[str] = []
    layout_pages: list[dict[str, Any]] = []
    images_written: dict[str, str] = {}
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)

        images_dir = dest_dir / "images"
        images_dir_created = False

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
                    for orig_key, b64 in images_block.items():
                        if not isinstance(orig_key, str) or not isinstance(b64, str):
                            continue
                        if orig_key in images_written:
                            continue
                        safe = _UNSAFE_FILENAME.sub("_", Path(orig_key).name)[:128] or "img"
                        try:
                            decoded = base64.b64decode(b64)
                        except (TypeError, ValueError):
                            continue
                        if not images_dir_created:
                            images_dir.mkdir(parents=True, exist_ok=True)
                            images_dir_created = True
                        (images_dir / safe).write_bytes(decoded)
                        images_written[orig_key] = safe
                for orig_key, safe in images_written.items():
                    md_text = md_text.replace(orig_key, f"images/{safe}")
                if md_text.strip():
                    md_parts.append(md_text.strip())
                pruned = lp.get("prunedResult")
                if isinstance(pruned, dict):
                    layout_pages.append(pruned)

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
        page_count=page_count, image_count=len(images_written),
    )
