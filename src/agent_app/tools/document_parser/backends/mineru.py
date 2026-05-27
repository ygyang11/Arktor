"""MinerU v4 batch + v1 lightweight backends."""
from __future__ import annotations

import asyncio
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from agent_app.tools.document_parser.backends.base import (
    BackendHTTPContext,
    DocumentBackend,
    DocumentBackendOutcome,
    MinerUOptions,
    get_bytes,
    get_envelope,
    get_text,
    post_envelope,
    put_upload,
    raise_for_code,
    require_data,
    require_str,
)
from agent_app.tools.document_parser.errors import (
    DocumentBackendError,
    DocumentErrorClass,
    classify_mineru,
)
from agent_harness.utils.http_retry import HttpRetryConfig

MINERU_BASE = "https://mineru.net"


@dataclass(frozen=True)
class _MinerUConfig:
    poll_interval: float = 5.0
    max_poll_attempts: int = 120
    request_max_attempts: int = 3
    request_base_delay: float = 1.0
    request_timeout_s: int = 30
    upload_timeout_s: int = 120
    download_timeout_s: int = 600


_CFG = _MinerUConfig()

_HTTP_CTX = BackendHTTPContext(
    classify=classify_mineru,
    retry=HttpRetryConfig(
        max_attempts=_CFG.request_max_attempts,
        base_delay=_CFG.request_base_delay,
    ),
    request_timeout_s=_CFG.request_timeout_s,
    upload_timeout_s=_CFG.upload_timeout_s,
    download_timeout_s=_CFG.download_timeout_s,
    backend_label="MinerU",
)


class MinerUV4Backend(DocumentBackend):
    max_mb_local: float | None = 200.0
    max_mb_url: float | None = 200.0
    max_pages: int | None = 200
    needs_key: str | None = "mineru"

    def __init__(
        self, api_key: str, *, opts: MinerUOptions = MinerUOptions(),
    ) -> None:
        self._key = api_key
        self._opts = opts
        self.name = opts.model
        self.model = opts.model.removeprefix("mineru-")

    async def parse_local(
        self, session: aiohttp.ClientSession, file_path: Path, dest_dir: Path,
    ) -> DocumentBackendOutcome:
        batch_id, urls = await self._submit_file_urls_batch(
            [{"name": file_path.name, "is_ocr": False}],
        )
        await put_upload(_HTTP_CTX, session, urls[0], file_path.read_bytes())
        return await self._collect(batch_id, dest_dir)

    async def parse_url(
        self, url: str, dest_dir: Path,
    ) -> DocumentBackendOutcome:
        result = await post_envelope(
            _HTTP_CTX,
            f"{MINERU_BASE}/api/v4/extract/task/batch",
            headers=self._json_headers(),
            json_body=self._batch_body([{"url": url, "is_ocr": False}]),
        )
        raise_for_code(result, ctx=_HTTP_CTX)
        data = require_data(result, ctx=_HTTP_CTX)
        batch_id = require_str(data, "batch_id", ctx=_HTTP_CTX)
        return await self._collect(batch_id, dest_dir)

    async def _submit_file_urls_batch(
        self, files: list[dict[str, Any]],
    ) -> tuple[str, list[str]]:
        result = await post_envelope(
            _HTTP_CTX,
            f"{MINERU_BASE}/api/v4/file-urls/batch",
            headers=self._json_headers(),
            json_body=self._batch_body(files),
        )
        raise_for_code(result, ctx=_HTTP_CTX)
        data = require_data(result, ctx=_HTTP_CTX)
        batch_id = require_str(data, "batch_id", ctx=_HTTP_CTX)
        file_urls_raw = data.get("file_urls")
        if not isinstance(file_urls_raw, list) or not file_urls_raw:
            raise DocumentBackendError(
                DocumentErrorClass.UNKNOWN, None,
                "MinerU returned no upload URLs",
            )
        return batch_id, [str(u) for u in file_urls_raw]

    def _batch_body(self, files: list[dict[str, Any]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model_version": self.model,
            "enable_formula": self._opts.enable_formula,
            "enable_table": self._opts.enable_table,
            "language": self._opts.language,
            "files": files,
        }
        if self._opts.extra_formats:
            body["extra_formats"] = list(self._opts.extra_formats)
        return body

    async def _collect(
        self, batch_id: str, dest_dir: Path,
    ) -> DocumentBackendOutcome:
        item = await self._poll_batch(batch_id)
        state = item.get("state")
        if state == "failed":
            err = item.get("err_msg")
            raise DocumentBackendError(
                DocumentErrorClass.BACKEND_READ_FAILED, None,
                err if isinstance(err, str) and err else "backend task failed without details",
            )
        zip_url_raw = item.get("full_zip_url")
        if not isinstance(zip_url_raw, str) or not zip_url_raw:
            raise DocumentBackendError(
                DocumentErrorClass.UNKNOWN, None,
                "backend reported done but no result archive url",
            )
        return await self._fetch_and_unpack(zip_url_raw, dest_dir)

    async def _poll_batch(self, batch_id: str) -> dict[str, Any]:
        url = f"{MINERU_BASE}/api/v4/extract-results/batch/{batch_id}"
        headers = {"Authorization": f"Bearer {self._key}"}
        for _ in range(_CFG.max_poll_attempts):
            await asyncio.sleep(_CFG.poll_interval)
            result = await get_envelope(_HTTP_CTX, url, headers=headers)
            raise_for_code(result, ctx=_HTTP_CTX)
            data = require_data(result, ctx=_HTTP_CTX)
            items_raw = data.get("extract_result")
            if not isinstance(items_raw, list):
                continue
            items = [it for it in items_raw if isinstance(it, dict)]
            if items and all(it.get("state") in ("done", "failed") for it in items):
                return items[0]
        raise DocumentBackendError(
            DocumentErrorClass.TIMEOUT, None,
            f"polling timeout: parse did not complete within "
            f"{int(_CFG.max_poll_attempts * _CFG.poll_interval)}s",
        )

    async def _fetch_and_unpack(
        self, zip_url: str, dest_dir: Path,
    ) -> DocumentBackendOutcome:
        blob = await get_bytes(_HTTP_CTX, zip_url, timeout_s=_CFG.download_timeout_s)

        image_count = 0
        page_count: int | None = None

        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    name = info.filename
                    if ".." in Path(name).parts or Path(name).is_absolute():
                        continue
                    if name.endswith("_origin.pdf"):
                        continue
                    data = zf.read(info)
                    if name == "full.md":
                        (dest_dir / "content.md").write_bytes(data)
                    elif name == "layout.json":
                        (dest_dir / "layout.json").write_bytes(data)
                        try:
                            parsed_layout = json.loads(data)
                        except json.JSONDecodeError:
                            parsed_layout = None
                        if isinstance(parsed_layout, dict):
                            pdf_info = parsed_layout.get("pdf_info")
                            if isinstance(pdf_info, list):
                                page_count = len(pdf_info)
                    elif name.startswith("images/"):
                        target = dest_dir / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(data)
                        image_count += 1
        except OSError as e:
            raise DocumentBackendError(
                DocumentErrorClass.IO_ERROR, None,
                f"I/O error materializing MinerU-vlm artifacts to {dest_dir}: {e}",
            ) from e

        return DocumentBackendOutcome(
            backend_name=self.name, backend_model=self.model,
            page_count=page_count, image_count=image_count,
        )

    def _json_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }


class MinerULightweightBackend(DocumentBackend):
    name = "mineru-lightweight"
    model = "lightweight"
    max_mb_local: float | None = 10.0
    max_mb_url: float | None = 10.0
    max_pages: int | None = 20
    needs_key: str | None = None

    async def parse_local(
        self, session: aiohttp.ClientSession, file_path: Path, dest_dir: Path,
    ) -> DocumentBackendOutcome:
        result = await post_envelope(
            _HTTP_CTX,
            f"{MINERU_BASE}/api/v1/agent/parse/file",
            headers={}, json_body={"file_name": file_path.name},
        )
        raise_for_code(result, ctx=_HTTP_CTX)
        data = require_data(result, ctx=_HTTP_CTX)
        file_url = require_str(data, "file_url", ctx=_HTTP_CTX)
        task_id = require_str(data, "task_id", ctx=_HTTP_CTX)
        await put_upload(_HTTP_CTX, session, file_url, file_path.read_bytes())
        return await self._poll_and_materialize(task_id, dest_dir)

    async def parse_url(
        self, url: str, dest_dir: Path,
    ) -> DocumentBackendOutcome:
        result = await post_envelope(
            _HTTP_CTX,
            f"{MINERU_BASE}/api/v1/agent/parse/url",
            headers={}, json_body={"url": url},
        )
        raise_for_code(result, ctx=_HTTP_CTX)
        data = require_data(result, ctx=_HTTP_CTX)
        task_id = require_str(data, "task_id", ctx=_HTTP_CTX)
        return await self._poll_and_materialize(task_id, dest_dir)

    async def _poll_and_materialize(
        self, task_id: str, dest_dir: Path,
    ) -> DocumentBackendOutcome:
        for _ in range(_CFG.max_poll_attempts):
            await asyncio.sleep(_CFG.poll_interval)
            result = await get_envelope(
                _HTTP_CTX, f"{MINERU_BASE}/api/v1/agent/parse/{task_id}",
            )
            raise_for_code(result, ctx=_HTTP_CTX)
            data = require_data(result, ctx=_HTTP_CTX)
            state = data.get("state")
            if state == "failed":
                err = data.get("err_msg")
                raise DocumentBackendError(
                    DocumentErrorClass.BACKEND_READ_FAILED, None,
                    err if isinstance(err, str) and err else "backend task failed without details",
                )
            if state == "done":
                md_url_raw = data.get("markdown_url")
                if not isinstance(md_url_raw, str) or not md_url_raw:
                    raise DocumentBackendError(
                        DocumentErrorClass.UNKNOWN, None,
                        "backend reported done but no markdown url",
                    )
                md_text = await get_text(
                    _HTTP_CTX, md_url_raw, timeout_s=_CFG.download_timeout_s,
                )
                try:
                    (dest_dir / "content.md").write_text(md_text, encoding="utf-8")
                    (dest_dir / "layout.json").write_text(
                        json.dumps({"pages": []}, ensure_ascii=False),
                        encoding="utf-8",
                    )
                except OSError as e:
                    raise DocumentBackendError(
                        DocumentErrorClass.IO_ERROR, None,
                        f"I/O error materializing MinerU-lightweight artifacts to {dest_dir}: {e}",
                    ) from e
                return DocumentBackendOutcome(
                    backend_name=self.name, backend_model=self.model,
                    page_count=None, image_count=0,
                )
        raise DocumentBackendError(
            DocumentErrorClass.TIMEOUT, None,
            f"polling timeout: parse did not complete within "
            f"{int(_CFG.max_poll_attempts * _CFG.poll_interval)}s",
        )
