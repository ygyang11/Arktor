"""Tiered document parsing (PDF / image) with on-disk artifacts."""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import aiohttp

from agent_app.tools.document_parser.backends import (
    DocumentBackend,
    MinerULightweightBackend,
    MinerUOptions,
    MinerUV4Backend,
    PaddleBackend,
    PaddleOptions,
)
from agent_app.tools.document_parser.errors import (
    DocumentBackendError,
    DocumentErrorClass,
    NoViableDocumentBackend,
)
from agent_app.tools.document_parser.pipeline import (
    DownloadFn,
    PipelineSuccess,
    run_pipeline,
)
from agent_app.tools.document_parser.storage import (
    _MIME_TO_SUFFIX,
    _USER_AGENT,
    TargetInspection,
    already_parsed,
    format_cached,
    format_no_viable,
    format_success,
    hash_source,
    inspect_target,
    make_slug,
    session_documents_root,
    write_manifest,
)
from agent_harness.core.config import (
    DocumentParserConfig,
    resolve_document_parser_config,
)
from agent_harness.core.errors import HttpResponseTooLargeError
from agent_harness.tool.base import BaseTool, ToolSchema
from agent_harness.utils.http_retry import HttpRetryConfig, http_get_bytes_with_retry
from agent_harness.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)

DOCUMENT_PARSER_DESCRIPTION = (
    "Extract structured content from a PDF or image document. "
    "Persists artifacts on disk and returns paths.\n\n"
    "## Usage\n"
    "- `target`: URL (http/https) or local file path\n"
    "- Supported formats: PDF, png, jpg, jpeg, jp2, webp, gif, bmp\n\n"
    "## Backends & Pipeline\n"
    "Routed automatically across tiers — PaddleOCR (PaddleOCR-VL-1.5 then "
    "PaddleOCR-VL), followed by MinerU (mineru-vlm then mineru-lightweight). "
    "For URL inputs, each tier first tries URL submission; if rejected, the file "
    "is downloaded to a temp local copy (shared across tiers, auto-cleaned at end) "
    "and retried as local-file submission on the same tier.\n\n"
    "Tiers are auto-filtered by preflight (eligibility on API keys, size, and "
    "page count) and reported as `Skipped (preflight)`. If a hard error makes "
    "further attempts impossible, the pipeline exits early and the remaining "
    "tiers are reported as `Skipped (aborted)`.\n\n"
    "## Return\n\n"
    "### Success\n"
    "Starts with `Document parsed and saved.` followed by these fields:\n"
    "- `source`: the input target\n"
    "- `format`: file type with page count and size\n"
    "- `backend`: tier and model that produced the result\n"
    "- `content`: path to `content.md` — full markdown body "
    "(read with `read_file` offset/limit)\n"
    "- `images`: path to `images/` folder , if the document contains figures. "
    "Any `images/<name>` reference inside `content.md` resolves to a file here."
    "(read images with `read_file`, or call `document_parser` again on "
    "a specific image path when `read_file` returns media you cannot read or parse, "
    "or when you need its underlying structured content)\n"
    "- `layout`: path to `layout.json` — per-page block-level structure "
    "(text per block, label, bbox, reading order) for targeted or structural "
    "queries; usually not needed for general reading\n"
    "- `manifest`: path to `manifest.json` — debug metadata (file info, pipeline "
    "trace, skipped tiers, stats); usually not needed unless diagnosing\n\n"
    "### Failure\n"
    "Starts with `Error: document parsing failed.` followed by the pipeline "
    "execution trail:\n"
    "- `Tried`: per-tier per-mode attempts with their error class and message\n"
    "- `Skipped (aborted)`: tiers not attempted due to an early hard-error exit "
    "(if any)\n"
    "- `Skipped (preflight)`: tiers filtered out before any attempt with the "
    "reason (if any)"
)


@dataclass(frozen=True)
class _DocumentToolConfig:
    download_max_bytes: int = 200 * 1024 * 1024
    download_timeout_s: int = 120
    inspect_head_timeout_s: int = 15
    executor_timeout_s: float = 720.0


_CFG = _DocumentToolConfig()

_slug_locks: weakref.WeakValueDictionary[
    tuple[str | None, str], asyncio.Lock
] = weakref.WeakValueDictionary()


def _get_slug_lock(session_id: str | None, slug: str) -> asyncio.Lock:
    """Per-(session, slug) asyncio lock so concurrent parses of the same
    document (e.g. parent + sub-agent within one session) don't race on
    the artifact directory. Held by callers via ``async with``, so the
    weak-dict entry stays alive during contention; once all callers
    drop their reference, the lock is GC'd and the entry cleared."""
    key = (session_id, slug)
    lock = _slug_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _slug_locks[key] = lock
    return lock


def _build_pipeline(cfg: DocumentParserConfig) -> list[DocumentBackend]:
    provider = (cfg.provider or "auto").lower()
    paddle_key = cfg.paddleocr_api_key or ""
    mineru_key = cfg.mineru_api_key or ""
    if provider == "paddleocr":
        return [
            PaddleBackend(paddle_key, opts=PaddleOptions(model="PaddleOCR-VL-1.5")),
            PaddleBackend(paddle_key, opts=PaddleOptions(model="PaddleOCR-VL")),
        ]
    if provider == "mineru":
        return [
            MinerUV4Backend(mineru_key, opts=MinerUOptions(model="mineru-vlm")),
            MinerULightweightBackend(),
        ]
    return [
        PaddleBackend(paddle_key, opts=PaddleOptions(model="PaddleOCR-VL-1.5")),
        PaddleBackend(paddle_key, opts=PaddleOptions(model="PaddleOCR-VL")),
        MinerUV4Backend(mineru_key, opts=MinerUOptions(model="mineru-vlm")),
        MinerULightweightBackend(),
    ]


def _make_downloader(url: str, *, mime: str | None) -> DownloadFn:
    async def _download() -> Path:
        try:
            status, data = await http_get_bytes_with_retry(
                url,
                headers={"User-Agent": _USER_AGENT},
                timeout=_CFG.download_timeout_s,
                retry=HttpRetryConfig(max_attempts=3, base_delay=1.0),
                max_bytes=_CFG.download_max_bytes,
                allow_redirects=True,
            )
        except HttpResponseTooLargeError as e:
            raise DocumentBackendError(
                DocumentErrorClass.FILE_TOO_LARGE, None,
                f"URL exceeds {_CFG.download_max_bytes // (1024 * 1024)} MB download limit",
            ) from e
        except (aiohttp.ClientError, TimeoutError) as e:
            raise DocumentBackendError(
                DocumentErrorClass.DOWNLOAD_FAILED, None,
                f"network error during URL fetch: {type(e).__name__}",
            ) from e
        if status != 200 or not data:
            raise DocumentBackendError(
                DocumentErrorClass.DOWNLOAD_FAILED, status,
                f"HTTP {status} while downloading URL for localization",
            )
        suffix = (
            _MIME_TO_SUFFIX.get(mime or "")
            or (Path(unquote(urlparse(url).path)).suffix.lower() or ".bin")
        )
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
        except OSError as e:
            raise DocumentBackendError(
                DocumentErrorClass.IO_ERROR, None,
                f"I/O error writing downloaded URL to tempfile: {e}",
            ) from e
        return Path(tmp_path)

    return _download


def _finalize(
    success: PipelineSuccess,
    target: str,
    insp: TargetInspection,
    dest_dir: Path,
    slug: str,
) -> str:
    content_md_path = dest_dir / "content.md"
    md_text = (
        content_md_path.read_text(encoding="utf-8")
        if content_md_path.exists() else ""
    )
    md_tokens = count_tokens(md_text)
    md_lines = md_text.count("\n") + 1 if md_text else 0
    outcome = success.outcome

    try:
        write_manifest(
            dest_dir, slug=slug,
            source={
                "target": target,
                "name": insp.name,
                "origin": "local" if insp.is_local else "remote_url",
            },
            backend={"name": outcome.backend_name, "model": outcome.backend_model},
            size_bytes=insp.size_bytes,
            mime=insp.mime, kind=insp.kind,
            page_count=outcome.page_count,
            image_count=outcome.image_count,
            content_md_tokens=md_tokens, content_md_lines=md_lines,
            successful_tier_elapsed_ms=success.successful_tier_elapsed_ms,
            fallback_chain=[a.to_dict() for a in success.fallback_chain],
            skipped_tiers=success.skipped_tiers,
        )
    except OSError as e:
        logger.warning(
            "manifest write failed (%s); artifacts intact, cache will rebuild next call",
            e,
        )

    return format_success(
        slug_dir=dest_dir, source=target, name=insp.name, kind=insp.kind,
        page_count=outcome.page_count, size_mb=insp.size_mb,
        backend_name=outcome.backend_name, backend_model=outcome.backend_model,
        content_md_tokens=md_tokens, content_md_lines=md_lines,
        image_count=outcome.image_count,
    )


async def parse_document(
    *, target: str, session_id: str | None, slug_hint: str | None,
) -> str:
    if not target.strip():
        return "Error: target cannot be empty"

    cfg = resolve_document_parser_config(None)
    try:
        insp = await inspect_target(target)
    except FileNotFoundError:
        return f"Error: local file not found: {target}"

    backends = _build_pipeline(cfg)

    slug = make_slug(
        source=target,
        content_hash=hash_source(target, insp.is_local),
        suggested=slug_hint,
    )
    dest_dir = session_documents_root(session_id) / slug

    async with _get_slug_lock(session_id, slug):
        dest_dir.mkdir(parents=True, exist_ok=True)

        if already_parsed(dest_dir):
            return format_cached(dest_dir, target)

        keys = {
            "mineru": cfg.mineru_api_key or "",
            "paddleocr": cfg.paddleocr_api_key or "",
        }
        async with aiohttp.ClientSession() as session:
            try:
                success = await run_pipeline(
                    session, backends, target, insp, dest_dir, keys,
                    download=_make_downloader(target, mime=insp.mime),
                )
            except NoViableDocumentBackend as e:
                # No tier produced artifacts; clean up the empty session dir so
                # failed parses don't accumulate as cruft.
                try:
                    dest_dir.rmdir()
                except OSError:
                    pass
                return format_no_viable(
                    e.skipped,
                    [a.to_dict() for a in e.fallback_chain],
                    unattempted=e.unattempted,
                )

        return _finalize(success, target, insp, dest_dir, slug)


class DocumentParserTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="document_parser",
            description=DOCUMENT_PARSER_DESCRIPTION,
            executor_timeout=_CFG.executor_timeout_s,
            approval_resource_key="target",
        )
        self._session_id: str | None = None

    def bind_session(self, session_id: str | None) -> None:
        self._session_id = session_id

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "URL (http/https) or local path of the document.",
                    },
                },
                "required": ["target"],
            },
        )

    async def execute(self, **kwargs: Any) -> str:
        target: str = kwargs.get("target") or ""
        return await parse_document(
            target=target,
            session_id=self._session_id,
            slug_hint=None,
        )


document_parser = DocumentParserTool()
