"""PDF parsing tool for extracting text from PDF documents."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from agent_harness.core.config import resolve_pdf_config
from agent_harness.tool.base import BaseTool
from agent_harness.tool.decorator import tool
from agent_harness.utils.http_retry import (
    HttpRetryConfig,
    http_get_bytes_with_retry,
    http_get_with_retry,
    http_post_json_with_retry,
)
from agent_harness.utils.token_counter import truncate_text_by_tokens

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PdfParserConfig:
    """Configuration for the pdf_parser tool."""

    max_output_tokens: int = 15_000
    # PaddleOCR's official sample polls every 5s with no upper bound and
    # MinerU's sample uses a 300s client timeout -- 60 * 5s = 300s
    poll_interval: float = 5.0
    max_poll_attempts: int = 60
    request_max_attempts: int = 3
    request_base_delay: float = 1.0
    # Single source of truth for per-call HTTP timeouts (submit / poll /
    # get-upload-url use request_timeout_s; the file PUT/upload uses
    # upload_timeout_s). The executor ceiling is derived from these.
    request_timeout_s: int = 30
    upload_timeout_s: int = 120
    executor_slack_s: float = 60.0


_CFG = PdfParserConfig()

# Executor ceiling must exceed the worst single run: poll budget + local
# file upload + submission request + slack.
_PDF_EXECUTOR_TIMEOUT = (
    _CFG.max_poll_attempts * _CFG.poll_interval
    + _CFG.upload_timeout_s
    + _CFG.request_timeout_s
    + _CFG.executor_slack_s
)

_MINERU_BASE = "https://mineru.net"
_PADDLEOCR_JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"


def _retry_policy() -> HttpRetryConfig:
    return HttpRetryConfig(
        max_attempts=max(1, _CFG.request_max_attempts),
        base_delay=_CFG.request_base_delay,
    )


async def _get_json_with_retry(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = _CFG.request_timeout_s,
) -> tuple[int, dict[str, object] | None, str]:
    status, body = await http_get_with_retry(
        url,
        headers=headers,
        timeout=timeout,
        retry=_retry_policy(),
    )
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        data = None
    return status, data, body


async def _post_json_with_retry(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: object | None = None,
    timeout: int = _CFG.request_timeout_s,
) -> tuple[int, dict[str, object] | None, str]:
    status, body = await http_post_json_with_retry(
        url,
        headers=headers,
        json_body=json_body,
        timeout=timeout,
        retry=_retry_policy(),
    )
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        data = None
    return status, data, body


def _is_local_file(path: str) -> bool:
    return not path.startswith(("http://", "https://"))


async def _read_local_file(path: str) -> tuple[str, bytes]:
    """Read a local file and return (filename, bytes). Returns error string on failure."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not p.suffix.lower() == ".pdf":
        raise ValueError(f"Expected a PDF file, got: {p.suffix}")
    return p.name, p.read_bytes()


# ---------------------------------------------------------------------------
# MinerU local file upload
# ---------------------------------------------------------------------------


async def _upload_to_mineru(file_name: str, file_bytes: bytes, api_key: str) -> str:
    """Upload a local file to MinerU and return the remote URL."""
    import aiohttp  # noqa: PLC0415

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{_MINERU_BASE}/api/v4/file-urls/batch",
            headers=headers,
            json={"file_names": [file_name]},
            timeout=aiohttp.ClientTimeout(total=_CFG.request_timeout_s),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Failed to get upload URL (HTTP {resp.status}): {body[:200]}")
            result = await resp.json()

    data = result.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("Invalid response from MinerU file-urls API")
    batch_id = data.get("batch_id", "")
    file_urls = data.get("file_urls", [])
    if not file_urls:
        raise RuntimeError("MinerU returned no upload URLs")

    upload_url = file_urls[0]

    import aiohttp as _aio  # noqa: PLC0415

    async with _aio.ClientSession() as session:
        async with session.put(
            upload_url,
            data=file_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=_aio.ClientTimeout(total=_CFG.upload_timeout_s),
        ) as resp:
            if resp.status not in (200, 201):
                raise RuntimeError(f"File upload failed (HTTP {resp.status})")

    return upload_url


async def _upload_to_mineru_lightweight(file_name: str, file_bytes: bytes) -> str:
    """Upload a local file via MinerU lightweight API and return the task_id directly."""
    import aiohttp  # noqa: PLC0415

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{_MINERU_BASE}/api/v1/agent/parse/file",
            json={"file_name": file_name},
            timeout=aiohttp.ClientTimeout(total=_CFG.request_timeout_s),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Failed to get upload URL (HTTP {resp.status}): {body[:200]}")
            result = await resp.json()

    data = result.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("Invalid response from MinerU lightweight file API")
    file_url = data.get("file_url", "")
    if not file_url:
        raise RuntimeError("MinerU returned no upload URL")

    async with aiohttp.ClientSession() as session:
        async with session.put(
            file_url,
            data=file_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=aiohttp.ClientTimeout(total=_CFG.upload_timeout_s),
        ) as resp:
            if resp.status not in (200, 201):
                raise RuntimeError(f"File upload failed (HTTP {resp.status})")

    return file_url


# ---------------------------------------------------------------------------
# PaddleOCR local file upload
# ---------------------------------------------------------------------------


async def _parse_paddleocr_file_with_model(
    file_name: str, file_bytes: bytes, api_key: str, model: str,
) -> str:
    """Parse a local PDF via PaddleOCR multipart upload."""
    import aiohttp  # noqa: PLC0415

    headers = {"Authorization": f"bearer {api_key}"}
    form = aiohttp.FormData()
    form.add_field("model", model)
    form.add_field(
        "optionalPayload",
        json.dumps({
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useChartRecognition": False,
        }),
    )
    form.add_field("file", file_bytes, filename=file_name, content_type="application/pdf")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                _PADDLEOCR_JOB_URL,
                headers=headers,
                data=form,
                timeout=aiohttp.ClientTimeout(total=_CFG.upload_timeout_s),
            ) as resp:
                status = resp.status
                body = await resp.text()
    except Exception as exc:
        return f"Error: PDF parsing task submission failed: {exc}"

    if status == 429:
        return "Error: rate limit exceeded"
    if status != 200:
        return f"Error: PDF parsing task submission failed (HTTP {status}): {body[:200]}"

    try:
        result = json.loads(body)
    except json.JSONDecodeError:
        return "Error: PDF parsing task submission returned invalid JSON"

    data_obj = result.get("data")
    if not isinstance(data_obj, dict):
        return "Error: PDF parsing service returned invalid submission payload"
    job_id_obj = data_obj.get("jobId")
    job_id = job_id_obj if isinstance(job_id_obj, str) else ""
    if not job_id:
        return "Error: PDF parsing service returned no job ID"

    for _ in range(_CFG.max_poll_attempts):
        await asyncio.sleep(_CFG.poll_interval)
        try:
            poll_status, poll_result, _body = await _get_json_with_retry(
                f"{_PADDLEOCR_JOB_URL}/{job_id}",
                headers={"Authorization": f"bearer {api_key}"},
            )
        except Exception as exc:
            return f"Error: PDF parsing status polling failed: {exc}"
        if poll_status == 429:
            return "Error: rate limit exceeded"
        if poll_status != 200:
            return f"Error: PDF parsing status polling failed (HTTP {poll_status})"
        if poll_result is None:
            return "Error: PDF parsing status polling returned invalid JSON"
        poll_data = poll_result.get("data")
        if not isinstance(poll_data, dict):
            return "Error: PDF parsing status polling returned invalid payload"
        state_obj = poll_data.get("state")
        state = state_obj if isinstance(state_obj, str) else ""
        if state == "done":
            result_url_obj = poll_data.get("resultUrl")
            result_url = result_url_obj if isinstance(result_url_obj, dict) else {}
            json_url_obj = result_url.get("jsonUrl")
            json_url = json_url_obj if isinstance(json_url_obj, str) else ""
            if not json_url:
                return "Error: PDF parsing service returned no result URL"
            return await _download_paddleocr_markdown(json_url)
        if state == "failed":
            err_obj = poll_data.get("errorMsg")
            err = err_obj if isinstance(err_obj, str) and err_obj else "unknown error"
            return f"Error: PDF parsing failed: {err}"

    return "Error: PDF parsing timed out"


# ---------------------------------------------------------------------------
# MinerU providers
# ---------------------------------------------------------------------------


async def _parse_mineru(url: str, api_key: str) -> str:
    """Parse PDF via MinerU precise API."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        status, result, body = await _post_json_with_retry(
            f"{_MINERU_BASE}/api/v4/extract/task",
            headers=headers,
            json_body={
                "url": url,
                "model_version": "vlm",
                "enable_formula": True,
                "enable_table": True,
            },
        )
    except Exception as exc:
        return f"Error: PDF parsing task submission failed: {exc}"
    if status != 200:
        return f"Error: PDF parsing task submission failed (HTTP {status}): {body[:200]}"
    if result is None:
        return "Error: PDF parsing task submission returned invalid JSON"

    if result.get("code") != 0:
        return f"Error: PDF parsing task submission failed: {result.get('msg', 'unknown error')}"
    data_obj = result.get("data")
    if not isinstance(data_obj, dict):
        return "Error: PDF parsing service returned invalid submission payload"
    task_id_obj = data_obj.get("task_id")
    task_id = task_id_obj if isinstance(task_id_obj, str) else ""
    if not task_id:
        return "Error: PDF parsing service returned no task ID"

    for _ in range(_CFG.max_poll_attempts):
        await asyncio.sleep(_CFG.poll_interval)
        try:
            status, result, _body = await _get_json_with_retry(
                f"{_MINERU_BASE}/api/v4/extract/task/{task_id}",
                headers=headers,
            )
        except Exception as exc:
            return f"Error: PDF parsing status polling failed: {exc}"
        if status != 200:
            return f"Error: PDF parsing status polling failed (HTTP {status})"
        if result is None:
            return "Error: PDF parsing status polling returned invalid JSON"
        poll_data = result.get("data")
        if not isinstance(poll_data, dict):
            return "Error: PDF parsing status polling returned invalid payload"
        state_obj = poll_data.get("state")
        state = state_obj if isinstance(state_obj, str) else ""
        if state == "done":
            zip_url_obj = poll_data.get("full_zip_url")
            zip_url = zip_url_obj if isinstance(zip_url_obj, str) else ""
            if not zip_url:
                return "Error: PDF parsing service returned no result URL"
            return await _download_mineru_markdown(zip_url)
        if state == "failed":
            err_obj = poll_data.get("err_msg")
            err = err_obj if isinstance(err_obj, str) and err_obj else "unknown error"
            return f"Error: PDF parsing failed: {err}"

    return "Error: PDF parsing timed out"


_MAX_ZIP_ENTRY_SIZE = 50 * 1024 * 1024  # 50 MB


async def _download_mineru_markdown(zip_url: str) -> str:
    """Download ZIP from MinerU and extract the markdown content."""
    import io  # noqa: PLC0415
    import zipfile  # noqa: PLC0415

    try:
        status, data = await http_get_bytes_with_retry(
            zip_url,
            retry=_retry_policy(),
        )
    except Exception as exc:
        return f"Error: failed to download PDF parsing result: {exc}"
    if status != 200:
        return f"Error: failed to download PDF parsing result (HTTP {status})"

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            md_files: list[str] = []
            for name in zf.namelist():
                if ".." in name or name.startswith("/"):
                    logger.warning("Skipping suspicious ZIP entry: %s", name)
                    continue
                info = zf.getinfo(name)
                if info.file_size > _MAX_ZIP_ENTRY_SIZE:
                    logger.warning(
                        "Skipping oversized ZIP entry: %s (%d bytes)",
                        name, info.file_size,
                    )
                    continue
                if name.endswith(".md"):
                    md_files.append(name)

            if not md_files:
                return "Error: no markdown file found in PDF parsing result"
            target = next((n for n in md_files if "full.md" in n), md_files[0])
            return zf.read(target).decode("utf-8", errors="replace")
    except zipfile.BadZipFile:
        return "Error: PDF parsing service returned invalid result"


async def _parse_mineru_lightweight(url: str) -> str:
    """Parse PDF via MinerU lightweight agent API (no token required)."""
    try:
        status, result, body = await _post_json_with_retry(
            f"{_MINERU_BASE}/api/v1/agent/parse/url",
            json_body={"url": url},
        )
    except Exception as exc:
        return f"Error: PDF parsing task submission failed: {exc}"
    if status != 200:
        return f"Error: PDF parsing task submission failed (HTTP {status}): {body[:200]}"
    if result is None:
        return "Error: PDF parsing task submission returned invalid JSON"

    data_obj = result.get("data")
    if not isinstance(data_obj, dict):
        return "Error: PDF parsing service returned invalid submission payload"
    task_id_obj = data_obj.get("task_id")
    task_id = task_id_obj if isinstance(task_id_obj, str) else ""
    if not task_id:
        msg_obj = result.get("msg")
        msg = msg_obj if isinstance(msg_obj, str) else ""
        return f"Error: PDF parsing task submission failed: {msg}"

    for _ in range(_CFG.max_poll_attempts):
        await asyncio.sleep(_CFG.poll_interval)
        try:
            status, result, _body = await _get_json_with_retry(
                f"{_MINERU_BASE}/api/v1/agent/parse/{task_id}",
            )
        except Exception as exc:
            return f"Error: PDF parsing status polling failed: {exc}"
        if status != 200:
            return f"Error: PDF parsing status polling failed (HTTP {status})"
        if result is None:
            return "Error: PDF parsing status polling returned invalid JSON"
        poll_data = result.get("data")
        if not isinstance(poll_data, dict):
            return "Error: PDF parsing status polling returned invalid payload"
        state_obj = poll_data.get("state")
        state = state_obj if isinstance(state_obj, str) else ""
        if state == "done":
            md_url_obj = poll_data.get("markdown_url")
            md_url = md_url_obj if isinstance(md_url_obj, str) else ""
            if not md_url:
                return "Error: PDF parsing service returned no result URL"
            try:
                md_status, md_body = await http_get_with_retry(
                    md_url,
                    retry=_retry_policy(),
                )
            except Exception as exc:
                return f"Error: failed to download PDF parsing result: {exc}"
            if md_status != 200:
                return f"Error: failed to download PDF parsing result (HTTP {md_status})"
            return md_body
        if state == "failed":
            err_obj = poll_data.get("err_msg")
            err = err_obj if isinstance(err_obj, str) and err_obj else "unknown error"
            return f"Error: PDF parsing failed: {err}"

    return "Error: PDF parsing timed out"


# ---------------------------------------------------------------------------
# PaddleOCR provider
# ---------------------------------------------------------------------------


async def _parse_paddleocr_with_model(url: str, api_key: str, model: str) -> str:
    """Parse PDF via PaddleOCR async API with the specified model."""
    headers = {"Authorization": f"bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "fileUrl": url,
        "model": model,
        "optionalPayload": {
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useChartRecognition": False,
        },
    }

    try:
        status, result, body = await _post_json_with_retry(
            _PADDLEOCR_JOB_URL,
            headers=headers,
            json_body=payload,
        )
    except Exception as exc:
        return f"Error: PDF parsing task submission failed: {exc}"
    if status == 429:
        return "Error: rate limit exceeded"
    if status != 200:
        return f"Error: PDF parsing task submission failed (HTTP {status}): {body[:200]}"
    if result is None:
        return "Error: PDF parsing task submission returned invalid JSON"

    data_obj = result.get("data")
    if not isinstance(data_obj, dict):
        return "Error: PDF parsing service returned invalid submission payload"
    job_id_obj = data_obj.get("jobId")
    job_id = job_id_obj if isinstance(job_id_obj, str) else ""
    if not job_id:
        return "Error: PDF parsing service returned no job ID"

    for _ in range(_CFG.max_poll_attempts):
        await asyncio.sleep(_CFG.poll_interval)
        try:
            status, result, _body = await _get_json_with_retry(
                f"{_PADDLEOCR_JOB_URL}/{job_id}",
                headers={"Authorization": f"bearer {api_key}"},
            )
        except Exception as exc:
            return f"Error: PDF parsing status polling failed: {exc}"
        if status == 429:
            return "Error: rate limit exceeded"
        if status != 200:
            return f"Error: PDF parsing status polling failed (HTTP {status})"
        if result is None:
            return "Error: PDF parsing status polling returned invalid JSON"
        poll_data = result.get("data")
        if not isinstance(poll_data, dict):
            return "Error: PDF parsing status polling returned invalid payload"
        state_obj = poll_data.get("state")
        state = state_obj if isinstance(state_obj, str) else ""
        if state == "done":
            result_url_obj = poll_data.get("resultUrl")
            result_url = result_url_obj if isinstance(result_url_obj, dict) else {}
            json_url_obj = result_url.get("jsonUrl")
            json_url = json_url_obj if isinstance(json_url_obj, str) else ""
            if not json_url:
                return "Error: PDF parsing service returned no result URL"
            return await _download_paddleocr_markdown(json_url)
        if state == "failed":
            err_obj = poll_data.get("errorMsg")
            err = err_obj if isinstance(err_obj, str) and err_obj else "unknown error"
            return f"Error: PDF parsing failed: {err}"

    return "Error: PDF parsing timed out"


async def _download_paddleocr_markdown(json_url: str) -> str:
    """Download JSONL result from PaddleOCR and extract markdown text."""
    try:
        status, text = await http_get_with_retry(
            json_url,
            retry=_retry_policy(),
        )
    except Exception as exc:
        return f"Error: failed to download PDF parsing result: {exc}"
    if status != 200:
        return f"Error: failed to download PDF parsing result (HTTP {status})"

    pages: list[str] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        for res in data.get("result", {}).get("layoutParsingResults", []):
            md_text: str = res.get("markdown", {}).get("text", "")
            if md_text.strip():
                pages.append(md_text.strip())

    if not pages:
        return "Error: PDF contains no extractable text"
    return "\n\n".join(pages)


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------


@tool(approval_resource_key="url", executor_timeout=_PDF_EXECUTOR_TIMEOUT)
async def pdf_parser(url: str) -> str:
    """Extract text from a PDF document and return structured Markdown text.

    Accepts a URL or local file path. The document is parsed via a
    cloud service and returned as markdown text. Supports complex
    layouts, tables, and formulas.

    Args:
        url: URL or local file path of the PDF document to parse.

    Returns:
        Extracted text in markdown format, truncated to token budget.
        Errors are prefixed with ``Error:``.
    """
    if not url.strip():
        return "Error: URL or file path cannot be empty"

    cfg = resolve_pdf_config(None)
    provider = cfg.provider
    is_local = _is_local_file(url)

    # Read local file upfront if needed
    file_name = ""
    file_bytes = b""
    if is_local:
        try:
            file_name, file_bytes = await _read_local_file(url)
        except (FileNotFoundError, ValueError) as exc:
            return f"Error: {exc}"

    if provider == "mineru":
        api_key = cfg.mineru_api_key or ""
        if is_local:
            if api_key:
                try:
                    remote_url = await _upload_to_mineru(file_name, file_bytes, api_key)
                except Exception as exc:
                    logger.warning("MinerU precise upload failed, trying lightweight: %s", exc)
                    try:
                        remote_url = await _upload_to_mineru_lightweight(file_name, file_bytes)
                    except Exception as exc2:
                        return f"Error: file upload failed: {exc2}"
            else:
                try:
                    remote_url = await _upload_to_mineru_lightweight(file_name, file_bytes)
                except Exception as exc:
                    return f"Error: file upload failed: {exc}"
            raw = await _parse_mineru(remote_url, api_key) if api_key else await _parse_mineru_lightweight(remote_url)
        else:
            if api_key:
                raw = await _parse_mineru(url, api_key)
                if raw.startswith("Error:"):
                    logger.warning("MinerU precise API failed, falling back to lightweight: %s", raw)
                    raw = await _parse_mineru_lightweight(url)
            else:
                raw = await _parse_mineru_lightweight(url)
    elif provider == "paddleocr":
        api_key = cfg.paddleocr_api_key or ""
        if not api_key:
            return (
                "PDF parsing not configured: PADDLEOCR_API_KEY not set. "
                "Set the environment variable or configure in config.yaml."
            )
        if is_local:
            raw = await _parse_paddleocr_file_with_model(
                file_name, file_bytes, api_key, "PaddleOCR-VL-1.5",
            )
            if raw.startswith("Error:"):
                logger.warning("PaddleOCR VL-1.5 file upload failed, falling back to VL: %s", raw)
                raw = await _parse_paddleocr_file_with_model(
                    file_name, file_bytes, api_key, "PaddleOCR-VL",
                )
        else:
            raw = await _parse_paddleocr_with_model(url, api_key, "PaddleOCR-VL-1.5")
            if raw.startswith("Error:"):
                logger.warning("PaddleOCR VL-1.5 failed, falling back to VL: %s", raw)
                raw = await _parse_paddleocr_with_model(url, api_key, "PaddleOCR-VL")
    else:
        return f"Unknown PDF provider: {provider!r}. Use 'mineru' or 'paddleocr'."

    if raw.startswith("Error:"):
        return raw

    return truncate_text_by_tokens(
        raw,
        max_tokens=_CFG.max_output_tokens,
        suffix="\n... (truncated)",
    )


PDF_TOOLS: list[BaseTool] = [pdf_parser]
