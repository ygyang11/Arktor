"""Web content fetching tool with automatic HTML text extraction."""
from __future__ import annotations

import ipaddress
import json
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

from agent_harness import __version__ as _HARNESS_VERSION
from agent_harness.core.errors import HttpResponseTooLargeError
from agent_harness.tool.decorator import tool
from agent_harness.utils.http_retry import (
    HttpRetryConfig,
    HttpTextResponse,
    http_get_text_with_retry,
)
from agent_harness.utils.token_counter import truncate_text_by_tokens


@dataclass(frozen=True)
class WebFetchConfig:
    """Configuration for the web_fetch tool."""

    max_response_tokens: int = 25_000
    max_response_bytes: int = 5 * 1024 * 1024
    max_redirects: int = 5
    default_timeout: int = 30
    max_timeout: int = 120
    retry_max_attempts: int = 3
    retry_base_delay: float = 0.5
    executor_timeout_slack: float = 5.0
    allowed_schemes: frozenset[str] = frozenset({"http", "https"})
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
    )
    # Content-Type prefixes that indicate binary (non-readable) content
    binary_content_types: frozenset[str] = frozenset({
        "application/zip",
        "application/octet-stream",
        "application/gzip",
        "application/x-tar",
        "image/",
        "audio/",
        "video/",
        "font/",
    })
    # PDF detection: Content-Type values and URL suffixes
    pdf_content_types: frozenset[str] = frozenset({"application/pdf"})
    pdf_url_suffixes: frozenset[str] = frozenset({".pdf"})


_CFG = WebFetchConfig()
_EXECUTOR_TIMEOUT = (
    _CFG.max_timeout * _CFG.retry_max_attempts
    + _CFG.retry_base_delay * max(0, _CFG.retry_max_attempts - 1)
    + _CFG.executor_timeout_slack
)


def _honest_user_agent() -> str:
    return f"agent-harness/{_HARNESS_VERSION}"


def _is_cf_challenge(response: HttpTextResponse) -> bool:
    if response.status != 403:
        return False
    cf = next(
        (v for k, v in response.headers.items() if k.lower() == "cf-mitigated"),
        "",
    )
    return cf.lower() == "challenge"


class _TextExtractor(HTMLParser):
    """Extract visible text from HTML, skipping script/style blocks."""

    _SKIP_TAGS: frozenset[str] = frozenset({"script", "style", "noscript"})
    _BLOCK_TAGS: frozenset[str] = frozenset({
        "br", "p", "div", "li", "tr",
        "h1", "h2", "h3", "h4", "h5", "h6",
    })

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag.lower() in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        lines = (line.strip() for line in raw.splitlines())
        return "\n".join(line for line in lines if line)


def _reject_internal_host(host: str) -> None:
    """SSRF guard on the URL host literal only; no DNS resolution by
    design — pre-flight resolve-then-classify breaks transparent-proxy /
    fake-IP environments and never closed DNS-rebinding regardless.
    """
    lower = host.lower()
    if lower in {"localhost", "metadata.google.internal"}:
        raise ValueError(f"internal/private host blocked: {host!r}")
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return
    if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast:
        raise ValueError(f"internal/private host blocked: {host!r}")


async def _validate_url(url: str) -> None:
    if not url.strip():
        raise ValueError("URL cannot be empty")
    parsed = urlparse(url)
    if parsed.scheme not in _CFG.allowed_schemes:
        raise ValueError(
            f"unsupported URL scheme: {parsed.scheme!r} "
            f"(allowed: {', '.join(sorted(_CFG.allowed_schemes))})"
        )
    if not parsed.netloc:
        raise ValueError("invalid URL: missing host")
    _reject_internal_host(parsed.hostname or "")


def _extract_text_from_html(html: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(html)
    return extractor.get_text()


def _extract_from_html(html: str) -> str:
    """Boilerplate-stripping extraction via trafilatura; naive fallback.

    Falls back to the bundled _TextExtractor when trafilatura is absent
    or yields nothing, so extraction quality only ever improves.
    """
    try:
        import trafilatura  # type: ignore[import-untyped]  # noqa: PLC0415

        extracted: str | None = trafilatura.extract(
            html,
            output_format="markdown",
            include_links=True,
            include_tables=True,
            favor_precision=True,
        )
    except Exception:  # noqa: BLE001
        return _extract_text_from_html(html)
    if extracted and extracted.strip():
        return extracted
    return _extract_text_from_html(html)


def _is_binary_content_type(content_type: str) -> bool:
    ct = content_type.lower().split(";")[0].strip()
    return any(ct.startswith(prefix) for prefix in _CFG.binary_content_types)


def _is_pdf(content_type: str, url: str) -> bool:
    ct = content_type.lower().split(";")[0].strip()
    if ct in _CFG.pdf_content_types:
        return True
    # Fallback: check URL path suffix (handles octet-stream or missing Content-Type)
    path = urlparse(url).path.lower()
    return any(path.endswith(suffix) for suffix in _CFG.pdf_url_suffixes)


def _format_response(body: str, content_type: str) -> str:
    ct = content_type.lower()
    if "application/json" in ct:
        try:
            parsed = json.loads(body)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            return body
    if "text/html" in ct or "application/xhtml+xml" in ct:
        extracted = _extract_from_html(body)
        if len(extracted.strip()) < 500 and len(body) > 5000:
            extracted += (
                "\n\n[Note: only a little text was extracted from a much larger "
                "HTML page; it may be JS-rendered, an anti-bot/login wall, or a "
                "non-article page. Verify before relying on this as full content.]"
            )
        return extracted
    return body


def _retry_policy() -> HttpRetryConfig:
    return HttpRetryConfig(
        max_attempts=max(1, _CFG.retry_max_attempts),
        base_delay=_CFG.retry_base_delay,
    )


@tool(approval_resource_key="url", executor_timeout=_EXECUTOR_TIMEOUT)
async def web_fetch(url: str, timeout: int = 30) -> str:
    """Fetch content from a URL and return readable text.

    Fetches the given URL via GET. HTML pages are converted to plain
    text automatically. JSON responses are pretty-printed. Only
    http and https URLs are allowed.

    Args:
        url: The URL to fetch (http or https only).
        timeout: Per-attempt request time in seconds (positive integer, default 30, capped at 120).

    Returns:
        Page content as readable text, truncated to token budget.
        Errors are prefixed with ``Error:``.
    """
    if not url.strip():
        return "Error: URL cannot be empty"

    if timeout <= 0:
        return "Error: timeout must be greater than 0"
    timeout = min(timeout, _CFG.max_timeout)

    try:
        await _validate_url(url)
    except ValueError as exc:
        return f"Error: {exc}"

    try:
        import aiohttp  # noqa: PLC0415
    except ImportError:
        return "Error: aiohttp is not installed. Run `pip install aiohttp`."

    try:
        current = url
        headers = {"User-Agent": _CFG.user_agent}
        ua_flipped = False
        response: HttpTextResponse | None = None
        for _hop in range(_CFG.max_redirects + 1):
            # Manual redirect handling: every hop is re-validated so a
            # 3xx to an internal address cannot bypass the SSRF guard.
            response = await http_get_text_with_retry(
                current,
                headers=headers,
                timeout=timeout,
                retry=_retry_policy(),
                max_bytes=_CFG.max_response_bytes,
                allow_redirects=False,
            )
            if not ua_flipped and _is_cf_challenge(response):
                ua_flipped = True
                headers = {"User-Agent": _honest_user_agent()}
                continue
            if 300 <= response.status < 400:
                location = next(
                    (
                        v
                        for k, v in response.headers.items()
                        if k.lower() == "location"
                    ),
                    "",
                )
                if not location:
                    return (
                        f"Error: HTTP {response.status} for {current} "
                        f"(redirect without Location)"
                    )
                current = urljoin(current, location)
                try:
                    await _validate_url(current)
                except ValueError as exc:
                    return f"Error: {exc}"
                continue
            break
        else:
            return f"Error: too many redirects (> {_CFG.max_redirects}(max)) for {url}"

        if response is None:
            return f"Error: no response for {url}"
        if response.status >= 400:
            return f"Error: HTTP {response.status} for {current}"

        content_type = response.headers.get("Content-Type", "")

        if _is_pdf(content_type, current):
            return (
                "Error: URL is a PDF document. "
                "Use `pdf_parser` tool to extract text from this PDF "
                "if needed and the tool is available."
            )

        if _is_binary_content_type(content_type):
            ct_short = content_type.split(";")[0].strip()
            return f"Error: unsupported content type: {ct_short} (binary content cannot be read)"

        formatted = _format_response(response.body, content_type)
        return truncate_text_by_tokens(
            formatted,
            max_tokens=_CFG.max_response_tokens,
            suffix="\n... (truncated)",
        )
    except HttpResponseTooLargeError as exc:
        limit_mb = (exc.limit or 0) // (1024 * 1024)
        return (
            f"Error: {url} exceeds web_fetch's {limit_mb} MB limit; "
            f"nothing was returned. "
        )
    except TimeoutError:
        return "Error: request timed out"
    except UnicodeDecodeError:
        return "Error: failed to decode response (binary or non-UTF-8 content)"
    except aiohttp.ClientError as exc:
        return f"Error: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"
