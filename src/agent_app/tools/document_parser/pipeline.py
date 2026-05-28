"""Fixed 4-tier pipeline: preflight + ordered fallback + URL-to-local in-tier retry."""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import aiohttp

from agent_app.tools.document_parser.backends import (
    DocumentBackend,
    DocumentBackendOutcome,
)
from agent_app.tools.document_parser.errors import (
    DocumentBackendError,
    DocumentErrorClass,
    NoViableDocumentBackend,
    TierAttempt,
)
from agent_app.tools.document_parser.storage import TargetInspection

logger = logging.getLogger(__name__)
T = TypeVar("T")

_FALLBACK_CLASSES = frozenset({
    DocumentErrorClass.AUTH_FAILED,
    DocumentErrorClass.RATE_LIMITED,
    DocumentErrorClass.QUOTA_EXCEEDED,
    DocumentErrorClass.BACKEND_TRANSIENT_ERROR,
    DocumentErrorClass.BACKEND_READ_FAILED,
    DocumentErrorClass.BACKEND_FETCH_FAILED,
    DocumentErrorClass.TIMEOUT,
    DocumentErrorClass.FILE_TOO_LARGE,
    DocumentErrorClass.PAGE_LIMIT,
    DocumentErrorClass.UNSUPPORTED_BY_TIER,
    DocumentErrorClass.DOWNLOAD_FAILED,
    DocumentErrorClass.UNKNOWN,
})
_URL_RETRY_CLASSES = frozenset({
    DocumentErrorClass.BACKEND_FETCH_FAILED,
    DocumentErrorClass.UNSUPPORTED_BY_TIER,
    DocumentErrorClass.TIMEOUT,
})
_INTRA_TIER_RETRY_CLASSES = frozenset({
    DocumentErrorClass.BACKEND_TRANSIENT_ERROR,
    DocumentErrorClass.RATE_LIMITED,
})


@dataclass(frozen=True)
class _PipelineRetryConfig:
    max_retries: int = 2
    base_delay: float = 2.0


_CFG = _PipelineRetryConfig()

DownloadFn = Callable[[], Awaitable[Path]]


@dataclass
class PipelineSuccess:
    outcome: DocumentBackendOutcome
    fallback_chain: list[TierAttempt]
    skipped_tiers: list[dict[str, str]]
    successful_tier_elapsed_ms: int


def preflight(
    backends: list[DocumentBackend], insp: TargetInspection, keys: dict[str, str],
) -> tuple[list[DocumentBackend], list[dict[str, str]]]:
    plan: list[DocumentBackend] = []
    skipped: list[dict[str, str]] = []
    for b in backends:
        if b.needs_key and not keys.get(b.needs_key):
            skipped.append({"tier": b.name, "reason": f"no_api_key:{b.needs_key}"})
            continue
        size_limit = b.max_mb_local if insp.is_local else b.max_mb_url
        size_mode = "local" if insp.is_local else "url"
        if (
            size_limit is not None
            and insp.size_mb is not None
            and insp.size_mb > size_limit
        ):
            skipped.append({"tier": b.name, "reason": f"size>{size_limit}MB({size_mode})"})
            continue
        if (
            b.max_pages is not None
            and insp.pages is not None
            and insp.pages > b.max_pages
        ):
            skipped.append({"tier": b.name, "reason": f"pages>{b.max_pages}"})
            continue
        plan.append(b)
    return plan, skipped


async def run_pipeline(
    session: aiohttp.ClientSession,
    backends: list[DocumentBackend],
    target: str,
    insp: TargetInspection,
    dest_dir: Path,
    keys: dict[str, str],
    *,
    download: DownloadFn,
) -> PipelineSuccess:
    plan, skipped = preflight(backends, insp, keys)
    if not plan:
        raise NoViableDocumentBackend(skipped)

    fallback_chain: list[TierAttempt] = []
    localized_cache: list[Path] = []
    localize_failure: list[DocumentBackendError] = []

    async def localize(tier_name: str) -> Path:
        if localize_failure:
            # A prior tier already tried and failed to download this URL —
            # the URL is the same, the failure mode (transport / redirect /
            # SSRF / size) is the same, so we re-raise the cached error
            raise localize_failure[0]
        if not localized_cache:
            start = time.monotonic()
            try:
                localized_cache.append(await download())
            except DocumentBackendError as e:
                localize_failure.append(e)
                fallback_chain.append(TierAttempt(
                    tier=tier_name, mode="localize",
                    error_class=e.error_class.value, error_message=e.message,
                    elapsed_ms=int((time.monotonic() - start) * 1000),
                ))
                raise
        return localized_cache[0]

    try:
        for i, tier in enumerate(plan):
            tier_start = time.monotonic()
            try:
                if insp.is_local:
                    local_path = Path(target).expanduser().resolve()

                    async def _run_local(
                        t: DocumentBackend = tier, p: Path = local_path,
                    ) -> DocumentBackendOutcome:
                        return await t.parse_local(session, p, dest_dir)

                    outcome = await _call_with_intra_retry(
                        tier.name, "local", fallback_chain, _run_local,
                    )
                else:
                    outcome = await _try_url_then_local(
                        session, tier, target, dest_dir,
                        localize=localize, chain=fallback_chain,
                    )
                tier_elapsed_ms = int((time.monotonic() - tier_start) * 1000)
                return PipelineSuccess(
                    outcome, fallback_chain, skipped,
                    successful_tier_elapsed_ms=tier_elapsed_ms,
                )
            except DocumentBackendError as e:
                if e.error_class in _FALLBACK_CLASSES:
                    continue
                raise NoViableDocumentBackend(
                    skipped, fallback_chain,
                    unattempted=[t.name for t in plan[i + 1:]],
                ) from e

        raise NoViableDocumentBackend(skipped, fallback_chain)
    finally:
        for p in localized_cache:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass


async def _call_with_intra_retry(
    tier_name: str,
    mode: str,
    chain: list[TierAttempt],
    call: Callable[[], Awaitable[T]],
) -> T:
    for attempt in range(_CFG.max_retries + 1):
        start = time.monotonic()
        try:
            return await call()
        except DocumentBackendError as e:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            label = mode if attempt == 0 else f"{mode}-retry{attempt}"
            chain.append(TierAttempt(
                tier=tier_name, mode=label,
                error_class=e.error_class.value, error_message=e.message,
                elapsed_ms=elapsed_ms,
            ))
            if (
                e.error_class not in _INTRA_TIER_RETRY_CLASSES
                or attempt == _CFG.max_retries
            ):
                raise
            wait_s = _CFG.base_delay * (2 ** attempt)
            logger.debug(
                "intra-tier retry %s/%s after %s, wait %.1fs",
                attempt + 1, _CFG.max_retries, e.error_class.value, wait_s,
            )
            await asyncio.sleep(wait_s)
    chain.append(TierAttempt(
        tier=tier_name, mode=mode,
        error_class=DocumentErrorClass.UNKNOWN.value,
        error_message="intra-tier retry loop exited unexpectedly",
        elapsed_ms=0,
    ))
    raise DocumentBackendError(
        DocumentErrorClass.UNKNOWN, None,
        "intra-tier retry loop exited unexpectedly",
    )


async def _try_url_then_local(
    session: aiohttp.ClientSession,
    tier: DocumentBackend,
    url: str,
    dest_dir: Path,
    *,
    localize: Callable[[str], Awaitable[Path]],
    chain: list[TierAttempt],
) -> DocumentBackendOutcome:
    async def _run_url() -> DocumentBackendOutcome:
        return await tier.parse_url(url, dest_dir)

    try:
        return await _call_with_intra_retry(
            tier.name, "url", chain, _run_url,
        )
    except DocumentBackendError as e:
        if e.error_class not in _URL_RETRY_CLASSES:
            raise
        local_path = await localize(tier.name)

        async def _run_local() -> DocumentBackendOutcome:
            return await tier.parse_local(session, local_path, dest_dir)

        return await _call_with_intra_retry(
            tier.name, "local", chain, _run_local,
        )
