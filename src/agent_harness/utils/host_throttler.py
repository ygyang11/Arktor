"""Per-host concurrency + min-interval throttling for outbound HTTP.

Transparent to all callers that go through ``http_retry``: a process-wide
registry maps each hostname to a shared semaphore + interval lock so that
concurrent requests to rate-limited hosts (arXiv, Semantic Scholar) are
serialized to their documented limits.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class HostProfile:
    concurrency: int = 4
    min_interval_s: float = 0.0


@dataclass
class _HostState:
    semaphore: asyncio.Semaphore
    interval_lock: asyncio.Lock
    last_request_ts: float = 0.0


# Only hosts whose limits are verified against official docs are listed.
_PROFILES: dict[str, HostProfile] = {
    "export.arxiv.org": HostProfile(concurrency=1, min_interval_s=3.0),
    "api.semanticscholar.org": HostProfile(concurrency=1, min_interval_s=1.0),
    "_default": HostProfile(concurrency=5, min_interval_s=0.0),
}

_STATES: dict[str, _HostState] = {}
_REGISTRY_LOCK = asyncio.Lock()


def _profile_for(host: str) -> HostProfile:
    return _PROFILES.get(host.lower(), _PROFILES["_default"])


async def _state_for(host: str) -> _HostState:
    state = _STATES.get(host)
    if state is not None:
        return state
    async with _REGISTRY_LOCK:
        state = _STATES.get(host)
        if state is None:
            profile = _profile_for(host)
            state = _HostState(
                semaphore=asyncio.Semaphore(profile.concurrency),
                interval_lock=asyncio.Lock(),
            )
            _STATES[host] = state
        return state


@asynccontextmanager
async def throttle(url: str) -> AsyncIterator[None]:
    """Acquire a per-host concurrency slot and honor the min-interval."""
    host = (urlparse(url).hostname or "").lower()
    profile = _profile_for(host)
    state = await _state_for(host)

    async with state.semaphore:
        if profile.min_interval_s > 0:
            async with state.interval_lock:
                wait = state.last_request_ts + profile.min_interval_s - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                state.last_request_ts = time.monotonic()
        yield
