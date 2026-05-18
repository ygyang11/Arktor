"""Tests for per-host concurrency + min-interval throttling."""
from __future__ import annotations

import asyncio
import time

import pytest

from agent_harness.utils import host_throttler
from agent_harness.utils.host_throttler import _profile_for, throttle


@pytest.fixture(autouse=True)
def _reset() -> None:
    host_throttler._STATES.clear()


def test_default_profile_for_unknown_host() -> None:
    for host in ("arxiv.org", "api.unpaywall.org", "example.com"):
        p = _profile_for(host)
        assert p.concurrency == 5
        assert p.min_interval_s == 0.0


def test_verified_profiles() -> None:
    arxiv = _profile_for("export.arxiv.org")
    assert (arxiv.concurrency, arxiv.min_interval_s) == (1, 3.0)
    s2 = _profile_for("api.semanticscholar.org")
    assert (s2.concurrency, s2.min_interval_s) == (1, 1.0)


@pytest.mark.asyncio
async def test_concurrency_limit_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        host_throttler._PROFILES,
        "export.arxiv.org",
        host_throttler.HostProfile(concurrency=1, min_interval_s=0.0),
    )
    in_flight = 0
    peak = 0

    async def worker() -> None:
        nonlocal in_flight, peak
        async with throttle("https://export.arxiv.org/api/query"):
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1

    await asyncio.gather(*[worker() for _ in range(4)])
    assert peak == 1


@pytest.mark.asyncio
async def test_min_interval_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        host_throttler._PROFILES,
        "export.arxiv.org",
        host_throttler.HostProfile(concurrency=1, min_interval_s=0.1),
    )
    starts: list[float] = []

    async def worker() -> None:
        async with throttle("https://export.arxiv.org/api/query"):
            starts.append(time.monotonic())

    await asyncio.gather(*[worker() for _ in range(3)])
    starts.sort()
    assert starts[2] - starts[0] >= 0.18


@pytest.mark.asyncio
async def test_default_host_no_interval_high_concurrency() -> None:
    in_flight = 0
    peak = 0

    async def worker() -> None:
        nonlocal in_flight, peak
        async with throttle("https://example.com/page"):
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1

    await asyncio.gather(*[worker() for _ in range(5)])
    assert peak == 5


@pytest.mark.asyncio
async def test_state_shared_across_calls_same_host() -> None:
    async with throttle("https://export.arxiv.org/a"):
        pass
    s1 = host_throttler._STATES["export.arxiv.org"]
    async with throttle("https://export.arxiv.org/b"):
        pass
    assert host_throttler._STATES["export.arxiv.org"] is s1
