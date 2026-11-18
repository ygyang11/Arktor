"""Cached release checks for the interactive CLI."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from packaging.version import InvalidVersion, Version

_CACHE_TTL = 24 * 60 * 60
_RELEASE_URL = "https://pypi.org/pypi/arktor/json"


def _cache_path() -> Path:
    return Path.home() / ".arktor" / "cache" / "update.json"


def _read_cache() -> dict[str, Any]:
    try:
        data = json.loads(_cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def newest_stable(versions: list[str]) -> str | None:
    latest: Version | None = None
    for value in versions:
        try:
            candidate = Version(value)
        except InvalidVersion:
            continue
        if candidate.is_prerelease or candidate.is_devrelease:
            continue
        if latest is None or candidate > latest:
            latest = candidate
    return str(latest) if latest is not None else None


def read_cached_notice(installed: str) -> tuple[str | None, bool]:
    """Return a newer cached version and whether a refresh is due."""
    if os.environ.get("ARKTOR_NO_UPDATE_CHECK") == "1":
        return None, False
    data = _read_cache()
    value = data.get("latest")
    latest = newest_stable([value]) if isinstance(value, str) else None
    try:
        notice = latest if latest is not None and Version(latest) > Version(installed) else None
    except InvalidVersion:
        notice = None
    checked_at = data.get("checked_at")
    now = time.time()
    fresh = isinstance(checked_at, (int, float)) and now - _CACHE_TTL < checked_at <= now
    return notice, not fresh


async def refresh_version_cache() -> None:
    if os.environ.get("ARKTOR_NO_UPDATE_CHECK") == "1":
        return
    previous = _read_cache().get("latest")
    latest = previous if isinstance(previous, str) else None
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(_RELEASE_URL)
            response.raise_for_status()
            releases = response.json()["releases"]
        if not isinstance(releases, dict):
            raise ValueError("Invalid release metadata.")
        versions = [
            version
            for version, artifacts in releases.items()
            if isinstance(artifacts, list)
            and any(
                isinstance(artifact, dict) and not artifact.get("yanked", False)
                for artifact in artifacts
            )
        ]
        latest = newest_stable(versions)
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        pass

    path = _cache_path()
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump({"checked_at": time.time(), "latest": latest}, output)
        temporary.replace(path)
    except OSError:
        pass
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
