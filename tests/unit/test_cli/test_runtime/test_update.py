from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from agent_cli.runtime import update


@pytest.fixture(autouse=True)
def cache_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.delenv('ARKTOR_NO_UPDATE_CHECK', raising=False)
    monkeypatch.setattr(update.time, 'time', lambda: 100000)


def cache(data):
    path = update._cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_version_selection_and_cache_age(monkeypatch):
    assert update.newest_stable(['bad', '0.9', '0.10', '1.0rc1', '2.0.dev1']) == '0.10'
    assert update.read_cached_notice('0.6.0') == (None, True)
    cache({'latest': '0.10', 'checked_at': 99999})
    assert update.read_cached_notice('0.6.0') == ('0.10', False)
    assert update.read_cached_notice('0.10') == (None, False)
    cache({'latest': '0.10', 'checked_at': 100001})
    assert update.read_cached_notice('0.6.0') == ('0.10', True)
    monkeypatch.setenv('ARKTOR_NO_UPDATE_CHECK', '1')
    assert update.read_cached_notice('0.6.0') == (None, False)


@pytest.mark.parametrize('status', [200, 503])
async def test_refresh_http_cache_and_yanked_versions(status):
    cache({'latest': '0.7', 'checked_at': 0})
    def handler(request):
        return httpx.Response(status, json={'releases': {
            '0.9': [{'yanked': False}], '1.0': [{'yanked': True}],
            '2.0': [], '3.0rc1': [{'yanked': False}], 'bad': [{}]}})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch.object(update.httpx, 'AsyncClient', return_value=client):
        await update.refresh_version_cache()
    assert update.read_cached_notice('0.6') == ('0.9' if status==200 else '0.7', False)
    assert not list(update._cache_path().parent.glob('tmp*'))


async def test_cancel_refresh_is_not_swallowed():
    entered = asyncio.Event()
    async def handler(request):
        entered.set()
        await asyncio.Event().wait()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch.object(update.httpx, 'AsyncClient', return_value=client):
        task = asyncio.create_task(update.refresh_version_cache())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not update._cache_path().exists()


async def test_cache_replace_failure_keeps_old_file(monkeypatch):
    cache({'latest': '0.7', 'checked_at': 0})
    original = update._cache_path().read_bytes()
    def handler(request):
        return httpx.Response(200, json={'releases': {'0.9': [{}]}})
    def fail_replace(self, target):
        raise OSError("read-only destination")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch.object(update.httpx, 'AsyncClient', return_value=client), patch.object(Path, 'replace', fail_replace):
        await update.refresh_version_cache()
    assert update._cache_path().read_bytes() == original
    assert list(update._cache_path().parent.iterdir()) == [update._cache_path()]
