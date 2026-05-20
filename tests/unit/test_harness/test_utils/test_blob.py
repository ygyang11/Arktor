"""Tests for the content-addressed blob store."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agent_harness.utils import blob as blob_module
from agent_harness.utils.blob import exists, get_bytes, make_attachment, put_bytes


@pytest.fixture(autouse=True)
def _isolate_blob_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(blob_module, "_BLOB_DIR", tmp_path / "blobs")


def test_put_bytes_returns_sha256_and_writes_file() -> None:
    data = b"hello world"
    digest = put_bytes(data)
    assert digest == hashlib.sha256(data).hexdigest()
    assert exists(digest)
    assert get_bytes(digest) == data


def test_put_bytes_is_idempotent_same_content() -> None:
    data = b"payload"
    d1 = put_bytes(data)
    d2 = put_bytes(data)
    assert d1 == d2
    assert get_bytes(d1) == data


def test_put_bytes_dedup_distinct_contents() -> None:
    a = put_bytes(b"AAAA")
    b = put_bytes(b"BBBB")
    assert a != b
    assert get_bytes(a) == b"AAAA"
    assert get_bytes(b) == b"BBBB"


def test_exists_false_when_missing() -> None:
    assert exists("0" * 64) is False


def test_make_attachment_writes_blob_and_returns_ref() -> None:
    data = b"\x89PNG\r\n\x1a\n" + b"X" * 32
    att = make_attachment(data, "image/png", "p.png")
    assert att.digest == hashlib.sha256(data).hexdigest()
    assert att.mime == "image/png"
    assert att.size == len(data)
    assert att.filename == "p.png"
    assert get_bytes(att.digest) == data


def test_make_attachment_no_filename_defaults_to_none() -> None:
    att = make_attachment(b"x", "image/png")
    assert att.filename is None
