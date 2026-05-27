"""Unit tests for document_parser.errors."""
from __future__ import annotations

from agent_app.tools.document_parser.errors import (
    DocumentBackendError,
    DocumentErrorClass,
    NoViableDocumentBackend,
    TierAttempt,
    classify_mineru,
    classify_paddle,
)


class TestClassify:
    def test_mineru_known_codes(self) -> None:
        assert classify_mineru(-30001) is DocumentErrorClass.FILE_TOO_LARGE
        assert classify_mineru(-30002) is DocumentErrorClass.UNSUPPORTED_BY_TIER
        assert classify_mineru(-30003) is DocumentErrorClass.PAGE_LIMIT
        assert classify_mineru(-30004) is DocumentErrorClass.INVALID_INPUT
        assert classify_mineru(-60005) is DocumentErrorClass.FILE_TOO_LARGE
        assert classify_mineru(-60006) is DocumentErrorClass.PAGE_LIMIT
        assert classify_mineru(-60013) is DocumentErrorClass.AUTH_FAILED
        assert classify_mineru(-10002) is DocumentErrorClass.INVALID_INPUT
        assert classify_mineru("A0202") is DocumentErrorClass.AUTH_FAILED

    def test_mineru_unknown_code(self) -> None:
        assert classify_mineru(9999) is DocumentErrorClass.UNKNOWN
        assert classify_mineru(None) is DocumentErrorClass.UNKNOWN

    def test_paddle_known_codes(self) -> None:
        assert classify_paddle(401) is DocumentErrorClass.AUTH_FAILED
        assert classify_paddle(403) is DocumentErrorClass.AUTH_FAILED
        assert classify_paddle(413) is DocumentErrorClass.FILE_TOO_LARGE
        assert classify_paddle(429) is DocumentErrorClass.QUOTA_EXCEEDED
        assert classify_paddle(503) is DocumentErrorClass.RATE_LIMITED
        assert classify_paddle(504) is DocumentErrorClass.TIMEOUT
        assert classify_paddle(10003) is DocumentErrorClass.FILE_TOO_LARGE
        assert classify_paddle(10004) is DocumentErrorClass.UNSUPPORTED_BY_TIER
        assert classify_paddle(10005) is DocumentErrorClass.BACKEND_READ_FAILED
        assert classify_paddle(10006) is DocumentErrorClass.PAGE_LIMIT
        assert classify_paddle(11003) is DocumentErrorClass.BACKEND_READ_FAILED
        assert classify_paddle(12001) is DocumentErrorClass.QUOTA_EXCEEDED
        assert classify_paddle(12002) is DocumentErrorClass.RATE_LIMITED

    def test_paddle_unknown_code(self) -> None:
        assert classify_paddle(0xBEEF) is DocumentErrorClass.UNKNOWN


class TestBackendError:
    def test_str_contains_class_and_code(self) -> None:
        e = DocumentBackendError(DocumentErrorClass.AUTH_FAILED, 401, "bad token")
        assert "[AUTH_FAILED]" in str(e)
        assert "code=401" in str(e)
        assert "bad token" in str(e)
        assert e.error_class is DocumentErrorClass.AUTH_FAILED
        assert e.code == 401
        assert e.message == "bad token"


class TestTierAttempt:
    def test_to_dict_keys_stable(self) -> None:
        a = TierAttempt(
            tier="paddleocr-vl-1.5",
            mode="url",
            error_class="INVALID_INPUT",
            error_message="code 10004",
            elapsed_ms=1234,
        )
        d = a.to_dict()
        assert d == {
            "tier": "paddleocr-vl-1.5",
            "mode": "url",
            "error_class": "INVALID_INPUT",
            "error_message": "code 10004",
            "elapsed_ms": 1234,
        }


class TestNoViableBackend:
    def test_defaults(self) -> None:
        e = NoViableDocumentBackend([{"tier": "x", "reason": "no_api_key:mineru"}])
        assert e.skipped == [{"tier": "x", "reason": "no_api_key:mineru"}]
        assert e.fallback_chain == []
        assert e.unattempted == []

    def test_full_args(self) -> None:
        chain = [
            TierAttempt(
                tier="paddleocr-vl-1.5", mode="url",
                error_class="INVALID_INPUT", error_message="code 10004",
                elapsed_ms=42,
            ),
        ]
        e = NoViableDocumentBackend(
            skipped=[{"tier": "mineru-lightweight", "reason": "size>10MB(url)"}],
            fallback_chain=chain,
            unattempted=["mineru-vlm"],
        )
        assert e.fallback_chain[0].tier == "paddleocr-vl-1.5"
        assert e.unattempted == ["mineru-vlm"]
