"""Unit tests for shared backend HTTP plumbing."""
from __future__ import annotations

import json

import pytest

from agent_app.tools.document_parser.backends.base import (
    BackendHTTPContext,
    decode_envelope,
)
from agent_app.tools.document_parser.errors import (
    DocumentBackendError,
    DocumentErrorClass,
    classify_paddle,
)
from agent_harness.utils.http_retry import HttpRetryConfig


def _ctx() -> BackendHTTPContext:
    return BackendHTTPContext(
        classify=classify_paddle,
        retry=HttpRetryConfig(max_attempts=1, base_delay=0),
        request_timeout_s=10,
        upload_timeout_s=10,
        download_timeout_s=10,
        backend_label="PaddleOCR",
    )


class TestDecodeEnvelope:
    def test_200_dict_returns_body(self) -> None:
        out = decode_envelope(200, '{"code": 0, "data": {"x": 1}}', ctx=_ctx())
        assert out == {"code": 0, "data": {"x": 1}}

    def test_200_non_json_raises_unknown(self) -> None:
        with pytest.raises(DocumentBackendError) as ei:
            decode_envelope(200, "not json", ctx=_ctx())
        assert ei.value.error_class is DocumentErrorClass.UNKNOWN

    def test_200_non_object_raises_unknown(self) -> None:
        with pytest.raises(DocumentBackendError) as ei:
            decode_envelope(200, '["not", "an", "object"]', ctx=_ctx())
        assert ei.value.error_class is DocumentErrorClass.UNKNOWN

    def test_paddle_400_body_code_10004_classified_as_unsupported(self) -> None:
        """PaddleOCR returns HTTP 400 with body `{"code": 10004, "msg": ...}`
        for unsupported formats (e.g. arxiv TeX-PDFs). The body `code` must
        win over the raw HTTP status so the pipeline sees the accurate
        UNSUPPORTED_BY_TIER class (which triggers URL→local fallback)
        instead of the generic UNKNOWN from `classify_paddle(400)`.
        """
        body = json.dumps({
            "traceId": "abc", "code": 10004, "msg": "文件格式不支持",
        })
        with pytest.raises(DocumentBackendError) as ei:
            decode_envelope(400, body, ctx=_ctx())
        assert ei.value.error_class is DocumentErrorClass.UNSUPPORTED_BY_TIER
        assert ei.value.code == 10004

    def test_paddle_400_body_code_10002_classified_as_fetch_failed(self) -> None:
        body = json.dumps({
            "traceId": "abc", "code": 10002, "msg": "文件 URL 无法识别",
        })
        with pytest.raises(DocumentBackendError) as ei:
            decode_envelope(400, body, ctx=_ctx())
        assert ei.value.error_class is DocumentErrorClass.BACKEND_FETCH_FAILED
        assert ei.value.code == 10002

    def test_non_200_without_json_body_falls_back_to_http_status(self) -> None:
        with pytest.raises(DocumentBackendError) as ei:
            decode_envelope(503, "Service Unavailable", ctx=_ctx())
        # 503 in paddle table maps to RATE_LIMITED
        assert ei.value.error_class is DocumentErrorClass.RATE_LIMITED
        assert ei.value.code == 503

    def test_non_200_with_json_body_but_no_code_falls_back_to_status(self) -> None:
        with pytest.raises(DocumentBackendError) as ei:
            decode_envelope(401, '{"traceId": "x"}', ctx=_ctx())
        assert ei.value.error_class is DocumentErrorClass.AUTH_FAILED
        assert ei.value.code == 401

    def test_non_200_with_zero_code_in_body_falls_back_to_status(self) -> None:
        """A `code: 0` in body should NOT override HTTP error status —
        zero is paddle's success marker, semantically meaningless on a
        non-200 envelope."""
        with pytest.raises(DocumentBackendError) as ei:
            decode_envelope(500, '{"code": 0, "msg": "weird"}', ctx=_ctx())
        # 500 maps to BACKEND_TRANSIENT_ERROR per paddle table
        assert ei.value.error_class is DocumentErrorClass.BACKEND_TRANSIENT_ERROR
        assert ei.value.code == 500
