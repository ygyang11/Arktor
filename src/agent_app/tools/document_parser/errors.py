"""Document parser error taxonomy and pipeline-level exceptions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from agent_harness.core.errors import HarnessError


class DocumentErrorClass(str, Enum):
    AUTH_FAILED = "AUTH_FAILED"
    INVALID_INPUT = "INVALID_INPUT"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    PAGE_LIMIT = "PAGE_LIMIT"
    RATE_LIMITED = "RATE_LIMITED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    BACKEND_TRANSIENT_ERROR = "BACKEND_TRANSIENT_ERROR"
    BACKEND_FETCH_FAILED = "BACKEND_FETCH_FAILED"
    BACKEND_READ_FAILED = "BACKEND_READ_FAILED"
    TIMEOUT = "TIMEOUT"
    UNSUPPORTED_BY_TIER = "UNSUPPORTED_BY_TIER"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    IO_ERROR = "IO_ERROR"
    UNKNOWN = "UNKNOWN"


class DocumentBackendError(HarnessError):
    def __init__(
        self, cls: DocumentErrorClass, code: Any, message: str,
    ) -> None:
        super().__init__(f"[{cls.value}] code={code}: {message}")
        self.error_class = cls
        self.code = code
        self.message = message


_MINERU_CODES: dict[Any, DocumentErrorClass] = {
    429: DocumentErrorClass.RATE_LIMITED,
    "A0202": DocumentErrorClass.AUTH_FAILED,
    "A0211": DocumentErrorClass.AUTH_FAILED,
    -500: DocumentErrorClass.INVALID_INPUT,
    -10001: DocumentErrorClass.BACKEND_TRANSIENT_ERROR,
    -10002: DocumentErrorClass.INVALID_INPUT,
    -30001: DocumentErrorClass.FILE_TOO_LARGE,
    -30002: DocumentErrorClass.UNSUPPORTED_BY_TIER,
    -30003: DocumentErrorClass.PAGE_LIMIT,
    -30004: DocumentErrorClass.INVALID_INPUT,
    -60001: DocumentErrorClass.BACKEND_TRANSIENT_ERROR,
    -60002: DocumentErrorClass.INVALID_INPUT,
    -60003: DocumentErrorClass.BACKEND_READ_FAILED,
    -60004: DocumentErrorClass.INVALID_INPUT,
    -60005: DocumentErrorClass.FILE_TOO_LARGE,
    -60006: DocumentErrorClass.PAGE_LIMIT,
    -60007: DocumentErrorClass.BACKEND_TRANSIENT_ERROR,
    -60008: DocumentErrorClass.BACKEND_FETCH_FAILED,
    -60009: DocumentErrorClass.RATE_LIMITED,
    -60010: DocumentErrorClass.BACKEND_TRANSIENT_ERROR,
    -60011: DocumentErrorClass.BACKEND_READ_FAILED,
    -60012: DocumentErrorClass.INVALID_INPUT,
    -60013: DocumentErrorClass.AUTH_FAILED,
    -60018: DocumentErrorClass.QUOTA_EXCEEDED,
    -60019: DocumentErrorClass.QUOTA_EXCEEDED,
    -60022: DocumentErrorClass.BACKEND_FETCH_FAILED,
}

_PADDLE_CODES: dict[Any, DocumentErrorClass] = {
    401: DocumentErrorClass.AUTH_FAILED,
    403: DocumentErrorClass.AUTH_FAILED,
    413: DocumentErrorClass.FILE_TOO_LARGE,
    422: DocumentErrorClass.INVALID_INPUT,
    429: DocumentErrorClass.QUOTA_EXCEEDED,
    500: DocumentErrorClass.BACKEND_TRANSIENT_ERROR,
    503: DocumentErrorClass.RATE_LIMITED,
    504: DocumentErrorClass.TIMEOUT,
    10001: DocumentErrorClass.INVALID_INPUT,
    10002: DocumentErrorClass.BACKEND_FETCH_FAILED,
    10003: DocumentErrorClass.FILE_TOO_LARGE,
    10004: DocumentErrorClass.UNSUPPORTED_BY_TIER,
    10005: DocumentErrorClass.BACKEND_READ_FAILED,
    10006: DocumentErrorClass.PAGE_LIMIT,
    10007: DocumentErrorClass.INVALID_INPUT,
    10008: DocumentErrorClass.INVALID_INPUT,
    10009: DocumentErrorClass.QUOTA_EXCEEDED,
    10010: DocumentErrorClass.RATE_LIMITED,
    11001: DocumentErrorClass.INVALID_INPUT,
    11002: DocumentErrorClass.INVALID_INPUT,
    11003: DocumentErrorClass.BACKEND_READ_FAILED,
    12001: DocumentErrorClass.QUOTA_EXCEEDED,
    12002: DocumentErrorClass.RATE_LIMITED,
}


def classify_mineru(code: Any) -> DocumentErrorClass:
    return _MINERU_CODES.get(code, DocumentErrorClass.UNKNOWN)


def classify_paddle(code: Any) -> DocumentErrorClass:
    return _PADDLE_CODES.get(code, DocumentErrorClass.UNKNOWN)


@dataclass
class TierAttempt:
    tier: str
    mode: str
    error_class: str
    error_message: str
    elapsed_ms: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NoViableDocumentBackend(HarnessError):
    def __init__(
        self,
        skipped: list[dict[str, str]],
        fallback_chain: list[TierAttempt] | None = None,
        unattempted: list[str] | None = None,
    ) -> None:
        super().__init__(
            f"no viable document backend "
            f"(tried={len(fallback_chain or [])}, "
            f"skipped={len(skipped)}, "
            f"unattempted={len(unattempted or [])})"
        )
        self.skipped = skipped
        self.fallback_chain = fallback_chain or []
        self.unattempted = unattempted or []
