"""Approval handlers — abstract interface and stdin default implementation."""
from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from typing import Any

from agent_harness.approval.types import ApprovalDecision, ApprovalRequest, ApprovalResult
from agent_harness.utils.theme import COLORS, ICONS


class ApprovalHandler(ABC):
    """Abstract interface for getting approval decisions from a human."""

    @abstractmethod
    async def request_approval(self, request: ApprovalRequest) -> ApprovalResult: ...


class StdinApprovalHandler(ApprovalHandler):
    """Interactive stdin-based approval handler.

    Displays the tool call and reads user input:
      [Y]es / [A]lways / [N]o (default: Y)
    """

    def __init__(self, output: Any = None, color: bool = True) -> None:
        self._output = output or sys.stdout
        self._color = color and hasattr(self._output, "isatty") and self._output.isatty()

    def _c(self, name: str) -> str:
        return COLORS.get(name, "") if self._color else ""

    def _icon(self, name: str) -> str:
        return ICONS.get(name, "")

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResult:
        tc = request.tool_call
        bold, yellow, reset = self._c("bold"), self._c("yellow"), self._c("reset")
        marker, lock = self._icon("marker"), self._icon("approval")
        args_preview = ", ".join(f'{k}="{v}"' for k, v in tc.arguments.items())
        self._output.write(
            f"{yellow}{marker}{reset} {lock} {bold}{tc.name}{reset}({args_preview})\n"
        )
        label = self._always_label(request)
        self._output.write(f"  Allow? [Y]es / {label} / [N]o <reason> (default: Y): ")
        self._output.flush()

        from agent_harness.utils.input_mux import mux_input

        raw: str = await mux_input("", priority=10)
        choice = raw.strip().lower()

        reason: str | None = None
        if choice.startswith("n"):
            decision = ApprovalDecision.DENY
            # Extract reason after "n" or "no": "n too dangerous" → "too dangerous"
            rest = choice.split(None, 1)
            if len(rest) > 1:
                reason = rest[1]
        elif choice in ("a", "always"):
            decision = ApprovalDecision.ALLOW_SESSION
        else:
            decision = ApprovalDecision.ALLOW_ONCE

        return ApprovalResult(
            tool_call_id=tc.id, tool_name=tc.name, decision=decision, reason=reason,
        )

    def _always_label(self, request: ApprovalRequest) -> str:
        """Build context-aware label for the Always option."""
        from agent_harness.approval.policy import (  # noqa: PLC0415
            _CHAIN_RE,
            derive_session_prefix,
        )

        tc = request.tool_call
        r, k = request.resource, request.resource_kind
        if r is None or k is None:
            return f"[A]lways allow {tc.name} this session"

        if k == "command":
            segments = [s.strip() for s in _CHAIN_RE.split(r) if s.strip()]
            prefixes = (
                sorted({derive_session_prefix(s, k) for s in segments})
                if segments else [derive_session_prefix(r, k)]
            )
            label = ", ".join(f"'{p}'" for p in prefixes)
            return f"[A]lways allow {label} commands this session"

        prefix = derive_session_prefix(r, k)
        if k == "url":
            return f"[A]lways allow requests to '{prefix}' this session"
        if k == "path":
            normed = os.path.normpath(r)
            if prefix == normed and not os.path.isdir(normed):
                return f"[A]lways allow {tc.name} on '{prefix}' this session"
            return f"[A]lways allow {tc.name} under '{prefix}/' this session"
        return f"[A]lways allow {tc.name} this session"


class AutoApproveHandler(ApprovalHandler):
    """Non-interactive handler that allows every request without prompting.

    For headless or programmatic runs where no human can answer. Pair with
    policy mode ``never`` for fully unrestricted execution; on its own it
    auto-allows any request the policy escalates to ASK.
    """

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResult:
        tc = request.tool_call
        return ApprovalResult(
            tool_call_id=tc.id,
            tool_name=tc.name,
            decision=ApprovalDecision.ALLOW_ONCE,
        )
