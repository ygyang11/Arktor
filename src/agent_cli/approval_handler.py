"""CliApprovalHandler — CLI approval entry point; routes fg / bg requests."""
from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markup import escape as rich_escape
from rich.panel import Panel
from rich.text import Text

from agent_cli.render.tool_display import _display_name
from agent_cli.runtime import background
from agent_cli.runtime import session as sess_rt
from agent_cli.theme import APPROVAL, SEP_DOT
from agent_harness.agent.base import BaseAgent
from agent_harness.approval.handler import ApprovalHandler
from agent_harness.approval.policy import (
    ApprovalPolicy,
    _UNSAFE_SHELL_RE,
    derive_session_prefix,
)
from agent_harness.approval.types import ApprovalDecision, ApprovalRequest, ApprovalResult

if TYPE_CHECKING:
    from agent_cli.adapter import CliAdapter

_COMMAND_CHAIN_RE = re.compile(r"\s*(?:&&|\|\||[;|])\s*")
_DENY_REASON_SEPARATORS = ",，:：;；"
_PROMPT_TEXT = HTML(
    f"Allow? <b>[Y]</b>es {SEP_DOT} <b>[A]</b>lways {SEP_DOT} "
    "<b>[N]</b>o &lt;reason&gt; (default: Y): "
)
_PROMPT_TEXT_NO_ALWAYS = HTML(
    f"Allow? <b>[Y]</b>es {SEP_DOT} "
    "<b>[N]</b>o &lt;reason&gt; (default: Y): "
)


@dataclass
class _PendingApproval:
    request: ApprovalRequest
    future: asyncio.Future[ApprovalResult]


class CliApprovalHandler(ApprovalHandler):
    def __init__(
        self,
        console: Console | None = None,
        adapter: CliAdapter | None = None,
        pt_session: PromptSession[str] | None = None,
    ) -> None:
        self._console = console or Console()
        self._adapter = adapter
        self._agent_ref: BaseAgent | None = None
        self._pending: asyncio.Queue[_PendingApproval] = asyncio.Queue()
        self._console_lock: asyncio.Lock = (
            adapter.lock() if adapter is not None else asyncio.Lock()
        )
        self._pt_session: PromptSession[str] = (
            pt_session if pt_session is not None else PromptSession()
        )
        self._approval_history: InMemoryHistory = InMemoryHistory()

    def bind_agent(self, agent: BaseAgent) -> None:
        self._agent_ref = agent

    def is_in_background_task(self) -> bool:
        if self._agent_ref is None:
            return False
        return background.is_current_task_background(self._agent_ref)

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResult:
        if self.is_in_background_task():
            return await self._handle_background(request)
        return await self._handle_foreground(request)

    def pending_queue(self) -> asyncio.Queue[_PendingApproval]:
        return self._pending

    def cancel_pending(self) -> None:
        """Drain queue and cancel each pending future."""
        while True:
            try:
                pending = self._pending.get_nowait()
            except asyncio.QueueEmpty:
                return
            if not pending.future.done():
                pending.future.cancel()

    async def resolve_pending(self, pending: _PendingApproval) -> None:
        try:
            result = await self._prompt_user(pending.request)
        except KeyboardInterrupt:
            result = ApprovalResult(
                tool_call_id=pending.request.tool_call.id,
                tool_name=pending.request.tool_call.name,
                decision=ApprovalDecision.DENY,
                reason="The user refused to execute the tool this time (interrupted by Ctrl+C).",
            )
        except EOFError:
            result = ApprovalResult(
                tool_call_id=pending.request.tool_call.id,
                tool_name=pending.request.tool_call.name,
                decision=ApprovalDecision.DENY,
                reason="The user refused to execute the tool this time (interrupted by Ctrl+D).",
            )
        except asyncio.CancelledError:
            if not pending.future.done():
                pending.future.set_exception(asyncio.CancelledError())
            raise
        except BaseException as e:
            if not pending.future.done():
                pending.future.set_exception(e)
            raise
        if not pending.future.done():
            pending.future.set_result(result)

    async def _handle_foreground(self, request: ApprovalRequest) -> ApprovalResult:
        try:
            return await self._prompt_user(request)
        except KeyboardInterrupt:
            raise asyncio.CancelledError(
                "The user refused to execute the tool this time (interrupted by Ctrl+C)."
            ) from None
        except EOFError:
            raise asyncio.CancelledError(
                "The user refused to execute the tool this time (interrupted by Ctrl+D)."
            ) from None

    async def _prompt_user(self, request: ApprovalRequest) -> ApprovalResult:
        async with self._console_lock:
            if self._adapter is not None:
                await self._adapter.pause_for_stdin()
            tc = request.tool_call

            display = _display_name(tc.name)
            if request.resource:
                main_content = (
                    f"[bold][accent]{APPROVAL}[/accent][/bold] "
                    f"[bold]{rich_escape(display)}[/bold]\n"
                    f"  {rich_escape(request.resource)}"
                )
            else:
                # Tools without approval_resource_key (sub_agent, memory_tool, ...)
                main_content = (
                    f"[bold][accent]{APPROVAL}[/accent][/bold] "
                    f"[bold]{rich_escape(display)}[/bold]"
                )

            assert self._agent_ref is not None
            can_always = self._can_grant_session(
                request, sess_rt.get_policy(self._agent_ref),
            )
            if can_always:
                always_sentence = rich_escape(self._always_label(request))
                panel_body = Text.from_markup(
                    f"{main_content}\n"
                    f"\n"
                    f"[muted]{always_sentence}[/muted]"
                )
            else:
                panel_body = Text.from_markup(main_content)
            self._console.print(
                Panel(
                    panel_body,
                    title=" Approval needed ",
                    title_align="left",
                    border_style="accent",
                    expand=False,
                    padding=(0, 1),
                )
            )

            # prompt_toolkit persists key_bindings / bottom_toolbar / completer
            # back onto the shared session. Snapshot and restore so the main
            # REPL's Ctrl+C semantics, status bar, and slash completer do not
            # leak into this approval prompt.
            saved = {
                "history":        self._pt_session.history,
                "key_bindings":   self._pt_session.key_bindings,
                "bottom_toolbar": self._pt_session.bottom_toolbar,
                "completer":      self._pt_session.completer,
            }
            self._pt_session.history = self._approval_history
            self._pt_session.key_bindings = None
            self._pt_session.bottom_toolbar = None
            self._pt_session.completer = None
            try:
                raw = await self._pt_session.prompt_async(
                    _PROMPT_TEXT if can_always else _PROMPT_TEXT_NO_ALWAYS,
                    set_exception_handler=False,
                )
            finally:
                for attr, value in saved.items():
                    setattr(self._pt_session, attr, value)
                # prompt_async(handle_sigint=True) removed our SIGINT handler;
                # restore task-bound (if inside bind_work) or idle no-op.
                from agent_cli.runtime.sigint import restore_current
                restore_current()
                self._console.print()
            return self._parse_answer(raw, request, allow_session=can_always)

    @staticmethod
    def _can_grant_session(
        request: ApprovalRequest, policy: ApprovalPolicy,
    ) -> bool:
        """False when policy will not honor a prefix grant for this resource.

        Mirrors policy._check_command's unsafe-shell fallback: heredocs,
        redirects, backticks, newlines etc. force per-call ASK regardless
        of session grants — so offering [A]lways would be a lie. Also
        false outside `auto` mode where session grants are disabled.
        """
        if not policy.allows_session_grants():
            return False
        if request.resource_kind == "command" and request.resource:
            return not _UNSAFE_SHELL_RE.search(request.resource)
        return True

    @staticmethod
    def _always_label(request: ApprovalRequest) -> str:
        tc = request.tool_call
        r, k = request.resource, request.resource_kind
        if r is None or k is None:
            return "[A]lways allow this session"

        if k == "command":
            segments = [s.strip() for s in _COMMAND_CHAIN_RE.split(r) if s.strip()]
            if segments:
                prefixes = sorted({derive_session_prefix(s, k) for s in segments})
            else:
                prefixes = [derive_session_prefix(r, k)]
            labels = ", ".join(f"'{p}'" for p in prefixes)
            return f"[A]lways allow {labels} commands this session"

        prefix = derive_session_prefix(r, k)
        display = _display_name(tc.name)
        if k == "url":
            return f"[A]lways allow requests to '{prefix}' this session"
        if k == "path":
            parent = os.path.dirname(os.path.normpath(r))
            if parent:
                return f"[A]lways allow {display} under '{prefix}/' this session"
            return f"[A]lways allow {display} on '{prefix}' this session"
        return f"[A]lways allow {display} this session"

    async def _handle_background(self, request: ApprovalRequest) -> ApprovalResult:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[ApprovalResult] = loop.create_future()
        await self._pending.put(_PendingApproval(request=request, future=fut))
        return await fut

    @staticmethod
    def _parse_answer(
        raw: str,
        request: ApprovalRequest,
        *,
        allow_session: bool = True,
    ) -> ApprovalResult:
        tc = request.tool_call
        choice = raw.strip()
        reason: str | None = None
        lowered = choice.casefold()

        if lowered in ("a", "always"):
            decision = (
                ApprovalDecision.ALLOW_SESSION if allow_session
                else ApprovalDecision.ALLOW_ONCE
            )
        else:
            decision = ApprovalDecision.ALLOW_ONCE
            for prefix in ("no", "n"):
                plen = len(prefix)
                if choice[:plen].casefold() != prefix:
                    continue
                if len(choice) == plen:
                    decision = ApprovalDecision.DENY
                    break

                next_char = choice[plen]
                if next_char.isspace():
                    decision = ApprovalDecision.DENY
                    reason = choice[plen:].strip() or None
                    break
                if next_char in _DENY_REASON_SEPARATORS:
                    decision = ApprovalDecision.DENY
                    reason = choice[plen + 1:].strip() or None
                    break

        return ApprovalResult(
            tool_call_id=tc.id,
            tool_name=tc.name,
            decision=decision,
            reason=reason,
        )
