"""Approval module — human-in-the-loop tool execution approval."""
from agent_harness.approval.handler import (
    ApprovalHandler,
    AutoApproveHandler,
    StdinApprovalHandler,
)
from agent_harness.approval.policy import ApprovalPolicy
from agent_harness.approval.types import (
    ApprovalAction,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResult,
)
from agent_harness.core.config import HarnessConfig


def resolve_approval(
    approval: ApprovalPolicy | None,
    config: HarnessConfig | None = None,
) -> ApprovalPolicy:
    """Resolve approval parameter to an ApprovalPolicy instance.

    Priority: explicit parameter > config.yaml > default.
    """
    if isinstance(approval, ApprovalPolicy):
        return approval
    cfg = (config or HarnessConfig.get()).approval
    return ApprovalPolicy(
        mode=cfg.mode,
        always_allow=set(cfg.always_allow),
        always_deny=set(cfg.always_deny),
    )


def resolve_approval_handler(
    handler: ApprovalHandler | None,
) -> ApprovalHandler:
    """Resolve approval handler. Defaults to StdinApprovalHandler."""
    if handler is not None:
        return handler
    return StdinApprovalHandler()


__all__ = [
    "ApprovalAction",
    "ApprovalDecision",
    "ApprovalHandler",
    "ApprovalPolicy",
    "ApprovalRequest",
    "ApprovalResult",
    "AutoApproveHandler",
    "StdinApprovalHandler",
    "resolve_approval",
    "resolve_approval_handler",
]
