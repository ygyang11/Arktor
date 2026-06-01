"""Cross-tool observability primitives (file freshness, future: git/env/...)."""

from agent_app.observability.file_freshness import (
    Drift,
    FileSignature,
    Verdict,
    mark_read,
    mark_seen,
    poll_drift,
    restore_state,
    snapshot_state,
    stale_guard,
)

__all__ = [
    "Drift",
    "FileSignature",
    "Verdict",
    "mark_read",
    "mark_seen",
    "poll_drift",
    "restore_state",
    "snapshot_state",
    "stale_guard",
]
