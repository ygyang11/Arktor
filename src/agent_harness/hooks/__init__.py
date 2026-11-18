"""Hooks module — lifecycle hooks for agent execution."""
from agent_harness.core.config import HarnessConfig
from agent_harness.hooks.base import DefaultHooks
from agent_harness.hooks.composite import CompositeHooks
from agent_harness.hooks.progress import ProgressHooks
from agent_harness.hooks.recording import RecordingHooks
from agent_harness.hooks.tracing import TracingHooks


def resolve_hooks(
    hooks: DefaultHooks | None,
    config: HarnessConfig | None,
) -> DefaultHooks:
    cfg = config or HarnessConfig.get()

    if hooks is not None:
        return hooks

    if cfg.tracing.enabled:
        cached = cfg.get_runtime_hooks()
        if cached is not None:
            return cached
        created = TracingHooks(
            trace_dir=cfg.tracing.export_path,
            exporter=cfg.tracing.exporter,
        )
        cfg.set_runtime_hooks(created)
        return created

    return ProgressHooks()


__all__ = [
    "DefaultHooks",
    "TracingHooks",
    "ProgressHooks",
    "CompositeHooks",
    "RecordingHooks",
    "resolve_hooks",
]
