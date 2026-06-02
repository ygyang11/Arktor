"""/provider — list or switch LLM provider profiles from cli-prefs.json."""
from __future__ import annotations

from typing import Any

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import err, ok, render_provider_list
from agent_cli.runtime import session as sess
from agent_cli.runtime.prefs import read_prefs
from agent_harness.core.config import LLMConfig
from agent_harness.llm import create_llm


def _active_profile(profiles: dict[str, Any], current: LLMConfig) -> str | None:
    if current.base_url is None:
        return None
    for name, prof in profiles.items():
        if prof.get("base_url") == current.base_url:
            return name
    return None


async def handle(ctx: CommandContext, args: str) -> CommandResult:
    name = args.strip()
    agent = ctx.agent
    raw = read_prefs().get("llm_profiles", {})
    profiles: dict[str, Any] = (
        {k: v for k, v in raw.items() if isinstance(v, dict)}
        if isinstance(raw, dict) else {}
    )

    if not name:
        active = _active_profile(profiles, agent.context.config.llm)
        return CommandResult(output=render_provider_list(profiles, active))

    prof = profiles.get(name)
    if not isinstance(prof, dict):
        avail = ", ".join(profiles) or "none"
        return CommandResult(output=err(
            f"Unknown provider: {name}",
            (f" · Available: {avail}", "muted"),
        ))

    unknown = set(prof) - set(LLMConfig.model_fields)
    if unknown:
        return CommandResult(output=err(
            f"Profile '{name}' has unknown field(s): {', '.join(sorted(unknown))}",
        ))

    keep = {
        k: v for k, v in agent.context.config.llm.model_dump().items()
        if k not in {"provider", "model", "base_url", "api_key"}
    }
    try:
        new_cfg = LLMConfig(**{**keep, **prof})
        new_llm = create_llm(new_cfg)
    except Exception as e:
        return CommandResult(output=err(f"Failed to switch provider: {e}"))

    new_llm.set_event_bus(agent.context.event_bus)
    agent.llm = new_llm
    agent.context.short_term_memory.model = new_cfg.model
    agent.context.short_term_memory.clear_call_snapshot()
    sess.update_compressor_model(agent, new_cfg.model, new_llm)
    agent.context.config.llm = new_cfg

    return CommandResult(output=ok(
        "Provider Switched to ",
        (name, "bold"),
        (f" · {new_cfg.provider} {new_cfg.model}", "muted"),
    ))


CMD = Command(
    name="/provider",
    description="List or switch LLM provider for this session",
    handler=handle,
)
