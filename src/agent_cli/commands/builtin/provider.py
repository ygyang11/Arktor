"""/provider — list or switch LLM provider profiles from cli-prefs.json."""
from __future__ import annotations

from contextlib import suppress
from typing import Any

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import err, ok, render_provider_list
from agent_cli.runtime.prefs import read_prefs
from agent_harness.core.config import LLMConfig, SubModelConfig
from agent_harness.llm import BaseLLM, create_llm, create_sub_llm


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
        if k not in {"provider", "model", "base_url", "api_key", "sub_model"}
    }
    profile = dict(prof)
    raw_sub_model = profile.pop("sub_model", None)
    new_llm: BaseLLM | None = None
    new_sub_llm: BaseLLM | None = None
    try:
        profile_sub = SubModelConfig(model=raw_sub_model)
        if profile_sub.model is not None:
            current_sub = agent.context.config.llm.sub_model
            profile["sub_model"] = {
                "model": profile_sub.model,
                "reasoning_effort": (
                    current_sub.reasoning_effort
                    if current_sub is not None
                    else None
                ),
            }

        new_cfg = LLMConfig(**{**keep, **profile})
        new_llm = create_llm(new_cfg)
        new_sub_llm = create_sub_llm(new_cfg) or new_llm
    except Exception as e:
        created = {
            id(llm): llm
            for llm in (new_llm, new_sub_llm)
            if llm is not None
        }
        for llm in created.values():
            with suppress(Exception):
                await llm.aclose()
        return CommandResult(output=err(f"Failed to switch provider: {e}"))

    agent.context.config.llm = new_cfg
    agent.replace_llms(new_llm, new_sub_llm)

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
