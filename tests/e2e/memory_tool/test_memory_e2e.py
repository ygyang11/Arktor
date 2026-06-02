"""Memory tool e2e validation — realistic work scenarios.

Usage:
    python tests/e2e/memory_tool/test_memory_e2e.py [scenario]

Scenarios: mixed, research, noise, all
"""
from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.e2e
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent_app.tools import FILESYSTEM_TOOLS, MEMORY_TOOLS, TERMINAL_TOOLS, WEB_TOOLS, PAPER_TOOLS
from agent_harness import HarnessConfig, ReActAgent
from agent_harness.tool.base import BaseTool


TOOLS: list[BaseTool] = [*FILESYSTEM_TOOLS, *TERMINAL_TOOLS, *WEB_TOOLS, *PAPER_TOOLS, *MEMORY_TOOLS]


def _memory_dir(scope: str = "project") -> Path:
    if scope == "global":
        return Path.home() / ".arktor" / "memory"
    return Path.cwd() / ".arktor" / "memory"


def _cleanup() -> None:
    for scope in ("project", "global"):
        d = _memory_dir(scope)
        if d.exists():
            shutil.rmtree(d)


def _inspect_tool_calls(result: object) -> list[dict[str, object]]:
    calls = []
    for step in result.steps:
        if step.action:
            for tc in step.action:
                if tc.name == "memory_tool":
                    args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments or "{}")
                    calls.append(args)
    return calls


def _inspect_all_memories() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for scope in ("project", "global"):
        idx = _memory_dir(scope) / "MEMORY.md"
        if idx.exists():
            content = idx.read_text(encoding="utf-8")
            entries = [l.strip() for l in content.splitlines() if l.strip().startswith("- [")]
            if entries:
                out[scope] = entries
    return out


def _report_turn(label: str, result: object) -> None:
    calls = _inspect_tool_calls(result)
    print(f"\n--- {label} ---")
    print(f"Steps: {result.step_count}")
    if calls:
        for c in calls:
            print(f"  MEMORY: action={c.get('action')} scope={c.get('scope')} "
                  f"type={c.get('type')} name={c.get('name')}")
            if c.get("description"):
                print(f"    desc: {c['description'][:120]}")
    else:
        print("  (no memory_tool calls)")


def _report_final() -> None:
    memories = _inspect_all_memories()
    print(f"\n{'='*60}")
    print("FINAL MEMORY STATE")
    print(f"{'='*60}")
    if not memories:
        print("  (empty)")
    for scope, entries in memories.items():
        print(f"\n  [{scope.upper()}]")
        for e in entries:
            print(f"    {e[:120]}")


async def test_mixed() -> None:
    """Scenario 1: Realistic mixed work session (10 turns)."""
    _cleanup()
    config = HarnessConfig.load(PROJECT_ROOT / "config.yaml")
    agent = ReActAgent(name="mixed_test", tools=TOOLS, max_steps=15, config=config)

    print(f"\n{'='*60}")
    print("SCENARIO 1: Mixed work session")
    print(f"{'='*60}")

    turns = [
        ("Turn 1: regular task", "Show me the project version in pyproject.toml"),
        ("Turn 2: regular task", "What does the README say? Give me a brief summary"),
        ("Turn 3: regular task", "Read the first 50 lines of src/agent_harness/agent/base.py"),
        ("Turn 4: user info (expect save)", "I work in wireless communications, these agent concepts are new to me. Use communication system analogies when explaining"),
        ("Turn 5: regular task", "What configuration options are in config.yaml?"),
        ("Turn 6: project info (expect save)", "Our team does code review on Wednesdays and releases on Fridays, so large changes should have PRs ready by Monday or Tuesday"),
        ("Turn 7: regular task", "Run pytest tests/unit/ -x -q"),
        ("Turn 8: feedback (expect save)", "Your replies are too long. Keep them concise from now on"),
        ("Turn 9: reference (expect save)", "By the way, the Slack channel #infra-alerts has production issue notifications. Keep that in mind"),
        ("Turn 10: small talk (no save)", "Alright, that's it for today. Thanks"),
    ]

    for label, prompt in turns:
        r = await agent.run(prompt)
        _report_turn(label, r)

    _report_final()


async def test_research() -> None:
    """Scenario 2: Deep research — test proactive knowledge save."""
    _cleanup()
    config = HarnessConfig.load(PROJECT_ROOT / "config.yaml")
    agent = ReActAgent(name="research_test", tools=TOOLS, max_steps=20, config=config)

    print(f"\n{'='*60}")
    print("SCENARIO 2: Deep research session")
    print(f"{'='*60}")

    turns = [
        ("Turn 1: start research", "We need to pick a TCP congestion control algorithm for a satellite link. Research the mainstream options for me"),
        ("Turn 2: deep comparison", "Compare BBR, CUBIC, and Copa specifically — their pros and cons in high-latency scenarios"),
        ("Turn 3: find papers", "Are there papers specifically about congestion control for satellite links? Search for them"),
        ("Turn 4: synthesize (expect proactive save)", "Based on all the research so far, which algorithm is best for our satellite link scenario? Give me a conclusion"),
        ("Turn 5: switch to task (no save)", "OK, now help me look at the networking-related code in this project"),
    ]

    for label, prompt in turns:
        r = await agent.run(prompt)
        _report_turn(label, r)

    _report_final()


async def test_noise() -> None:
    """Scenario 3: Noise resistance — daily work with few save-worthy items."""
    _cleanup()
    config = HarnessConfig.load(PROJECT_ROOT / "config.yaml")
    agent = ReActAgent(name="noise_test", tools=TOOLS, max_steps=15, config=config)

    print(f"\n{'='*60}")
    print("SCENARIO 3: Noise resistance")
    print(f"{'='*60}")

    turns = [
        ("Turn 1: small talk (no save)", "Good morning"),
        ("Turn 2: regular task", "Run pytest tests/unit/ -x -q for me"),
        ("Turn 3: regular task", "Did any tests fail? Show me the last part of the output"),
        ("Turn 4: user info (expect save)", "I have 3 years of embedded development experience, strong in low-level C and assembly. Picked up Python later"),
        ("Turn 5: noise (no save)", "These tests are a bit slow"),
        ("Turn 6: regular task", "Show me the implementation of src/agent_harness/tool/executor.py"),
        ("Turn 7: regular task", "List all TODO comments in that file"),
        ("Turn 8: reference (expect save)", "Our internal docs are centrally stored in the Notion Engineering space"),
        ("Turn 9: small talk (no save)", "Thanks, that's all for today"),
    ]

    for label, prompt in turns:
        r = await agent.run(prompt)
        _report_turn(label, r)

    _report_final()


SCENARIOS = {
    "mixed": test_mixed,
    "research": test_research,
    "noise": test_noise,
}


async def main() -> None:
    scenario = sys.argv[1] if len(sys.argv) > 1 else "all"
    if scenario == "all":
        for fn in SCENARIOS.values():
            await fn()
    elif scenario in SCENARIOS:
        await SCENARIOS[scenario]()
    else:
        print(f"Unknown: {scenario}. Available: {', '.join(SCENARIOS.keys())}, all")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
