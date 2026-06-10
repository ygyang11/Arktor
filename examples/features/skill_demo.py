"""Skill demo — show how to use skill_tool for text rewriting.

Usage:
    python examples/skill_demo.py
"""

import asyncio
from pathlib import Path

from agent_harness import HarnessConfig, ReActAgent
from agent_app.tools import SKILL_TOOLS, skill_tool

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def print_skill_catalog_preview() -> bool:
    schema = skill_tool.get_schema()
    print("=== Skill catalog preview ===")
    print(f"Tool: {schema.name}")
    print(schema.description)
    print()
    return "(no skills available)" not in schema.description


def build_rewrite_query() -> str:
    return (
        "Please rewrite the text below so it sounds natural, concise, and human-written. "
        "Reduce obvious AI-style phrasing while preserving meaning, tone, and key points.\n\n"
        "Original text:\n"
        "This proposal aims to improve overall efficiency through a systematic methodology, "
        "while aligning cross-functional collaboration to ensure objective completion. "
        "We will continue to advance key actions to guarantee closed-loop delivery across tasks."
    )


async def run_demo(config: HarnessConfig) -> None:
    agent = ReActAgent(
        name="skill-demo-agent",
        tools=[*SKILL_TOOLS],
        config=config,
    )
    result = await agent.run(build_rewrite_query())
    print("=== Final output ===")
    print(result.output)
    print(f"\nSteps: {result.step_count}")
    print(f"Tokens: {result.usage.total_tokens}")


async def main() -> None:
    config = HarnessConfig.load(PROJECT_ROOT / "arktor.yaml")
    has_skills = print_skill_catalog_preview()
    if not has_skills:
        print("No skills available. Check arktor.yaml -> skill.dirs and SKILL.md files.")
        return
    await run_demo(config)


if __name__ == "__main__":
    asyncio.run(main())
