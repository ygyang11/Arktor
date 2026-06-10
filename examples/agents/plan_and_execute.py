"""PlanAndExecuteAgent for complex task decomposition and multi-step execution.

Demonstrates: The Plan-Execute-Replan loop architecture.
"""

import asyncio
from pathlib import Path

from agent_app.tools import PDF_TOOLS, WEB_TOOLS
from agent_harness import HarnessConfig, PlanAndExecuteAgent, tool


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@tool
async def analyze_data(data: str) -> str:
    """Analyze provided data and extract key insights.

    Args:
        data: The raw data or text to analyze.
    """
    word_count = len(data.split())
    return (
        f"Analysis of {word_count}-word input: "
        f"Identified 3 key trends, 2 statistical claims, and 1 projection. "
        f"Data appears reliable with multiple corroborating sources."
    )


@tool
async def write_summary(topic: str, key_points: str) -> str:
    """Write a structured summary given a topic and key points.

    Args:
        topic: The topic to summarize.
        key_points: Comma-separated key points to include.
    """
    return (
        f"Summary: {topic}\n"
        f"Key findings: {key_points}\n"
        f"Conclusion: The data supports continued growth in this sector "
        f"with strong investment signals and declining costs."
    )


async def main() -> None:
    config = HarnessConfig.load(PROJECT_ROOT / "arktor.yaml")

    agent = PlanAndExecuteAgent(
        name="researcher",
        tools=[*WEB_TOOLS, *PDF_TOOLS, analyze_data, write_summary],
        executor_max_steps=20,
        config=config,
    )

    query = (
        "Research the current state of renewable energy and produce "
        "a brief report covering solar, wind, and overall trends."
    )
    print(f"Query: {query}\n")

    result = await agent.run(query)

    print(f"Final Report:\n{result.output}\n")
    print(f"Total usage: {result.usage}")


if __name__ == "__main__":
    asyncio.run(main())
