"""Deep Research — multi-agent orchestration for complex research tasks."""

import asyncio
from pathlib import Path

from agent_harness import BaseTool, ConversationalAgent, HarnessConfig, ReActAgent
from agent_harness.agent.base import AgentResult
from agent_harness.orchestration import AgentTeam, DAGNode, DAGOrchestrator, TeamMode
from agent_app.tools import WEB_TOOLS, PAPER_TOOLS, PDF_TOOLS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEEP_RESEARCH_PROMPTS: dict[str, str] = {
    "planner.system": (
        "You are a research planner for academic deep research. Decompose the user "
        "question into an actionable blueprint that balances timely evidence and "
        "scholarly evidence, defines evidence quality criteria, includes uncertainty "
        "handling, and specifies the expected report structure."
    ),
    "worker.hardware": (
        "You are responsible for hardware-track research: chips, error correction, "
        "and compute milestones. Select suitable retrieval methods autonomously, "
        "prioritize high-quality evidence, and attribute every major claim."
    ),
    "worker.applications": (
        "You are responsible for application-track research: pharma, finance, "
        "optimization, and cryptography. Use rigorous source evaluation and "
        "distinguish established findings from speculative claims."
    ),
    "worker.market": (
        "You are responsible for market-track research: ecosystem, funding, "
        "commercialization, and policy impact. Distinguish facts, forecasts, and "
        "assumptions with explicit evidence tags."
    ),
    "review.methodology": (
        "You are a methodology reviewer. Evaluate methodological rigor, source "
        "quality, and evidence traceability."
    ),
    "review.risk": (
        "You are a risk reviewer. Identify uncertainties, evidence conflicts, "
        "limitations, and counterexamples."
    ),
    "review.business": (
        "You are a business reviewer. Evaluate decision value, feasibility, and "
        "strategic impact while separating facts from assumptions."
    ),
    "writer.system": (
        "You are the final report writer. Produce an academic-style report with "
        "evidence mapping, source index, and uncertainty disclosure based on the "
        "plan, parallel research outputs, and review feedback."
    ),
}


def format_dag_outputs(outputs: dict[str, AgentResult]) -> str:
    ordered_ids = ("hardware", "applications", "market")
    sections: list[str] = []
    for node_id in ordered_ids:
        result = outputs.get(node_id)
        if result is None:
            continue
        sections.append(f"## {node_id}\n{result.output}")
    return "\n\n".join(sections)


def sum_usage_tokens(results: dict[str, AgentResult]) -> int:
    return sum(result.usage.total_tokens for result in results.values())


def build_research_dag(config: HarnessConfig, tools: list[BaseTool]) -> DAGOrchestrator:
    hardware = ReActAgent(
        name="hardware_researcher",
        tools=tools,
        system_prompt=DEEP_RESEARCH_PROMPTS["worker.hardware"],
        config=config,
    )
    applications = ReActAgent(
        name="applications_researcher",
        tools=tools,
        system_prompt=DEEP_RESEARCH_PROMPTS["worker.applications"],
        config=config,
    )
    market = ReActAgent(
        name="market_researcher",
        tools=tools,
        system_prompt=DEEP_RESEARCH_PROMPTS["worker.market"],
        config=config,
    )

    return DAGOrchestrator(
        nodes=[
            DAGNode(id="hardware", agent=hardware),
            DAGNode(id="applications", agent=applications),
            DAGNode(id="market", agent=market),
        ],
    )


def build_review_team(config: HarnessConfig) -> AgentTeam:
    methodology = ConversationalAgent(
        name="methodology_reviewer",
        system_prompt=DEEP_RESEARCH_PROMPTS["review.methodology"],
        config=config,
    )
    risk = ConversationalAgent(
        name="risk_reviewer",
        system_prompt=DEEP_RESEARCH_PROMPTS["review.risk"],
        config=config,
    )
    business = ConversationalAgent(
        name="business_reviewer",
        system_prompt=DEEP_RESEARCH_PROMPTS["review.business"],
        config=config,
    )
    return AgentTeam(
        agents=[methodology, risk, business],
        mode=TeamMode.SUPERVISOR,
        max_rounds=2,
    )


async def main() -> None:
    config = HarnessConfig.load(PROJECT_ROOT / "arktor.yaml")
    research_tools: list[BaseTool] = [
        *WEB_TOOLS,
        *PAPER_TOOLS,
        *PDF_TOOLS,
    ]
    query = "What is the current state of quantum computing and its potential impact?"

    print(f"Research query: {query}\n")
    print("Phase 1: Planning...")
    planner = ConversationalAgent(
        name="research_planner",
        system_prompt=DEEP_RESEARCH_PROMPTS["planner.system"],
        config=config,
    )
    plan_result = await planner.run(query)

    print("Phase 2: Parallel research...")
    dag = build_research_dag(config, research_tools)
    dag_result = await dag.run(
        f"Question:\n{query}\n\n"
        f"Planning Blueprint:\n{plan_result.output}\n\n"
        "Research Requirements:\n"
        "- Each major finding should be supported by traceable evidence.\n"
        "- Prefer multi-source corroboration when possible.\n"
        "- Mark evidence as [WEB] or [PAPER].\n"
        "- If full-text evidence is unavailable, disclose limitations explicitly."
    )
    research_bundle = format_dag_outputs(dag_result.outputs)
    print(f"Execution order: {dag_result.execution_order}")

    print("Phase 3: Cross review...")
    review_team = build_review_team(config)
    review_result = await review_team.run(
        "Evaluate the research findings with focus on:\n"
        "- methodological rigor\n"
        "- evidence quality and credibility\n"
        "- consistency across source types\n"
        "- unsupported claims and uncertainty disclosure\n\n"
        f"Question:\n{query}\n\n"
        f"Plan:\n{plan_result.output}\n\n"
        f"Research:\n{research_bundle}"
    )

    print("Phase 4: Final synthesis...")
    writer = ReActAgent(
        name="final_writer",
        tools=[],
        system_prompt=DEEP_RESEARCH_PROMPTS["writer.system"],
        config=config,
    )
    final_result = await writer.run(
        "Generate an academic-style final report with sections:\n"
        "1. Research Scope and Questions\n"
        "2. Key Findings\n"
        "3. Evidence Synthesis ([WEB] vs [PAPER])\n"
        "4. Conflicts and Uncertainties\n"
        "5. Actionable Recommendations\n"
        "6. Source Index\n\n"
        "Every key finding must map to concrete sources.\n\n"
        f"Question:\n{query}\n\n"
        f"Plan:\n{plan_result.output}\n\n"
        f"Parallel Research:\n{research_bundle}\n\n"
        f"Review Summary:\n{review_result.output}"
    )

    total_tokens = (
        plan_result.usage.total_tokens
        + sum_usage_tokens(dag_result.outputs)
        + sum_usage_tokens(review_result.agent_results)
        + final_result.usage.total_tokens
    )

    print("\n" + "=" * 60)
    print("FINAL RESEARCH REPORT")
    print("=" * 60)
    print(final_result.output)
    print(f"\nTotal tokens: {total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
