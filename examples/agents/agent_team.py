"""AgentTeam with three collaboration modes: supervisor, debate, round_robin.

Demonstrates multi-agent collaboration where specialist agents contribute
different perspectives and are coordinated via different strategies.
"""

import asyncio
from pathlib import Path

from agent_harness import ConversationalAgent, HarnessConfig
from agent_harness.orchestration import AgentTeam, TeamMode


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


AgentsTriple = tuple[ConversationalAgent, ConversationalAgent, ConversationalAgent]


def create_agents(config: HarnessConfig) -> AgentsTriple:
    """Create specialist worker agents."""
    optimist = ConversationalAgent(
        name="optimist",
        system_prompt=(
            "You are an optimistic analyst. You always highlight opportunities, "
            "potential upsides, and reasons for enthusiasm. Keep it to 2-3 sentences."
        ),
        config=config,
    )

    pessimist = ConversationalAgent(
        name="pessimist",
        system_prompt=(
            "You are a cautious, risk-focused analyst. You highlight potential "
            "downsides, risks, and pitfalls. Keep it to 2-3 sentences."
        ),
        config=config,
    )

    realist = ConversationalAgent(
        name="realist",
        system_prompt=(
            "You are a pragmatic realist. You weigh both sides objectively and "
            "provide a balanced, evidence-based view. Keep it to 2-3 sentences."
        ),
        config=config,
    )

    return optimist, pessimist, realist


async def demo_supervisor(agents: AgentsTriple, topic: str, config: HarnessConfig) -> None:
    """Supervisor mode: delegates to all workers, then synthesizes."""
    optimist, pessimist, realist = agents

    print("=" * 60)
    print("MODE 1: Supervisor")
    print("=" * 60)

    team = AgentTeam(
        agents=[optimist, pessimist, realist],
        mode=TeamMode.SUPERVISOR,
    )

    result = await team.run(topic)
    print(f"\nSynthesized output:\n{result.output}")
    print(f"Rounds: {result.rounds} | Contributors: {list(result.agent_results.keys())}")


async def demo_debate(agents: AgentsTriple, topic: str, config: HarnessConfig) -> None:
    """Debate mode: agents argue independently, judge picks best."""
    optimist, pessimist, realist = agents

    print("\n" + "=" * 60)
    print("MODE 2: Debate")
    print("=" * 60)

    team = AgentTeam(
        agents=[optimist, pessimist, realist],
        mode=TeamMode.DEBATE,
    )

    result = await team.run(topic)
    print(f"\nJudge's verdict:\n{result.output}")
    print(f"Rounds: {result.rounds} | Debaters: {list(result.agent_results.keys())}")


async def demo_round_robin(agents: AgentsTriple, topic: str, config: HarnessConfig) -> None:
    """Round-robin mode: agents build on each other's work iteratively."""
    optimist, pessimist, realist = agents

    print("\n" + "=" * 60)
    print("MODE 3: Round Robin")
    print("=" * 60)

    team = AgentTeam(
        agents=[optimist, pessimist, realist],
        mode=TeamMode.ROUND_ROBIN,
        max_rounds=2,
    )

    result = await team.run(topic)
    print(f"\nFinal output (after iterative refinement):\n{result.output}")
    print(f"Rounds: {result.rounds} | Participants: {list(result.agent_results.keys())}")


async def main() -> None:
    config = HarnessConfig.load(PROJECT_ROOT / "arktor.yaml")
    agents = create_agents(config)

    topic = "Should a mid-size company invest heavily in AI automation this year?"

    print(f"Topic: {topic}\n")

    await demo_supervisor(agents, topic, config)
    await demo_debate(agents, topic, config)
    await demo_round_robin(agents, topic, config)


if __name__ == "__main__":
    asyncio.run(main())
