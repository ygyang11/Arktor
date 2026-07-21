"""Tests for AgentTeam modes and internal judge behavior."""
from __future__ import annotations

import pytest

from agent_harness.agent.conversational import ConversationalAgent
from agent_harness.hooks import DefaultHooks, TracingHooks
from agent_harness.core.config import HarnessConfig, TracingConfig
from agent_harness.orchestration.team import AgentTeam, TeamMode, TEAM_PROMPTS
from tests.conftest import MockLLM, MockTool


class _TeamSpyHooks(DefaultHooks):
    def __init__(self) -> None:
        self.started: list[tuple[str, str]] = []
        self.ended: list[tuple[str, str]] = []

    async def on_team_start(self, team_name: str, mode: str) -> None:
        self.started.append((team_name, mode))

    async def on_team_end(self, team_name: str, mode: str) -> None:
        self.ended.append((team_name, mode))


def _make_worker(name: str, llm: MockLLM, system_prompt: str = "") -> ConversationalAgent:
    return ConversationalAgent(name=name, llm=llm, system_prompt=system_prompt)


class TestTeamMode:
    def test_enum_and_init_contract(self) -> None:
        assert TeamMode.SUPERVISOR.value == "supervisor"
        assert TeamMode.DEBATE.value == "debate"
        assert TeamMode.ROUND_ROBIN.value == "round_robin"

        assert TeamMode("supervisor") == TeamMode.SUPERVISOR
        assert TeamMode("debate") == TeamMode.DEBATE
        assert TeamMode("round_robin") == TeamMode.ROUND_ROBIN

        team_from_string = AgentTeam(agents=[_make_worker("w1", MockLLM())], mode="supervisor")
        assert team_from_string.mode == TeamMode.SUPERVISOR

        team_from_enum = AgentTeam(agents=[_make_worker("w1", MockLLM())], mode=TeamMode.DEBATE)
        assert team_from_enum.mode == TeamMode.DEBATE

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError):
            TeamMode("invalid_mode")

    def test_team_requires_at_least_one_worker(self) -> None:
        with pytest.raises(ValueError, match="at least one worker"):
            AgentTeam(agents=[], mode=TeamMode.SUPERVISOR)


class TestInternalJudge:
    def test_judge_is_cached_and_distinct_from_worker(self) -> None:
        worker = _make_worker("worker", MockLLM(), system_prompt="WORKER_ONLY")
        team = AgentTeam(agents=[worker], mode=TeamMode.SUPERVISOR)

        judge_one = team._get_judge()
        judge_two = team._get_judge()

        assert judge_one is judge_two
        assert judge_one is not worker
        assert judge_one.name == "team_judge"
        assert judge_one.system_prompt == TEAM_PROMPTS["judge.system"]
        assert judge_one.llm is worker.llm

    def test_judge_and_rebuilt_worker_keep_dual_llm_wiring(self) -> None:
        main = MockLLM()
        main.config.model = "worker-main"
        sub = MockLLM()
        sub.config.model = "worker-sub"
        worker = ConversationalAgent(
            name="worker",
            llm=main,
            sub_llm=sub,
            tools=[MockTool()],
            system_prompt="",
        )
        team = AgentTeam(agents=[worker], mode=TeamMode.SUPERVISOR)

        judge = team._get_judge()
        team._fork_worker_contexts(judge)

        assert judge.llm is main
        assert judge.sub_llm is sub
        assert worker.context.short_term_memory.model == "worker-main"
        compressor = worker.context.short_term_memory.compressor
        assert compressor is not None
        assert compressor._llm is sub
        assert compressor._model == "worker-main"
        assert worker._event_bus is worker.context.event_bus
        assert main._event_bus is worker.context.event_bus
        assert sub._event_bus is worker.context.event_bus
        assert worker.tool_executor._event_bus is worker.context.event_bus


class TestHooks:
    @pytest.mark.asyncio
    async def test_team_uses_hooks_configured_in_init(self) -> None:
        llm = MockLLM()
        llm.add_text_response("DONE")
        llm.add_text_response("final synthesis")

        hooks = _TeamSpyHooks()
        team = AgentTeam(
            agents=[_make_worker("a", llm)],
            mode=TeamMode.ROUND_ROBIN,
            max_rounds=1,
            hooks=hooks,
        )

        await team.run("test task")

        assert hooks.started == [("team", "round_robin")]
        assert hooks.ended == [("team", "round_robin")]

    def test_team_uses_explicit_config_for_hooks_resolution(self) -> None:
        llm = MockLLM()
        cfg = HarnessConfig(tracing=TracingConfig(enabled=False))
        team = AgentTeam(
            agents=[_make_worker("a", llm)],
            mode=TeamMode.SUPERVISOR,
            config=cfg,
        )

        assert isinstance(team.hooks, DefaultHooks)
        assert not isinstance(team.hooks, TracingHooks)

    def test_team_without_config_uses_active_global_config(self) -> None:
        original = HarnessConfig._instance
        HarnessConfig._instance = HarnessConfig(tracing=TracingConfig(enabled=False))
        llm = MockLLM()

        try:
            team = AgentTeam(
                agents=[_make_worker("a", llm)],
                mode=TeamMode.SUPERVISOR,
            )
            assert isinstance(team.hooks, DefaultHooks)
            assert not isinstance(team.hooks, TracingHooks)
        finally:
            HarnessConfig._instance = original


class TestSupervisorMode:
    @pytest.mark.asyncio
    async def test_supervisor_mode_runs_without_supervisor_param(self) -> None:
        llm_w1 = MockLLM()
        llm_w1.add_text_response('{"assignments": {"w1": "Analyze upside", "w2": "Analyze risk"}}')
        llm_w1.add_text_response("worker w1 analysis")
        llm_w1.add_text_response("final synthesis")

        llm_w2 = MockLLM()
        llm_w2.add_text_response("worker w2 analysis")

        worker_one = _make_worker("w1", llm_w1, system_prompt="Upside-focused")
        worker_two = _make_worker("w2", llm_w2, system_prompt="Risk-focused")

        team = AgentTeam(agents=[worker_one, worker_two], mode=TeamMode.SUPERVISOR)
        result = await team.run("Evaluate this strategy")

        assert result.output == "final synthesis"
        assert result.rounds == 1
        assert "judge_delegation" in result.agent_results
        assert "w1" in result.agent_results
        assert "w2" in result.agent_results
        assert "judge_synthesis" in result.agent_results

    @pytest.mark.asyncio
    async def test_supervisor_mode_forks_worker_contexts_from_judge(self) -> None:
        llm = MockLLM()
        llm.add_text_response('{"assignments": {"worker": "Do it"}}')
        llm.add_text_response("worker result")
        llm.add_text_response("final synthesis")

        worker = _make_worker("worker", llm)
        original_context = worker.context

        team = AgentTeam(agents=[worker], mode=TeamMode.SUPERVISOR)
        await team.run("test task")

        assert worker.context is not original_context

    @pytest.mark.asyncio
    async def test_supervisor_workers_receive_context_in_working_memory(self) -> None:
        llm = MockLLM()
        llm.add_text_response('{"assignments": {"reviewer": "Check quality"}}')
        llm.add_text_response("review done")
        llm.add_text_response("final synthesis")

        worker = _make_worker("reviewer", llm)
        team = AgentTeam(agents=[worker], mode=TeamMode.SUPERVISOR)
        original_input = "Full research data here with lots of detail"

        await team.run(original_input)

        assert worker.context.working_memory.get("overall_task") == original_input

    @pytest.mark.asyncio
    async def test_supervisor_workers_receive_subtask_prefix(self) -> None:
        llm = MockLLM()
        llm.add_text_response('{"assignments": {"reviewer": "Check quality"}}')
        llm.add_text_response("review done")
        llm.add_text_response("final synthesis")

        worker = _make_worker("reviewer", llm)
        team = AgentTeam(agents=[worker], mode=TeamMode.SUPERVISOR)

        await team.run("Full task")

        worker_call = next(
            msgs for msgs in llm.call_history
            if any(
                msg.role.value == "user"
                and "Your assigned subtask:" in (msg.content or "")
                for msg in msgs
            )
        )
        user_msgs = [m for m in worker_call if m.role.value == "user"]
        assert user_msgs[0].content is not None
        assert user_msgs[0].content.startswith("Your assigned subtask:")


class TestRoundRobinMode:
    @pytest.mark.asyncio
    async def test_round_robin_always_synthesizes_with_judge(self) -> None:
        llm_w1 = MockLLM()
        llm_w1.add_text_response("worker one contribution")
        llm_w1.add_text_response("round robin synthesis")

        llm_w2 = MockLLM()
        llm_w2.add_text_response("worker two contribution")

        team = AgentTeam(
            agents=[_make_worker("w1", llm_w1), _make_worker("w2", llm_w2)],
            mode=TeamMode.ROUND_ROBIN,
            max_rounds=1,
        )
        result = await team.run("Complete the task")

        assert result.output == "round robin synthesis"
        assert "judge_synthesis" in result.agent_results
        assert result.rounds == 1


class TestDebateMode:
    @pytest.mark.asyncio
    async def test_debate_uses_judge_llm_for_convergence_and_full_history(self) -> None:
        llm_w1 = MockLLM()
        llm_w1.add_text_response("pos_a1")
        llm_w1.add_text_response("pos_a2")
        llm_w1.add_text_response('{"converged": true, "reason": "aligned"}')
        llm_w1.add_text_response("final answer")

        llm_w2 = MockLLM()
        llm_w2.add_text_response("pos_b1")
        llm_w2.add_text_response("pos_b2")

        team = AgentTeam(
            agents=[_make_worker("a", llm_w1), _make_worker("b", llm_w2)],
            mode=TeamMode.DEBATE,
            max_rounds=4,
        )
        result = await team.run("What is the best approach?")

        assert result.output == "final answer"
        assert result.rounds == 2
        assert "judge" in result.agent_results

        convergence_calls = [
            messages
            for messages in llm_w1.call_history
            if messages
            and messages[0].role.value == "system"
            and "debate has converged" in (messages[0].content or "")
        ]
        assert len(convergence_calls) == 1

        judge_payload = "\n".join(msg.content or "" for msg in llm_w1.call_history[-1])
        assert "=== Round 1 ===" in judge_payload
        assert "=== Round 2 ===" in judge_payload
        assert "pos_a1" in judge_payload
        assert "pos_a2" in judge_payload
        assert "pos_b1" in judge_payload
        assert "pos_b2" in judge_payload

    @pytest.mark.asyncio
    async def test_debate_continues_when_convergence_output_is_not_json(self) -> None:
        llm_w1 = MockLLM()
        llm_w1.add_text_response("a1")
        llm_w1.add_text_response("a2")
        llm_w1.add_text_response("not-json")
        llm_w1.add_text_response("a3")
        llm_w1.add_text_response("still-not-json")
        llm_w1.add_text_response("final synthesis")

        llm_w2 = MockLLM()
        llm_w2.add_text_response("b1")
        llm_w2.add_text_response("b2")
        llm_w2.add_text_response("b3")

        team = AgentTeam(
            agents=[_make_worker("a", llm_w1), _make_worker("b", llm_w2)],
            mode=TeamMode.DEBATE,
            max_rounds=3,
        )
        result = await team.run("Debate this")

        assert result.rounds == 3
        assert result.output == "final synthesis"
