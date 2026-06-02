"""Multi-agent team collaboration patterns."""
from __future__ import annotations

import asyncio
import logging
from enum import Enum

from pydantic import BaseModel, Field

from agent_harness.agent.base import AgentResult, BaseAgent
from agent_harness.core.config import HarnessConfig
from agent_harness.core.message import Message
from agent_harness.hooks import DefaultHooks, resolve_hooks
from agent_harness.llm.base import BaseLLM
from agent_harness.utils.json_utils import parse_json_lenient

logger = logging.getLogger(__name__)

TEAM_PROMPTS: dict[str, str] = {
    "judge.system": (
        "## Role\n"
        "You are the impartial coordination judge for a high-performing multi-agent team.\n\n"
        "## Objectives\n"
        "- Maximize factual reliability and decision quality.\n"
        "- Preserve original task intent while improving clarity and executability.\n"
        "- Enforce disciplined collaboration and consistent output standards.\n\n"
        "## Rules\n"
        "- Stay neutral, evidence-oriented, and outcome-focused.\n"
        "- Prefer explicit assumptions, traceable logic, and direct conclusions.\n"
        "- Resolve conflicts using argument quality, not confidence or length.\n"
        "- Surface critical uncertainty when it materially affects decisions.\n"
        "- Never return markdown code fences for JSON-only tasks."
    ),
    "supervisor.delegate": (
        "## Role\n"
        "You are the task decomposition and delegation supervisor.\n\n"
        "## Inputs\n"
        "Available workers:\n"
        "{worker_info}\n\n"
        "Original task:\n"
        "{task}\n\n"
        "## Objectives\n"
        "- Produce a complete, non-overlapping execution split.\n"
        "- Assign each selected worker a high-impact subtask aligned with their expertise.\n"
        "- Avoid ambiguity so each subtask is directly executable.\n\n"
        "## Rules\n"
        "- Use exact worker names from the provided list.\n"
        "- Assign only listed workers and only when they add value.\n"
        "- Write subtasks as concrete deliverables with scope boundaries.\n"
        "- Preserve core intent, required constraints, and decision criteria.\n"
        "- Minimize redundant work and dependency conflicts.\n"
        "- If the task is simple, you may assign the same core objective to all "
        "workers with distinct angles.\n\n"
        "## Output Format\n"
        "Return ONLY JSON:\n"
        "{{\"assignments\": {{\"worker_name\": \"subtask description\"}}}}"
    ),
    "supervisor.synthesize": (
        "## Role\n"
        "You are the final synthesis supervisor.\n\n"
        "## Inputs\n"
        "Original task:\n"
        "{task}\n\n"
        "Worker outputs:\n"
        "{worker_results}\n\n"
        "## Objectives\n"
        "- Integrate the strongest compatible insights into one coherent answer.\n"
        "- Resolve conflicts with explicit reasoning and practical trade-offs.\n"
        "- Deliver a decision-ready result with clear actionability.\n\n"
        "## Rules\n"
        "- Prioritize correctness, completeness, and practical usefulness.\n"
        "- Reject weak claims and unsupported assertions.\n"
        "- State key assumptions when evidence is incomplete.\n"
        "- Preserve important minority concerns if they change risk posture.\n\n"
        "## Output Format\n"
        "Return a comprehensive final answer in plain text."
    ),
    "debate.refine": (
        "## Role\n"
        "You are a debate participant refining your position.\n\n"
        "## Inputs\n"
        "Original question:\n"
        "{task}\n\n"
        "Your previous position:\n"
        "{previous_position}\n\n"
        "Other agents' previous-round positions:\n"
        "{other_views}\n\n"
        "Current round: {round_num}/{max_rounds}\n\n"
        "## Objectives\n"
        "- Improve epistemic quality from round to round.\n"
        "- Converge on stronger conclusions when justified.\n"
        "- Expose unresolved disagreements that materially matter.\n\n"
        "## Rules\n"
        "- Critically evaluate competing views and cite their strongest points fairly.\n"
        "- Revise your position when counterarguments are stronger.\n"
        "- Keep claims specific, testable, and decision-relevant.\n"
        "- Avoid rhetorical repetition and generic statements.\n"
        "- Be concise but substantively complete.\n\n"
        "## Output Format\n"
        "Return your updated position in plain text."
    ),
    "debate.convergence": (
        "## Role\n"
        "You evaluate whether a multi-agent debate has converged.\n\n"
        "## Inputs\n"
        "{positions_text}\n\n"
        "## Objectives\n"
        "- Detect meaningful stabilization of conclusions and reasoning.\n"
        "- Distinguish semantic agreement from superficial wording overlap.\n\n"
        "## Rules\n"
        "- Converged: core conclusions and key arguments are substantively aligned.\n"
        "- Not converged: any material stance shift or genuinely new argument appears.\n"
        "- Ignore minor phrasing differences and stylistic variation.\n"
        "- If uncertainty remains on critical decision points, mark as not converged.\n\n"
        "## Output Format\n"
        "Return ONLY JSON:\n"
        "{{\"converged\": true/false, \"reason\": \"brief explanation\"}}"
    ),
    "debate.judge": (
        "## Role\n"
        "You are an impartial judge synthesizing a multi-round debate.\n\n"
        "## Inputs\n"
        "Original question:\n"
        "{task}\n\n"
        "Completed rounds: {rounds}\n\n"
        "Debate transcript:\n"
        "{transcript}\n\n"
        "## Objectives\n"
        "- Deliver a robust final position grounded in the best arguments.\n"
        "- Make trade-offs explicit and decision-useful.\n\n"
        "## Rules\n"
        "- Prioritize arguments that remained robust under critique.\n"
        "- Separate validated consensus from unresolved disagreement.\n"
        "- Resolve disagreements with explicit rationale and risk framing.\n"
        "- Avoid overclaiming when evidence is mixed.\n\n"
        "## Output Format\n"
        "Return one authoritative, self-contained final answer."
    ),
    "round_robin.turn": (
        "## Role\n"
        "You are a round-robin contributor in a multi-agent team.\n\n"
        "## Inputs\n"
        "Task:\n"
        "{task}\n\n"
        "Current accumulated discussion:\n"
        "{accumulated}\n\n"
        "Your name: {agent_name}\n"
        "Round: {round_num}/{max_rounds}\n\n"
        "## Objectives\n"
        "- Add net-new value each turn.\n"
        "- Push the discussion toward a high-quality final answer.\n\n"
        "## Rules\n"
        "- Add distinct value beyond prior contributions.\n"
        "- Explicitly avoid repeating points already covered.\n"
        "- Keep output concise, concrete, and actionable.\n"
        "- If the task is already sufficiently complete, respond with exactly DONE.\n\n"
        "## Output Format\n"
        "Return plain text or DONE."
    ),
    "round_robin.judge": (
        "## Role\n"
        "You are a final synthesis judge for round-robin collaboration.\n\n"
        "## Inputs\n"
        "Original task:\n"
        "{task}\n\n"
        "Rounds completed: {rounds}\n\n"
        "Discussion transcript:\n"
        "{discussion}\n\n"
        "## Objectives\n"
        "- Convert iterative discussion into a polished final deliverable.\n"
        "- Preserve critical insights while removing duplication and noise.\n\n"
        "## Rules\n"
        "- Merge complementary insights into one coherent structure.\n"
        "- Eliminate redundancy, contradictions, and unresolved ambiguity.\n"
        "- Prioritize recommendations by impact and feasibility.\n"
        "- Include caveats only when they materially affect execution.\n\n"
        "## Output Format\n"
        "Return a comprehensive final answer in plain text."
    ),
}


class TeamMode(str, Enum):
    """Team collaboration modes."""
    SUPERVISOR = "supervisor"
    DEBATE = "debate"
    ROUND_ROBIN = "round_robin"


class TeamResult(BaseModel):
    """Result from team execution."""
    output: str
    agent_results: dict[str, AgentResult] = Field(default_factory=dict)
    rounds: int = 0


class AgentTeam:
    """Multi-agent team with supervisor, debate, and round-robin modes."""

    def __init__(
        self,
        agents: list[BaseAgent],
        mode: str | TeamMode = TeamMode.SUPERVISOR,
        max_rounds: int = 3,
        hooks: DefaultHooks | None = None,
        config: HarnessConfig | None = None,
    ) -> None:
        self.agents = agents
        self.mode = TeamMode(mode) if isinstance(mode, str) else mode
        self.max_rounds = max_rounds
        self.hooks = resolve_hooks(hooks, config)
        self._judge: BaseAgent | None = None

        self._ensure_agents()

    async def run(self, input: str) -> TeamResult:
        started = False
        if hasattr(self.hooks, "on_team_start"):
            await self.hooks.on_team_start("team", self.mode.value)
            started = True

        try:
            if self.mode == TeamMode.SUPERVISOR:
                return await self._run_supervisor(input)
            if self.mode == TeamMode.DEBATE:
                return await self._run_debate(input)
            if self.mode == TeamMode.ROUND_ROBIN:
                return await self._run_round_robin(input)
            raise ValueError(f"Unknown team mode: {self.mode}")
        finally:
            if started and hasattr(self.hooks, "on_team_end"):
                await self.hooks.on_team_end("team", self.mode.value)

    def _ensure_agents(self) -> None:
        if not self.agents:
            raise ValueError("AgentTeam requires at least one worker agent")


    def _get_judge(self) -> BaseAgent:
        if self._judge is not None:
            return self._judge

        from agent_harness.agent.conversational import ConversationalAgent  # noqa: PLC0415

        primary_worker = self.agents[0]
        judge_context = primary_worker.context.fork(name="team_judge")
        self._judge = ConversationalAgent(
            name="team_judge",
            llm=primary_worker.llm,
            system_prompt=TEAM_PROMPTS["judge.system"],
            context=judge_context,
            hooks=self.hooks,
        )
        return self._judge

    def _fork_worker_contexts(self, judge: BaseAgent) -> None:
        for worker in self.agents:
            worker.context = judge.context.fork(name=worker.name)

    async def _run_named_inputs_parallel(
        self,
        named_inputs: list[tuple[str, str]],
    ) -> tuple[dict[str, AgentResult], list[str]]:
        worker_map = {worker.name: worker for worker in self.agents}
        results: dict[str, AgentResult] = {}
        execution_order: list[str] = []

        coroutines = [worker_map[name].run(worker_input) for name, worker_input in named_inputs]
        outputs = await asyncio.gather(*coroutines, return_exceptions=True)

        for (worker_name, _), output in zip(named_inputs, outputs):
            execution_order.append(worker_name)
            if isinstance(output, BaseException):
                logger.warning("Worker '%s' failed: %s", worker_name, output)
                results[worker_name] = AgentResult(
                    output=f"Error: {output}",
                    messages=[],
                    steps=[],
                )
            else:
                results[worker_name] = output

        return results, execution_order

    async def _synthesize_with_judge(
        self,
        judge: BaseAgent,
        prompt_key: str,
        **format_kwargs: str | int,
    ) -> AgentResult:
        prompt = TEAM_PROMPTS[prompt_key].format(**format_kwargs)
        return await judge.run(prompt)

    async def _run_supervisor(self, input: str) -> TeamResult:
        judge = self._get_judge()
        self._fork_worker_contexts(judge)
        worker_map = {worker.name: worker for worker in self.agents}
        worker_info = "\n".join(
            f"- {worker.name}: {worker.system_prompt}" if worker.system_prompt
            else f"- {worker.name}"
            for worker in self.agents
        )

        delegation_result = await self._synthesize_with_judge(
            judge,
            "supervisor.delegate",
            worker_info=worker_info,
            task=input,
        )

        assignments: dict[str, str] | None = None
        try:
            delegation_data = parse_json_lenient(delegation_result.output)
            if isinstance(delegation_data, dict):
                raw_assignments = delegation_data.get("assignments", delegation_data)
                if isinstance(raw_assignments, dict):
                    cleaned: dict[str, str] = {}
                    for key, value in raw_assignments.items():
                        if isinstance(key, str):
                            cleaned[key] = value if isinstance(value, str) else str(value)
                    assignments = cleaned or None
        except (ValueError, TypeError):
            assignments = None

        named_inputs: list[tuple[str, str]] = []
        if assignments:
            for worker_name, subtask in assignments.items():
                if worker_name in worker_map:
                    worker_map[worker_name].context.working_memory.set(
                        "overall_task", input
                    )
                    named_inputs.append(
                        (worker_name, f"Your assigned subtask:\n{subtask}")
                    )

        if not named_inputs:
            named_inputs = [(worker.name, input) for worker in self.agents]

        worker_results, worker_order = await self._run_named_inputs_parallel(named_inputs)
        worker_sections: list[str] = []
        for worker_name in worker_order:
            worker_sections.append(f"--- {worker_name} ---")
            worker_sections.append(worker_results[worker_name].output)
            worker_sections.append("")
        formatted_worker_results = "\n".join(worker_sections).strip()

        synthesis_result = await self._synthesize_with_judge(
            judge,
            "supervisor.synthesize",
            task=input,
            worker_results=formatted_worker_results,
        )

        agent_results: dict[str, AgentResult] = {"judge_delegation": delegation_result}
        agent_results.update(worker_results)
        agent_results["judge_synthesis"] = synthesis_result

        return TeamResult(
            output=synthesis_result.output,
            agent_results=agent_results,
            rounds=1,
        )

    async def _run_debate(self, input: str) -> TeamResult:
        judge = self._get_judge()
        self._fork_worker_contexts(judge)

        agent_results: dict[str, AgentResult] = {}
        debate_history: dict[int, dict[str, str]] = {}

        round_one_inputs = [(worker.name, input) for worker in self.agents]
        round_one_results, _ = await self._run_named_inputs_parallel(round_one_inputs)

        current_positions: dict[str, str] = {}
        for worker in self.agents:
            worker_result = round_one_results[worker.name]
            current_positions[worker.name] = worker_result.output
            agent_results[f"{worker.name}_round1"] = worker_result
        debate_history[1] = dict(current_positions)
        actual_rounds = 1

        for round_num in range(2, self.max_rounds + 1):
            previous_positions = dict(current_positions)
            named_inputs: list[tuple[str, str]] = []

            for worker in self.agents:
                other_views = "\n\n".join(
                    f"--- {name} ---\n{position}"
                    for name, position in previous_positions.items()
                    if name != worker.name
                ) or "(no other positions)"
                worker_prompt = TEAM_PROMPTS["debate.refine"].format(
                    task=input,
                    previous_position=previous_positions.get(worker.name, ""),
                    other_views=other_views,
                    round_num=round_num,
                    max_rounds=self.max_rounds,
                )
                named_inputs.append((worker.name, worker_prompt))

            round_results, _ = await self._run_named_inputs_parallel(named_inputs)
            for worker in self.agents:
                worker_result = round_results[worker.name]
                current_positions[worker.name] = worker_result.output
                agent_results[f"{worker.name}_round{round_num}"] = worker_result

            debate_history[round_num] = dict(current_positions)
            actual_rounds = round_num

            converged = await self._check_debate_convergence(
                llm=judge.llm,
                current=current_positions,
                previous=previous_positions,
            )
            if converged:
                logger.info("Debate converged at round %d/%d", round_num, self.max_rounds)
                break

        transcript_lines: list[str] = []
        for round_num in range(1, actual_rounds + 1):
            transcript_lines.append(f"=== Round {round_num} ===")
            round_positions = debate_history.get(round_num, {})
            for worker_name, position in round_positions.items():
                transcript_lines.append(f"\n--- {worker_name} ---")
                transcript_lines.append(position)
            transcript_lines.append("")
        transcript = "\n".join(transcript_lines).strip()

        final_result = await self._synthesize_with_judge(
            judge,
            "debate.judge",
            task=input,
            rounds=actual_rounds,
            transcript=transcript,
        )
        agent_results["judge"] = final_result

        return TeamResult(
            output=final_result.output,
            agent_results=agent_results,
            rounds=actual_rounds,
        )

    async def _check_debate_convergence(
        self,
        llm: BaseLLM,
        *,
        current: dict[str, str],
        previous: dict[str, str],
    ) -> bool:
        positions_lines: list[str] = []
        for worker_name in current:
            previous_position = previous.get(worker_name, "(no previous position)")
            current_position = current[worker_name]
            positions_lines.append(f"--- {worker_name} (previous round) ---")
            positions_lines.append(previous_position)
            positions_lines.append("")
            positions_lines.append(f"--- {worker_name} (current round) ---")
            positions_lines.append(current_position)
            positions_lines.append("")
        positions_text = "\n".join(positions_lines).strip()

        prompt = TEAM_PROMPTS["debate.convergence"].format(positions_text=positions_text)
        try:
            response = await llm.generate([Message.system(prompt)], tools=None)
            content = response.message.content or ""
            parsed = parse_json_lenient(content)
            if isinstance(parsed, dict):
                return bool(parsed.get("converged", False))
        except Exception as exc:
            logger.warning("Debate convergence check failed: %s", exc)
        return False

    async def _run_round_robin(self, input: str) -> TeamResult:
        judge = self._get_judge()
        self._fork_worker_contexts(judge)

        agent_results: dict[str, AgentResult] = {}
        accumulated = ""
        actual_rounds = 0

        for round_num in range(1, self.max_rounds + 1):
            actual_rounds = round_num
            all_done = True

            for worker in self.agents:
                worker_prompt = TEAM_PROMPTS["round_robin.turn"].format(
                    task=input,
                    accumulated=accumulated or "(empty)",
                    agent_name=worker.name,
                    round_num=round_num,
                    max_rounds=self.max_rounds,
                )
                result = await worker.run(worker_prompt)
                agent_results[f"{worker.name}_round{round_num}"] = result
                accumulated += (
                    f"\n--- {worker.name} (round {round_num}) ---\n"
                    f"{result.output}\n"
                )
                if result.output.strip().upper() != "DONE":
                    all_done = False

            if all_done:
                logger.info("Round robin: all agents responded DONE at round %d", round_num)
                break

        synthesis_result = await self._synthesize_with_judge(
            judge,
            "round_robin.judge",
            task=input,
            rounds=actual_rounds,
            discussion=accumulated.strip() or "(empty)",
        )
        agent_results["judge_synthesis"] = synthesis_result

        return TeamResult(
            output=synthesis_result.output,
            agent_results=agent_results,
            rounds=actual_rounds,
        )
