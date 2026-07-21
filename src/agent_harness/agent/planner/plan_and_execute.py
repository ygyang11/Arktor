"""PlanAndExecuteAgent — multi-agent plan-execute-evaluate orchestrator."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_harness.session.base import BaseSession

from agent_harness.agent.base import AgentResult, BaseAgent, StepResult
from agent_harness.agent.planner.executor_agent import ExecutorAgent
from agent_harness.agent.planner.planner_agent import PlannerAgent
from agent_harness.agent.planner.replanner_agent import ReplannerAgent
from agent_harness.agent.planner.types import Plan, PlanStep, ReplanDecision
from agent_harness.context.state import AgentState
from agent_harness.core.message import Message
from agent_harness.llm.types import Usage
from agent_harness.utils.json_utils import parse_json_lenient
from agent_harness.utils.token_counter import truncate_text_by_tokens

logger = logging.getLogger(__name__)

_PLAN_EVENT_OUTPUT_MAX_TOKENS = 64
_REPLANNER_INPUT_PROMPT = "Please provide your replanning decision in valid JSON."


class PlanAndExecuteAgent(BaseAgent):
    """Multi-agent Plan-and-Execute orchestrator.

    Coordinates three sub-agents (Planner, Executor, Replanner) through
    a plan-execute-evaluate loop. Each sub-agent runs in a forked context
    for memory isolation.

    Workflow:
        1. PlannerAgent generates a structured plan
        2. ExecutorAgent executes steps one by one (with tool access)
        3. ReplannerAgent evaluates results and adjusts the plan
        4. Loop until goal achieved or max iterations reached
    """

    def __init__(
        self,
        executor_max_steps: int = 20,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(system_prompt=system_prompt or "", **kwargs)
        self._executor_max_steps = executor_max_steps
        self._executor_approval = self._approval
        self._executor_approval_handler = self._approval_handler

    async def step(self) -> StepResult:
        """Not used — run() is overridden."""
        return StepResult()

    async def run(
        self,
        input: str | Message,
        *,
        session: str | BaseSession | None = None,
        after_input_appended: (Callable[[BaseAgent, Message, str], Awaitable[None]] | None) = None,
    ) -> AgentResult:
        from datetime import datetime

        from agent_harness.session.base import resolve_session

        resolved_session: BaseSession | None = resolve_session(session)

        if self.context.state.is_terminal:
            self.context.state.reset()

        if resolved_session:
            self._bind_session_id(resolved_session.session_id)
            if not await self.context.short_term_memory.get_context_messages():
                state = await resolved_session.load_state()
                if state:
                    await self.apply_session_state(state)

        if isinstance(input, str):
            input_msg = Message.user(input)
            input_text = input
        else:
            input_msg = input
            input_text = input.content or ""

        await self._sync_system_prompt()
        await self.context.short_term_memory.add_message(input_msg)

        self.context.state.transition(AgentState.THINKING)
        await self.hooks.on_run_start(self.name, input_text)
        await self.emit("agent.run.start", agent=self.name, input=input_text)

        total_usage = Usage()
        steps: list[StepResult] = []

        try:
            if after_input_appended is not None:
                await after_input_appended(self, input_msg, input_text)

            planner = PlannerAgent(
                name=f"{self.name}.planner",
                llm=self.llm,
                sub_llm=self.sub_llm,
                tools=[],
                context=self.context.fork("planner"),
                hooks=self.hooks,
                max_steps=1,
                prompt_builder=self._prompt_builder.fork(),
            )
            planner._usage_source = "subagent"
            executor = ExecutorAgent(
                name=f"{self.name}.executor",
                llm=self.llm,
                sub_llm=self.sub_llm,
                tools=self.tools,
                context=self.context.fork("executor"),
                hooks=self.hooks,
                max_steps=self._executor_max_steps,
                approval=self._executor_approval,
                approval_handler=self._executor_approval_handler,
                prompt_builder=self._prompt_builder.fork(),
            )
            executor._usage_source = "subagent"
            executor._sandbox = self._sandbox
            replanner = ReplannerAgent(
                name=f"{self.name}.replanner",
                llm=self.llm,
                sub_llm=self.sub_llm,
                tools=[],
                context=self.context.fork("replanner"),
                hooks=self.hooks,
                max_steps=1,
                prompt_builder=self._prompt_builder.fork(),
            )
            replanner._usage_source = "subagent"

            plan_result = await planner.run(input_text)
            total_usage = total_usage + plan_result.usage
            plan = self._parse_plan(plan_result.output, input_text)

            await self.emit(
                "agent.plan.created",
                agent=self.name,
                plan=plan.progress_summary,
            )

            final_output = ""
            iteration = 0

            while iteration < self.max_steps:
                iteration += 1

                if plan.is_complete:
                    break

                current = plan.current_step
                if current is None:
                    break

                current.status = "in_progress"
                self.context.state.transition(AgentState.ACTING)

                executor.context.working_memory.set("plan", plan.progress_summary)
                executor.context.working_memory.set("current_step", current.description)

                exec_result = await executor.run(current.description)
                total_usage = total_usage + exec_result.usage

                current.status = "done"
                current.result = exec_result.output

                steps.append(StepResult(thought=f"Executed step [{current.id}]"))

                self.context.state.transition(AgentState.OBSERVING)
                self.context.state.transition(AgentState.THINKING)

                replanner.context.working_memory.set("goal", plan.goal)
                replanner.context.working_memory.set("plan", plan.detailed_progress)
                replanner.context.working_memory.set(
                    "current_step",
                    f"Step [{current.id}] result: {current.result}",
                )
                pending = [s for s in plan.steps if s.status == "pending"]
                replanner.context.working_memory.set(
                    "remaining_steps",
                    f"{len(pending)} step(s) remaining"
                    if pending
                    else "All steps completed — decide if goal is achieved.",
                )

                replan_result = await replanner.run(_REPLANNER_INPUT_PROMPT)
                total_usage = total_usage + replan_result.usage

                decision = self._parse_replan_decision(replan_result.output)
                await self.emit("agent.replan.decision", agent=self.name)

                if decision.goal_achieved:
                    final_output = decision.final_answer or exec_result.output
                    break

                if decision.should_replan and decision.updated_steps:
                    previous_pending = [
                        s.model_copy(deep=True)
                        for s in plan.steps
                        if s.status not in ("done", "failed")
                    ]
                    completed = [s for s in plan.steps if s.status in ("done", "failed")]
                    plan.steps = completed + decision.updated_steps

                    reason_text = decision.reason.strip() or "No reason provided by replanner."
                    change_summary = self._summarize_plan_change(
                        previous_pending,
                        decision.updated_steps,
                    )
                    executor.context.working_memory.set(
                        "replan_reason",
                        reason_text,
                    )
                    executor.context.working_memory.set(
                        "plan_change",
                        change_summary,
                    )
                    await executor.context.short_term_memory.add_message(
                        Message.system(
                            "Plan has been updated. Review fields "
                            "'replan_reason' and 'plan_change', ignore superseded "
                            "pending steps, and execute only current_step."
                        )
                    )
                    await self.emit(
                        "agent.replan.applied",
                        agent=self.name,
                        reason=reason_text,
                        updated_steps=len(decision.updated_steps),
                    )

            else:
                final_output = (
                    f"Max iterations reached ({self.max_steps}). "
                    f"Partial result: {plan.progress_summary}"
                )

            if not final_output:
                final_output = plan.progress_summary

            await self.context.short_term_memory.add_message(Message.assistant(final_output))
            self.context.state.transition(AgentState.FINISHED)
            messages = await self.context.short_term_memory.get_context_messages()

            result = AgentResult(
                output=final_output,
                messages=messages,
                steps=steps,
                usage=total_usage,
            )

            await self.hooks.on_run_end(self.name, final_output)
            await self.emit(
                "agent.run.end",
                agent=self.name,
                output=truncate_text_by_tokens(
                    final_output,
                    max_tokens=_PLAN_EVENT_OUTPUT_MAX_TOKENS,
                    suffix="",
                ),
            )
            return result

        except Exception as e:
            await self.hooks.on_error(self.name, e)
            await self.emit("agent.run.error", agent=self.name, error=str(e))
            if not self.context.state.is_terminal:
                self.context.state.transition(AgentState.ERROR)
            raise

        finally:
            if resolved_session:
                now = datetime.now()
                ss = self.context.to_session_state(
                    resolved_session.session_id,
                    agent_name=self.name,
                )
                ss.created_at = self._ensure_session_created_at()
                ss.updated_at = now
                await resolved_session.save_state(ss)

    @staticmethod
    def _step_signature(step: PlanStep) -> tuple[str, str]:
        return (step.id.strip(), step.description.strip())

    @staticmethod
    def _format_step_list(steps: list[PlanStep]) -> str:
        if not steps:
            return "- (none)"
        return "\n".join(f"- [{step.id}] {step.description}" for step in steps)

    @classmethod
    def _summarize_plan_change(
        cls,
        previous_pending: list[PlanStep],
        updated_steps: list[PlanStep],
    ) -> str:
        previous_signatures = {cls._step_signature(step) for step in previous_pending}
        updated_signatures = {cls._step_signature(step) for step in updated_steps}

        kept = [step for step in updated_steps if cls._step_signature(step) in previous_signatures]
        removed = [
            step for step in previous_pending if cls._step_signature(step) not in updated_signatures
        ]
        added = [
            step for step in updated_steps if cls._step_signature(step) not in previous_signatures
        ]

        sections = [
            "### Plan Change",
            "#### Kept (same id+description)",
            "- done or failed steps are retained from the previous plan.",
            cls._format_step_list(kept),
            "",
            "#### Removed",
            cls._format_step_list(removed),
            "",
            "#### Added",
            cls._format_step_list(added),
        ]
        return "\n".join(sections)

    @staticmethod
    def _parse_plan(content: str, fallback_goal: str) -> Plan:
        """Parse planner output into a Plan object, with fallback."""
        try:
            data = parse_json_lenient(content)
            if isinstance(data, dict):
                raw_steps = data.get("steps", [])
                parsed_steps = [
                    PlanStep(**s)
                    if isinstance(s, dict)
                    else PlanStep(id=str(i), description=str(s))
                    for i, s in enumerate(raw_steps)
                ]
                return Plan(goal=data.get("goal", fallback_goal), steps=parsed_steps)
        except (ValueError, TypeError):
            pass
        return Plan(
            goal=fallback_goal,
            steps=[PlanStep(id="1", description=fallback_goal)],
        )

    @staticmethod
    def _parse_replan_decision(content: str) -> ReplanDecision:
        """Parse replanner output into a ReplanDecision."""
        try:
            data = parse_json_lenient(content)
            if isinstance(data, dict):
                updated = None
                if data.get("updated_steps"):
                    updated = [
                        PlanStep(**s)
                        if isinstance(s, dict)
                        else PlanStep(id=str(i), description=str(s))
                        for i, s in enumerate(data["updated_steps"])
                    ]
                return ReplanDecision(
                    goal_achieved=data.get("goal_achieved", False),
                    should_replan=data.get("should_replan", False),
                    reason=data.get("reason", ""),
                    updated_steps=updated,
                    final_answer=data.get("final_answer"),
                )
        except (ValueError, TypeError):
            pass
        return ReplanDecision()


PlanAgent = PlanAndExecuteAgent
