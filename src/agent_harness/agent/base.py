"""Base agent class for agent_harness."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_harness.background import BackgroundTask
    from agent_harness.prompt.system_builder import SystemPromptBuilder
    from agent_harness.sandbox.backend import SandboxBackend
    from agent_harness.session.base import BaseSession, SessionState

from pydantic import BaseModel, Field

from agent_harness.approval import (
    ApprovalAction,
    ApprovalDecision,
    ApprovalHandler,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalResult,
    resolve_approval,
    resolve_approval_handler,
)
from agent_harness.approval.rules import extract_resource
from agent_harness.context.context import AgentContext
from agent_harness.context.state import AgentState
from agent_harness.core.config import HarnessConfig
from agent_harness.core.errors import MaxStepsExceededError
from agent_harness.core.event import EventEmitter
from agent_harness.core.message import Message, Role, ToolCall, ToolResult
from agent_harness.hooks import DefaultHooks, resolve_hooks
from agent_harness.llm import create_llm
from agent_harness.llm.base import BaseLLM
from agent_harness.llm.types import LLMResponse, LLMRetryInfo, StreamDelta, Usage, UsageSource
from agent_harness.memory.short_term import CallSnapshot, SectionWeights
from agent_harness.tool.base import BaseTool, ToolSchema
from agent_harness.tool.executor import ToolExecutor
from agent_harness.tool.registry import ToolRegistry
from agent_harness.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)


class StepResult(BaseModel):
    """Result of a single agent step."""

    thought: str | None = None
    action: list[ToolCall] | None = None
    observation: list[ToolResult] | None = None
    response: str | None = None  # final response if step produced one


class AgentResult(BaseModel):
    """Final result of an agent run."""

    output: str
    messages: list[Message] = Field(default_factory=list)
    steps: list[StepResult] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)

    @property
    def step_count(self) -> int:
        return len(self.steps)


class BaseAgent(ABC, EventEmitter):
    """Abstract base class for all agents.

    Provides the run loop, tool execution, and lifecycle management.
    Subclasses implement step() to define their reasoning strategy.

    Args:
        name: Unique agent name.
        llm: LLM provider for generation.
        tools: List of available tools.
        context: Agent runtime context.
        hooks: Lifecycle hooks (inherits DefaultHooks). When tracing.enabled=True
            and hooks is not provided, TracingHooks is auto-created from config.
        max_steps: Maximum steps before forced termination.
        system_prompt: System prompt for the agent.
        use_long_term_memory: If True, call_llm() queries long-term memory by default.
        config: Optional config used to create context when context is not provided.
    """

    def __init__(
        self,
        name: str,
        llm: BaseLLM | None = None,
        tools: list[BaseTool] | None = None,
        context: AgentContext | None = None,
        hooks: DefaultHooks | None = None,
        max_steps: int = 100,
        system_prompt: str = "",
        use_long_term_memory: bool = False,
        stream: bool = True,
        *,
        config: HarnessConfig | None = None,
        approval: ApprovalPolicy | None = None,
        approval_handler: ApprovalHandler | None = None,
        prompt_builder: SystemPromptBuilder | None = None,
        sandbox: SandboxBackend | None = None,
    ) -> None:
        from agent_harness.prompt.runtime_context import RuntimeContextProvider
        from agent_harness.prompt.sections import create_default_builder, make_intro_section

        self.name = name
        if context is not None:
            self.context = context
        else:
            self.context = AgentContext.create(config=config)
        self.llm = llm or create_llm(self.context.config)
        self.hooks = resolve_hooks(hooks, self.context.config)
        self.max_steps = max_steps
        self.use_long_term_memory = use_long_term_memory
        self._stream = stream
        self._run_usage = Usage()
        self._usage_source: UsageSource = "main"
        self._session_created_at: datetime | None = None
        self._session_metadata_extras: dict[str, Any] = {}

        # Approval setup
        self._approval = resolve_approval(approval, self.context.config)
        self._approval_handler: ApprovalHandler = resolve_approval_handler(approval_handler)

        # Set up tool registry and executor
        from agent_harness.tool.base import AgentAware

        self.tool_registry = ToolRegistry()
        for t in tools or []:
            tool = t.clone()
            self.tool_registry.register(tool)
            if isinstance(tool, AgentAware):
                tool.bind_agent(self)
        self.tool_executor = ToolExecutor(
            self.tool_registry,
            config=self.context.config,
        )

        # System prompt builder
        if prompt_builder is not None:
            self._prompt_builder = prompt_builder
            if system_prompt:
                self._prompt_builder.register(make_intro_section(system_prompt))
        else:
            self._prompt_builder = create_default_builder(system_prompt)
        self.system_prompt = self._prompt_builder.build(self._make_builder_context())

        # Runtime context provider (ephemeral layer)
        self._runtime_ctx = RuntimeContextProvider()

        # Wire event bus
        self.set_event_bus(self.context.event_bus)
        self.tool_executor.set_event_bus(self.context.event_bus)
        self.llm.set_event_bus(self.context.event_bus)

        # Loop detection
        from agent_harness.utils.loop_detector import LoopDetector

        self._loop_detector = LoopDetector()
        self._pending_loop_warning: Message | None = None

        # Background task manager
        from agent_harness.background import BackgroundTaskManager

        self._bg_manager = BackgroundTaskManager()

        # Sandbox, LocalBackend when disabled, DockerBackend when enabled)
        from agent_harness.sandbox import SandboxManager, resolve_sandbox

        self._sandbox: SandboxManager = resolve_sandbox(sandbox, self.context.config)

        # Context compression setup
        if (
            self.context.config.memory.strategy == "summarize"
            and self.context.short_term_memory.compressor is None
        ):
            self._init_compressor()

    def _init_compressor(self) -> None:
        from agent_harness.memory.compressor import create_compressor

        comp_cfg = self.context.config.memory.compression
        summary_llm = self.llm
        if comp_cfg.summary_model:
            summary_llm = create_llm(
                self.context.config,
                model_override=comp_cfg.summary_model,
            )
        compressor = create_compressor(
            llm=summary_llm,
            memory_config=self.context.config.memory,
            model=self.context.config.llm.model,
        )
        self.context.short_term_memory.compressor = compressor
        self.context._compressor = compressor

    def _make_builder_context(self) -> dict[str, Any]:
        """Prepare context dict for SystemPromptBuilder.build()."""
        from pathlib import Path

        skill_loader = None
        for tool in self.tools:
            if tool.name == "skill_tool" and hasattr(tool, "loader"):
                skill_loader = tool.loader
                break
        return {
            "tools": self.tools,
            "config": self.context.config,
            "cwd": str(Path.cwd()),
            "skill_loader": skill_loader,
        }

    async def _collect_background_results(self) -> list[Any]:
        """Harvest completed background tasks and inject results into memory."""
        completed = self._bg_manager.collect_completed()
        for task in completed:
            if task.status == "completed" and task.result:
                content = (
                    f"[Background Task Completed] {task.task_id} ({task.tool_name}): "
                    f"{task.description}\n{task.result.summary}"
                )
                if task.result.output_path:
                    content += f"\nFull output: {task.result.output_path}"
            elif task.status == "failed":
                content = (
                    f"[Background Task Failed] {task.task_id} ({task.tool_name}): "
                    f"{task.description}\nError: {task.error}"
                )
            else:
                continue
            await self.context.short_term_memory.add_message(
                Message.system(content, metadata={"is_background_result": True})
            )
        return completed

    async def _check_loop(self, tool_calls: list[ToolCall]) -> None:
        """Record tool calls and check for repetitive loop pattern."""
        self._loop_detector.record(tool_calls)
        signal = self._loop_detector._check()
        if signal.level == "break":
            from agent_harness.core.errors import LoopDetectedError

            names = [tc.name for tc in tool_calls]
            raise LoopDetectedError(
                f"Agent '{self.name}' stuck in loop: "
                f"{signal.streak} consecutive identical calls to {names}",
                streak=signal.streak,
            )
        warning = self._loop_detector.build_warning_message(signal)
        if warning:
            self._pending_loop_warning = warning
            logger.debug(
                "Loop %s warning for agent '%s': streak=%d",
                signal.level,
                self.name,
                signal.streak,
            )

    async def _sync_system_prompt(self) -> None:
        if not self.system_prompt:
            return
        stm = self.context.short_term_memory
        msgs = stm._messages
        changed = False
        if not msgs:
            msgs.append(Message.system(self.system_prompt))
            changed = True
        elif msgs[0].role == Role.SYSTEM:
            if msgs[0].content != self.system_prompt:
                msgs[0] = msgs[0].model_copy(update={"content": self.system_prompt})
                changed = True
        else:
            msgs.insert(0, Message.system(self.system_prompt))
            changed = True
        if changed:
            stm.clear_call_snapshot()

    def _compute_section_weights(
        self,
        messages: list[Message],
        tool_schemas: list[ToolSchema],
    ) -> SectionWeights:
        main_sys = self.system_prompt or None
        consumed = False
        dyn_parts: list[str] = []
        hist_parts: list[str] = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                if not consumed and main_sys is not None and (msg.content or "") == main_sys:
                    consumed = True
                    continue
                dyn_parts.append(msg.content or "")
            else:
                hist_parts.append(msg.content or "")

        tools_text = "\n".join(
            json.dumps(s.to_openai_format(), ensure_ascii=False) for s in tool_schemas
        )
        model = self.llm.model_name
        return SectionWeights(
            system_prompt=count_tokens(self.system_prompt or "", model),
            tools_schema=count_tokens(tools_text, model),
            dynamic_system=count_tokens("\n".join(dyn_parts), model),
            history=count_tokens("\n".join(hist_parts), model),
        )

    @property
    def tools(self) -> list[BaseTool]:
        tools: list[BaseTool] = self.tool_registry.list_tools()
        return tools

    @property
    def tool_schemas(self) -> list[ToolSchema]:
        schemas: list[ToolSchema] = self.tool_registry.get_schemas()
        return schemas

    def _bind_session_id(self, session_id: str) -> None:
        compressor = self.context.short_term_memory.compressor
        if compressor:
            compressor.bind_session(session_id)
        self._bg_manager.bind_session(session_id)

    def _reset_stateful_tools(self) -> None:
        for tool in self.tool_registry.list_tools():
            tool.reset_state()

    async def apply_session_state(self, state: SessionState) -> None:
        self._bind_session_id(state.session_id)
        await self.context.restore_from_state(state, self.system_prompt)
        self._session_created_at = state.created_at
        compressor = self.context.short_term_memory.compressor
        if compressor:
            compressor.restore_runtime_state(state.messages)
        self._reset_stateful_tools()
        await self.tool_registry.restore_states(
            state.metadata.get("_tool_states", {}), self.hooks, self.name,
        )
        self._approval.import_session_grants(state.metadata.get("_approval_grants", {}))
        if "_approval_mode" in state.metadata:
            self._approval.set_mode(state.metadata["_approval_mode"])

    async def reset_session_state(self, new_id: str) -> None:
        await self.context.short_term_memory.clear()
        await self.context.working_memory.clear()
        self.context.variables._agent_store.clear()
        self.context.variables._global_store.clear()
        self.context.context_patches.clear()
        self._reset_stateful_tools()
        self._approval.reset_session()
        self.context.state.reset()
        compressor = self.context.short_term_memory.compressor
        if compressor:
            compressor.restore_runtime_state([])
        self._bind_session_id(new_id)
        self._run_usage = Usage()
        self._session_created_at = None
        self._session_metadata_extras.clear()

    async def run(
        self,
        input: str | Message,
        *,
        session: str | BaseSession | None = None,
        after_input_appended: (Callable[[BaseAgent, Message, str], Awaitable[None]] | None) = None,
    ) -> AgentResult:
        """Main execution loop.

        Repeatedly calls step() until:
        1. step() returns a final response, or
        2. max_steps is reached.

        Safe to call multiple times — state is reset automatically when
        the agent is in a terminal state (FINISHED or ERROR).

        Pass session (str or BaseSession) to enable persistence across restarts.

        ``after_input_appended`` fires once inside the run-level try block,
        after input append + on_run_start, before the first step.
        """
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

        # Normalize input
        if isinstance(input, str):
            input_msg = Message.user(input)
            input_text = input
        else:
            input_msg = input
            input_text = input.content or ""

        # Initialize context
        await self._sync_system_prompt()
        await self.context.short_term_memory.add_message(input_msg)
        self.context.state.transition(AgentState.THINKING)

        await self.hooks.on_run_start(self.name, input_text)
        await self.emit("agent.run.start", agent=self.name, input=input_text)

        steps: list[StepResult] = []
        self._run_usage = Usage()
        self._loop_detector.reset()
        self._pending_loop_warning = None
        final_output = ""

        try:
            if after_input_appended is not None:
                await after_input_appended(self, input_msg, input_text)

            for step_num in range(1, self.max_steps + 1):
                # Harvest completed background tasks
                await self._collect_background_results()

                await self.hooks.on_step_start(self.name, step_num)
                await self.emit("agent.step.start", agent=self.name, step=step_num)

                step_result = await self.step()
                steps.append(step_result)

                await self.hooks.on_step_end(self.name, step_num)
                await self.emit("agent.step.end", agent=self.name, step=step_num)

                # Loop detection (after step hooks to ensure span closure)
                if step_result.action:
                    await self._check_loop(step_result.action)

                if step_result.response is not None:
                    final_output = step_result.response
                    self.context.state.transition(AgentState.FINISHED)
                    break
            else:
                # max_steps exceeded
                self.context.state.transition(AgentState.ERROR)
                raise MaxStepsExceededError(f"Agent '{self.name}' exceeded {self.max_steps} steps")

        except Exception as e:
            await self.hooks.on_error(self.name, e)
            await self.emit("agent.run.error", agent=self.name, error=str(e))
            if not isinstance(e, MaxStepsExceededError):
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
                ss.created_at = self._session_created_at or now
                ss.updated_at = now
                ss.metadata.update(self._session_metadata_extras)
                tool_states = self.tool_registry.save_states()
                if tool_states:
                    ss.metadata["_tool_states"] = tool_states
                ss.metadata["_approval_mode"] = self._approval.mode
                grants = self._approval.export_session_grants()
                if grants:
                    ss.metadata["_approval_grants"] = grants
                await resolved_session.save_state(ss)

        messages = await self.context.short_term_memory.get_context_messages()
        result = AgentResult(
            output=final_output,
            messages=messages,
            steps=steps,
            usage=self._run_usage,
        )

        await self.hooks.on_run_end(self.name, final_output)
        await self.emit("agent.run.end", agent=self.name, output=final_output, steps=len(steps))
        return result

    async def chat(
        self,
        *,
        session: str | BaseSession | None = None,
        prompt: str = "> ",
        exit_commands: tuple[str, ...] = ("exit", "quit", "bye"),
    ) -> None:
        """Interactive REPL with auto-trigger on background task completion."""
        import asyncio as _asyncio
        import readline  # noqa: F401 — enables arrow keys, history, proper backspace

        from agent_harness.utils.input_mux import mux_input

        self._approval.reset_session()

        input_task: _asyncio.Task[str] | None = None

        try:
            while True:
                # Collect completed background tasks before waiting
                collected = await self._collect_background_results()
                if collected:
                    if input_task and not input_task.done():
                        input_task.cancel()
                        input_task = None
                    print("\n[Background task completed]")
                    try:
                        await self.run(
                            Message.system(
                                "[Background Task Notification] "
                                "Process the completed background task results.",
                                metadata={"is_background_result": True},
                            ),
                            session=session,
                        )
                    except Exception:
                        pass
                    continue

                # Only create new input task if previous one is done
                if input_task is None or input_task.done():
                    input_task = _asyncio.create_task(mux_input(prompt, priority=0))

                # Race: user input vs background completion
                wait_set: set[_asyncio.Task[Any]] = {input_task}
                bg_wait_task: _asyncio.Task[Any] | None = None
                if self._bg_manager.has_running():
                    bg_wait_task = _asyncio.create_task(self._bg_manager.wait_next())
                    wait_set.add(bg_wait_task)

                done, _ = await _asyncio.wait(wait_set, return_when=_asyncio.FIRST_COMPLETED)

                # Cancel bg observer if it didn't fire
                if bg_wait_task and bg_wait_task not in done:
                    bg_wait_task.cancel()

                if input_task in done:
                    user_input = input_task.result().strip()
                    input_task = None
                    if not user_input:
                        continue
                    if user_input.lower() in exit_commands:
                        break
                    try:
                        result = await self.run(user_input, session=session)
                        if not self._stream:
                            print(result.output)
                    except Exception as e:
                        print(f"Error: {e}")

                elif bg_wait_task and bg_wait_task in done:
                    if input_task and not input_task.done():
                        input_task.cancel()
                        input_task = None
                    print("\n[Background task completed]")
                    await self._collect_background_results()
                    try:
                        await self.run(
                            Message.system(
                                "[Background Task Notification] "
                                "Process the completed background task results.",
                                metadata={"is_background_result": True},
                            ),
                            session=session,
                        )
                    except Exception:
                        pass

        except (KeyboardInterrupt, EOFError, _asyncio.CancelledError):
            pass

        # Cleanup: cancel pending input and background tasks
        if input_task and not input_task.done():
            input_task.cancel()
        if self._bg_manager.has_running():
            count = self._bg_manager.cancel_all()
            if count:
                print(f"\nCancelled {count} running background task(s).")
        await self._bg_manager.shutdown()

    @abstractmethod
    async def step(self) -> StepResult:
        """Execute a single reasoning step.

        Subclasses implement their strategy here:
        - ReActAgent: think -> act -> observe
        - PlanAgent: plan -> execute step
        - ConversationalAgent: generate response

        Returns:
            StepResult. If response is not None, the run loop ends.
        """
        ...

    async def call_llm(
        self,
        messages: list[Message] | None = None,
        tools: list[ToolSchema] | None = None,
        use_long_term: bool | None = None,
        long_term_query: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call the LLM with current context messages or provided messages.

        Args:
            messages: Override messages. If None, uses short-term memory.
            tools: Override tool schemas. If None, uses registered tools.
            use_long_term: Query long-term memory and inject results.
                If None, falls back to self.use_long_term_memory.
            long_term_query: Custom query for long-term retrieval.
            **kwargs: Passed through to the LLM call.
        """
        if use_long_term is None:
            use_long_term = self.use_long_term_memory

        stm = self.context.short_term_memory

        await self.context.maybe_auto_compress(
            self.hooks, self.name,
            authoritative_input=stm.displayed_input_tokens,
        )

        extra_sys: list[Message] = []

        for patch in self.context.context_patches:
            if patch.at != "system":
                continue
            msg = patch.build()
            if msg:
                extra_sys.append(msg)

        sorted_tools = sorted(self.tools, key=lambda t: t.context_order)
        for tool in sorted_tools:
            ctx_msg = tool.build_context_message()
            if ctx_msg:
                extra_sys.append(ctx_msg)

        runtime_msg = self._runtime_ctx.build_context_message()
        if runtime_msg:
            extra_sys.append(runtime_msg)

        messages = await self.context.build_llm_messages(
            base_messages=messages,
            include_working=True,
            include_long_term=use_long_term,
            long_term_query=long_term_query,
            extra_system_messages=extra_sys or None,
        )

        for patch in self.context.context_patches:
            if patch.at != "tail":
                continue
            msg = patch.build()
            if msg:
                messages.append(msg)

        if self._pending_loop_warning:
            messages.append(self._pending_loop_warning)
            self._pending_loop_warning = None

        if tools is None and self.tool_schemas:
            tools = self.tool_schemas

        weights = self._compute_section_weights(messages, tools or [])

        await self.hooks.on_llm_call(self.name, messages)
        if self.context.state.current != AgentState.THINKING:
            self.context.state.transition(AgentState.THINKING)

        async def _on_retry(info: LLMRetryInfo) -> None:
            await self.hooks.on_llm_retry(self.name, info)

        if self._stream:

            async def _on_delta(delta: StreamDelta) -> None:
                await self.hooks.on_llm_stream_delta(self.name, delta)

            response = await self.llm.stream_with_events(
                messages,
                tools=tools,
                on_delta=_on_delta,
                on_retry=_on_retry,
                **kwargs,
            )
        else:
            response = await self.llm.generate_with_events(
                messages,
                tools=tools,
                on_retry=_on_retry,
                **kwargs,
            )

        self._run_usage = self._run_usage + response.usage
        self.context.usage_meter.record(
            response.usage,
            model=self.llm.model_name,
            source=self._usage_source,
        )

        await stm.add_message(response.message)

        stm.record_call(CallSnapshot(
            input_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            cache_read=response.usage.cache_read_tokens,
            cache_creation=response.usage.cache_creation_tokens,
            reasoning_tokens=response.usage.reasoning_tokens,
            model=self.llm.model_name,
            message_count=len(stm._messages),
            section_weights=weights,
        ))
        return response

    async def execute_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tool calls with optional human approval."""
        self.context.state.transition(AgentState.ACTING)

        approved: list[ToolCall] = []
        denied_results: list[ToolResult] = []

        for tc in tool_calls:
            # Resource extraction
            tool_obj = (
                self.tool_executor.registry.get(tc.name)
                if self.tool_executor.registry.has(tc.name)
                else None
            )
            resource_key = tool_obj.approval_resource_key if tool_obj else None
            default_val: str | None = None
            if tool_obj and resource_key:
                props = tool_obj.get_schema().parameters.get("properties", {})
                prop = props.get(resource_key, {})
                if "default" in prop:
                    default_val = str(prop["default"])
            resource, kind = extract_resource(
                tc.name,
                tc.arguments,
                resource_key,
                default=default_val,
            )

            action = self._approval.check(tc, resource=resource, kind=kind)

            if action == ApprovalAction.EXECUTE:
                await self.hooks.on_tool_call(self.name, tc)
                approved.append(tc)

            elif action == ApprovalAction.DENY:
                await self.hooks.on_approval_result(
                    self.name,
                    ApprovalResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        decision=ApprovalDecision.DENY,
                        reason="Not allowed by policy.",
                    ),
                )
                denied_results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        content=f"Tool '{tc.name}' is not allowed by policy.",
                        is_error=True,
                    )
                )

            else:  # ApprovalAction.ASK
                request = ApprovalRequest(
                    tool_call=tc,
                    agent_name=self.name,
                    resource=resource,
                    resource_kind=kind,
                )
                await self.hooks.on_approval_request(self.name, request)

                try:
                    result = await self._approval_handler.request_approval(request)
                except Exception as e:
                    logger.warning("Approval handler failed for '%s': %s", tc.name, e)
                    result = ApprovalResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        decision=ApprovalDecision.DENY,
                        reason=f"Approval handler error: {e}",
                    )

                await self.hooks.on_approval_result(self.name, result)

                if result.decision == ApprovalDecision.ALLOW_ONCE:
                    await self.hooks.on_tool_call(self.name, tc)
                    approved.append(tc)
                elif result.decision == ApprovalDecision.ALLOW_SESSION:
                    self._approval.grant_session(
                        tc.name,
                        resource=resource,
                        kind=kind,
                    )
                    await self.hooks.on_tool_call(self.name, tc)
                    approved.append(tc)
                else:
                    reason = result.reason or "Denied by user."
                    denied_results.append(
                        ToolResult(
                            tool_call_id=tc.id,
                            content=f"Tool '{tc.name}' was denied: {reason}",
                            is_error=True,
                        )
                    )

        # ── Hooks: denied immediately, approved in completion order ──
        result_map: dict[str, ToolResult] = {}

        for r in denied_results:
            result_map[r.tool_call_id] = r
            await self.hooks.on_tool_result(self.name, r)

        if approved:
            async for result in self.tool_executor.execute_stream(approved):
                result_map[result.tool_call_id] = result
                await self.hooks.on_tool_result(self.name, result)

        # ── Memory: write in original call order (transcript stable) ──
        self.context.state.transition(AgentState.OBSERVING)

        ordered: list[ToolResult] = []
        for tc in tool_calls:
            r = result_map[tc.id]
            ordered.append(r)
            await self.context.short_term_memory.add_message(
                Message.tool(
                    tool_call_id=r.tool_call_id,
                    content=r.content,
                    is_error=r.is_error,
                )
            )

        # Notify stateful tools (in call order, after memory write)
        for tc in tool_calls:
            r = result_map[tc.id]
            if not r.is_error and self.tool_registry.has(tc.name):
                tool = self.tool_registry.get(tc.name)
                await tool.notify_state(self.hooks, self.name)

        return ordered

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} tools={len(self.tools)}>"
