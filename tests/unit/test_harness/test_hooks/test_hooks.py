"""Tests for TracingHooks — span creation and lifecycle management."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from agent_harness.hooks import (
    CompositeHooks,
    DefaultHooks,
    ProgressHooks,
    TracingHooks,
    resolve_hooks,
)
from agent_harness.core.config import HarnessConfig, TracingConfig
from agent_harness.core.message import MessageChunk, ToolCall, ToolResult
from agent_harness.core.errors import LLMConnectionError
from agent_harness.llm.types import FinishReason, LLMRetryInfo, StreamDelta, Usage
from agent_harness.tracing.tracer import Span
from agent_harness.utils.token_counter import count_tokens


class TestTracingHooksLifecycle:
    @pytest.mark.asyncio
    async def test_run_lifecycle_creates_and_finishes_root_span(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_run_start("test_agent", "hello")
        assert hooks._root_span is not None
        assert hooks._root_span.kind == "agent"
        assert hooks._root_span.name == "agent.test_agent"

        await hooks.on_run_end("test_agent", "goodbye")
        assert hooks._root_span is None
        assert len(hooks._all_spans) == 1
        assert hooks._depth == 0
        assert hooks._all_spans[0].end_time is not None

    @pytest.mark.asyncio
    async def test_step_spans_are_children(self) -> None:
        from agent_harness.hooks.tracing import _active_step_span

        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_run_start("agent", "input")

        await hooks.on_step_start("agent", 1)
        step_span = _active_step_span.get(None)
        assert step_span is not None
        assert step_span.parent_span_id == hooks._root_span.span_id

        await hooks.on_step_end("agent", 1)
        assert _active_step_span.get(None) is None
        assert step_span in hooks._all_spans

        await hooks.on_run_end("agent", "done")

    @pytest.mark.asyncio
    async def test_llm_and_tool_events_recorded(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_run_start("agent", "input")

        await hooks.on_step_start("agent", 1)
        await hooks.on_llm_call("agent", [{"role": "user", "content": "hi"}])
        await hooks.on_tool_call("agent", ToolCall(name="search", arguments={"query": "hi"}))
        await hooks.on_tool_result("agent", type("TR", (), {"output": "result"})())
        await hooks.on_step_end("agent", 1)

        step_span = hooks._all_spans[0]
        event_names = [e.name for e in step_span.events]
        assert "llm_call" in event_names
        assert "tool_call" in event_names
        assert "tool_result" in event_names

        await hooks.on_run_end("agent", "done")

    @pytest.mark.asyncio
    async def test_text_attributes_truncated_by_tokens(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        long_text = "signal " * 600

        await hooks.on_run_start("agent", long_text)
        assert hooks._root_span is not None
        run_input = hooks._root_span.attributes["input"]
        assert count_tokens(run_input) <= 64

        await hooks.on_tool_result("agent", type("TR", (), {"output": long_text})())
        assert hooks._root_span.events
        last_event = hooks._root_span.events[-1]
        assert last_event.name == "tool_result"
        assert count_tokens(last_event.attributes["output"]) <= 64

        await hooks.on_run_end("agent", long_text)
        run_span = hooks._all_spans[0]
        assert count_tokens(run_span.attributes["output"]) <= 64

    @pytest.mark.asyncio
    async def test_run_start_prints_start_marker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        start_calls: list[tuple[str, str, int, dict[str, object] | None]] = []

        def capture_start(
            *,
            kind: str,
            name: str,
            indent: int = 0,
            attributes: Mapping[str, object] | None = None,
        ) -> None:
            payload = dict(attributes) if attributes is not None else None
            start_calls.append((kind, name, indent, payload))

        monkeypatch.setattr(hooks._console, "print_start", capture_start)

        await hooks.on_run_start("agent", "hello world")

        assert len(start_calls) == 1
        kind, name, indent, attrs = start_calls[0]
        assert kind == "agent"
        assert name == "agent.agent"
        assert indent == 0
        assert attrs is not None
        assert attrs.get("input") == "hello world"

    @pytest.mark.asyncio
    async def test_finish_span_uses_runtime_stack_indent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        export_calls: list[tuple[str, int]] = []

        def capture_export(span: Span, *, indent: int) -> None:
            export_calls.append((span.name, indent))

        monkeypatch.setattr(hooks._console, "export_one", capture_export)

        await hooks.on_run_start("agent", "input")
        await hooks.on_step_start("agent", 1)
        await hooks.on_step_end("agent", 1)
        await hooks.on_run_end("agent", "done")

        # step span no longer uses export_one (inline output instead)
        assert ("step.1", 1) not in export_calls
        assert ("agent.agent", 0) in export_calls

    @pytest.mark.asyncio
    async def test_error_marks_span(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_run_start("agent", "input")

        error = ValueError("test error")
        await hooks.on_error("agent", error)

        assert hooks._root_span is None
        assert len(hooks._all_spans) == 1
        assert hooks._all_spans[0].status == "error"
        assert "test error" in hooks._all_spans[0].error_message

    @pytest.mark.asyncio
    async def test_subagent_error_marks_span_status(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_run_start("agent", "input")
        await hooks.on_subagent_start(
            "agent", "agent.sub.research.1", "research", "do X", "prompt",
        )

        await hooks.on_subagent_end(
            "agent", "agent.sub.research.1", "research", "do X",
            0, 0, 12.0, error="boom",
        )

        sub = next(s for s in hooks._all_spans if s.kind == "subagent")
        assert sub.status == "error"
        assert sub.error_message == "boom"

    @pytest.mark.asyncio
    async def test_subagent_success_keeps_ok_status(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_run_start("agent", "input")
        await hooks.on_subagent_start(
            "agent", "agent.sub.research.1", "research", "do X", "prompt",
        )

        await hooks.on_subagent_end(
            "agent", "agent.sub.research.1", "research", "do X", 3, 5, 12.0,
        )

        sub = next(s for s in hooks._all_spans if s.kind == "subagent")
        assert sub.status == "ok"
        assert sub.error_message is None

    @pytest.mark.asyncio
    async def test_tool_call_records_sanitized_args(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_run_start("agent", "input")
        await hooks.on_step_start("agent", 1)

        await hooks.on_tool_call(
            "agent",
            ToolCall(
                name="web_search",
                arguments={
                    "query": "renewable energy trend 2025",
                    "api_key": "secret-token",
                    "max_results": 3,
                },
            ),
        )
        await hooks.on_step_end("agent", 1)

        step_span = hooks._all_spans[0]
        event = next(e for e in step_span.events if e.name == "tool_call")
        assert event.attributes["tool"] == "web_search"
        args = event.attributes["args"]
        assert isinstance(args, dict)
        assert args["api_key"] == "[REDACTED]"
        assert args["query"] == "renewable energy trend 2025"
        assert args["max_results"] == "3"

        await hooks.on_run_end("agent", "done")

    @pytest.mark.asyncio
    async def test_tool_call_argument_value_truncated_by_tokens(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_run_start("agent", "input")
        await hooks.on_step_start("agent", 1)

        long_query = "signal " * 400
        await hooks.on_tool_call(
            "agent",
            ToolCall(name="web_search", arguments={"query": long_query}),
        )
        await hooks.on_step_end("agent", 1)

        step_span = hooks._all_spans[0]
        event = next(e for e in step_span.events if e.name == "tool_call")
        args = event.attributes["args"]
        assert isinstance(args, dict)
        assert count_tokens(args["query"]) <= 20

        await hooks.on_run_end("agent", "done")


class TestTracingHooksOrchestration:
    @pytest.mark.asyncio
    async def test_pipeline_spans(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_pipeline_start("my_pipeline")
        assert len(hooks._span_stack) == 1
        assert "pipeline" in hooks._span_stack[0].name

        await hooks.on_pipeline_end("my_pipeline")
        assert len(hooks._all_spans) == 1

    @pytest.mark.asyncio
    async def test_team_spans(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_team_start("my_team", "supervisor")
        assert hooks._span_stack[-1].attributes["mode"] == "supervisor"

        await hooks.on_team_end("my_team", "supervisor")
        assert len(hooks._all_spans) == 1

    @pytest.mark.asyncio
    async def test_dag_spans_with_nodes(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_dag_start("my_dag")

        await hooks.on_dag_node_start("node_1")
        await hooks.on_dag_node_end("node_1")

        await hooks.on_dag_end("my_dag")
        assert len(hooks._all_spans) == 2  # dag_node + dag

    @pytest.mark.asyncio
    async def test_orchestration_starts_print_markers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        start_calls: list[str] = []

        def capture_start(
            *,
            kind: str,
            name: str,
            indent: int = 0,
            attributes: Mapping[str, object] | None = None,
        ) -> None:
            del kind, indent, attributes
            start_calls.append(name)

        monkeypatch.setattr(hooks._console, "print_start", capture_start)

        await hooks.on_pipeline_start("pipe")
        await hooks.on_pipeline_end("pipe")

        await hooks.on_dag_start("dag")
        await hooks.on_dag_node_start("node_1")
        await hooks.on_dag_node_end("node_1")
        await hooks.on_dag_end("dag")

        await hooks.on_team_start("team", "supervisor")
        await hooks.on_team_end("team", "supervisor")

        assert "pipeline.pipe" in start_calls
        assert "dag.dag" in start_calls
        assert "dag_node.node_1" in start_calls
        assert "team.team" in start_calls

    @pytest.mark.asyncio
    async def test_nested_tracing_preserves_outer_spans(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_pipeline_start("pipe")

        await hooks.on_run_start("agent1", "input1")
        await hooks.on_run_end("agent1", "output1")
        assert hooks._depth == 1

        await hooks.on_run_start("agent2", "input2")
        await hooks.on_run_end("agent2", "output2")
        assert hooks._depth == 1

        await hooks.on_pipeline_end("pipe")
        assert hooks._depth == 0
        assert len(hooks._all_spans) == 3
        names = {span.name for span in hooks._all_spans}
        assert "pipeline.pipe" in names
        assert "agent.agent1" in names
        assert "agent.agent2" in names

    @pytest.mark.asyncio
    async def test_nested_error_does_not_break_outer_orchestration_span(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_pipeline_start("pipe")
        await hooks.on_run_start("agent1", "input1")
        await hooks.on_step_start("agent1", 1)

        await hooks.on_error("agent1", RuntimeError("boom"))
        assert hooks._depth == 1
        assert [span.name for span in hooks._span_stack] == ["pipeline.pipe"]

        await hooks.on_pipeline_end("pipe")
        assert hooks._depth == 0
        assert hooks._span_stack == []

        names = {span.name for span in hooks._all_spans}
        assert "pipeline.pipe" in names
        assert "agent.agent1" in names
        assert "step.1" in names


class TestResolveHooks:
    def test_explicit_hooks_always_respected(self) -> None:
        explicit = TracingHooks(trace_dir="/tmp/explicit")
        cfg = HarnessConfig(tracing=TracingConfig(enabled=False))

        resolved = resolve_hooks(explicit, cfg)

        assert resolved is explicit

    def test_enabled_tracing_uses_explicit_hooks(self) -> None:
        explicit = DefaultHooks()
        cfg = HarnessConfig(tracing=TracingConfig(enabled=True))

        resolved = resolve_hooks(explicit, cfg)

        assert resolved is explicit

    def test_enabled_tracing_autocreates_tracing_hooks(self) -> None:
        cfg = HarnessConfig(tracing=TracingConfig(enabled=True, export_path="/tmp/from_config"))

        resolved = resolve_hooks(None, cfg)

        assert isinstance(resolved, TracingHooks)
        assert resolved._trace_dir == "/tmp/from_config"

    def test_enabled_tracing_reuses_cached_hooks(self) -> None:
        cfg = HarnessConfig(tracing=TracingConfig(enabled=True, export_path="/tmp/from_config"))

        first = resolve_hooks(None, cfg)
        second = resolve_hooks(None, cfg)

        assert isinstance(first, TracingHooks)
        assert first is second


class TestOrchestrationSpanKind:
    @pytest.mark.asyncio
    async def test_orchestration_spans_use_orchestration_kind(self) -> None:
        """Pipeline, DAG, DAG node, and team spans should have kind='orchestration'."""
        hooks = TracingHooks(trace_dir="/tmp/test_traces")

        await hooks.on_pipeline_start("pipe")
        assert hooks._span_stack[-1].kind == "orchestration"
        await hooks.on_pipeline_end("pipe")

        await hooks.on_dag_start("dag")
        assert hooks._span_stack[-1].kind == "orchestration"
        await hooks.on_dag_node_start("n1")
        assert hooks._span_stack[-1].kind == "orchestration"
        await hooks.on_dag_node_end("n1")
        await hooks.on_dag_end("dag")

        await hooks.on_team_start("team", "supervisor")
        assert hooks._span_stack[-1].kind == "orchestration"
        await hooks.on_team_end("team", "supervisor")


class TestParallelDAGNodes:
    @pytest.mark.asyncio
    async def test_parallel_dag_nodes_are_siblings(self) -> None:
        """Parallel DAG nodes should share the DAG span as parent and have equal depth."""
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_dag_start("dag")
        dag_span = hooks._dag_span_stack[-1]

        await hooks.on_dag_node_start("a")
        span_a = hooks._dag_node_span_stack[-1]
        await hooks.on_dag_node_start("b")
        span_b = hooks._dag_node_span_stack[-1]

        assert span_a.parent_span_id == dag_span.span_id
        assert span_b.parent_span_id == dag_span.span_id
        assert hooks._span_depth_map[span_a.span_id] == hooks._span_depth_map[span_b.span_id]

        await hooks.on_dag_node_end("b")
        await hooks.on_dag_node_end("a")
        await hooks.on_dag_end("dag")

    @pytest.mark.asyncio
    async def test_agent_inside_dag_node_uses_node_as_parent(self) -> None:
        """An agent started inside a DAG node should use the node span as parent."""
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_dag_start("dag")

        await hooks.on_dag_node_start("n1")
        node_span = hooks._dag_node_span_stack[-1]

        await hooks.on_run_start("agent1", "hello")
        agent_span = hooks._run_span_stack[-1]
        assert agent_span.parent_span_id == node_span.span_id

        await hooks.on_run_end("agent1", "done")
        await hooks.on_dag_node_end("n1")
        await hooks.on_dag_end("dag")


class TestParallelTeamWorkers:
    @pytest.mark.asyncio
    async def test_parallel_team_workers_are_siblings(self) -> None:
        """Parallel team workers should share the team span as parent."""
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_team_start("team", "supervisor")
        team_span = hooks._team_span_stack[-1]

        await hooks.on_run_start("worker_a", "task a")
        span_a = hooks._run_span_stack[-1]
        await hooks.on_run_start("worker_b", "task b")
        span_b = hooks._run_span_stack[-1]

        assert span_a.parent_span_id == team_span.span_id
        assert span_b.parent_span_id == team_span.span_id
        assert hooks._span_depth_map[span_a.span_id] == hooks._span_depth_map[span_b.span_id]

        await hooks.on_run_end("worker_b", "done b")
        await hooks.on_run_end("worker_a", "done a")
        await hooks.on_team_end("team", "supervisor")

    @pytest.mark.asyncio
    async def test_pipeline_agents_use_pipeline_as_parent(self) -> None:
        """Agents inside a pipeline should use the pipeline span as parent."""
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_pipeline_start("pipe")
        pipe_span = hooks._pipeline_span_stack[-1]

        await hooks.on_run_start("agent1", "hello")
        agent_span = hooks._run_span_stack[-1]
        assert agent_span.parent_span_id == pipe_span.span_id

        await hooks.on_run_end("agent1", "done")
        await hooks.on_pipeline_end("pipe")


class TestParallelStepSpans:
    @pytest.mark.asyncio
    async def test_parallel_step_spans_tracked_independently(self) -> None:
        """Concurrent step spans in separate tasks should not overwrite each other."""
        import asyncio

        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_team_start("team", "supervisor")

        async def run_worker(name: str) -> None:
            await hooks.on_run_start(name, f"task {name}")
            await hooks.on_step_start(name, 1)
            await asyncio.sleep(0.01)
            await hooks.on_step_end(name, 1)
            await hooks.on_run_end(name, f"done {name}")

        await asyncio.gather(run_worker("a"), run_worker("b"))
        await hooks.on_team_end("team", "supervisor")

        step_spans = [s for s in hooks._all_spans if s.name == "step.1"]
        agent_spans = [s for s in hooks._all_spans if s.kind == "agent"]
        assert len(step_spans) == 2
        assert len(agent_spans) == 2

    @pytest.mark.asyncio
    async def test_parallel_workers_all_run_spans_collected(self) -> None:
        """All parallel worker agent spans should appear in _all_spans."""
        import asyncio

        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_team_start("team", "supervisor")

        async def run_worker(name: str) -> None:
            await hooks.on_run_start(name, f"task {name}")
            await hooks.on_step_start(name, 1)
            await asyncio.sleep(0.01)
            await hooks.on_step_end(name, 1)
            await hooks.on_run_end(name, f"done {name}")

        await asyncio.gather(
            run_worker("a"), run_worker("b"), run_worker("c"),
        )
        await hooks.on_team_end("team", "supervisor")

        agent_spans = [s for s in hooks._all_spans if s.kind == "agent"]
        agent_names = {s.name for s in agent_spans}
        assert len(agent_spans) == 3
        assert agent_names == {"agent.a", "agent.b", "agent.c"}

    @pytest.mark.asyncio
    async def test_parallel_step_parent_is_own_run_span(self) -> None:
        """Each parallel step span should be a child of its own agent run span."""
        import asyncio

        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_team_start("team", "supervisor")

        async def run_worker(name: str) -> None:
            await hooks.on_run_start(name, f"task {name}")
            await hooks.on_step_start(name, 1)
            await asyncio.sleep(0.01)
            await hooks.on_step_end(name, 1)
            await hooks.on_run_end(name, f"done {name}")

        await asyncio.gather(run_worker("a"), run_worker("b"))
        await hooks.on_team_end("team", "supervisor")

        agent_spans = {s.name: s for s in hooks._all_spans if s.kind == "agent"}
        step_spans = [s for s in hooks._all_spans if s.name == "step.1"]

        assert len(step_spans) == 2
        for step_span in step_spans:
            assert step_span.parent_span_id in {s.span_id for s in agent_spans.values()}


def _text_delta(content: str) -> StreamDelta:
    return StreamDelta(chunk=MessageChunk(delta_content=content))


def _tool_call_delta(name: str, args: dict[str, str]) -> StreamDelta:
    return StreamDelta(
        chunk=MessageChunk(
            delta_tool_calls=[ToolCall(name=name, arguments=args)],
            finish_reason="tool_calls",
        ),
        finish_reason=FinishReason.TOOL_CALLS,
    )


def _empty_delta() -> StreamDelta:
    return StreamDelta(chunk=MessageChunk(), usage=Usage(prompt_tokens=10))


class TestStreamingHooks:

    @pytest.mark.asyncio
    async def test_default_hooks_stream_delta_is_noop(self) -> None:
        hooks = DefaultHooks()
        await hooks.on_llm_stream_delta("agent", _text_delta("hello"))

    @pytest.mark.asyncio
    async def test_text_delta_writes_inline_with_prefix(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        written: list[str] = []
        monkeypatch.setattr(hooks._console, "write_inline", lambda text: written.append(text))

        await hooks.on_run_start("agent", "input")
        await hooks.on_step_start("agent", 1)

        await hooks.on_llm_stream_delta("agent", _text_delta("Hello"))
        await hooks.on_llm_stream_delta("agent", _text_delta(" world"))

        # on_step_start writes step marker first, then stream deltas follow
        stream_writes = [w for w in written if "▸" in w or w in ("Hello", " world")]
        assert any("▸" in w for w in written)
        assert "Hello" in written
        assert " world" in written

        await hooks.on_step_end("agent", 1)
        await hooks.on_run_end("agent", "done")

    @pytest.mark.asyncio
    async def test_prefix_only_printed_once_per_step(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        written: list[str] = []
        monkeypatch.setattr(hooks._console, "write_inline", lambda text: written.append(text))

        await hooks.on_run_start("agent", "input")
        await hooks.on_step_start("agent", 1)

        for word in ["A", "B", "C"]:
            await hooks.on_llm_stream_delta("agent", _text_delta(word))

        prefix_count = sum(1 for w in written if "▸" in w)
        assert prefix_count == 1

        await hooks.on_step_end("agent", 1)
        await hooks.on_run_end("agent", "done")

    @pytest.mark.asyncio
    async def test_step_end_flushes_newline_after_streaming(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        written: list[str] = []
        monkeypatch.setattr(hooks._console, "write_inline", lambda text: written.append(text))
        monkeypatch.setattr(hooks._console, "export_one", lambda span, *, indent: None)

        await hooks.on_run_start("agent", "input")
        await hooks.on_step_start("agent", 1)
        await hooks.on_llm_stream_delta("agent", _text_delta("text"))
        await hooks.on_step_end("agent", 1)

        # step_end writes newline (for stream) then completion line
        assert any("\n" == w for w in written)
        assert any("✓" in w for w in written)

        await hooks.on_run_end("agent", "done")

    @pytest.mark.asyncio
    async def test_step_end_no_newline_without_streaming(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        written: list[str] = []
        monkeypatch.setattr(hooks._console, "write_inline", lambda text: written.append(text))
        monkeypatch.setattr(hooks._console, "export_one", lambda span, *, indent: None)

        await hooks.on_run_start("agent", "input")
        await hooks.on_step_start("agent", 1)
        await hooks.on_step_end("agent", 1)

        # on_step_start and on_step_end now write inline (step marker + completion)
        assert any("⟳" in w for w in written)
        assert any("✓" in w for w in written)

        await hooks.on_run_end("agent", "done")

    @pytest.mark.asyncio
    async def test_error_flushes_newline_after_streaming(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        written: list[str] = []
        monkeypatch.setattr(hooks._console, "write_inline", lambda text: written.append(text))
        monkeypatch.setattr(hooks._console, "export_one", lambda span, *, indent: None)

        await hooks.on_run_start("agent", "input")
        await hooks.on_step_start("agent", 1)
        await hooks.on_llm_stream_delta("agent", _text_delta("partial"))
        await hooks.on_error("agent", RuntimeError("disconnect"))

        newlines = [w for w in written if w == "\n"]
        assert len(newlines) == 1

    @pytest.mark.asyncio
    async def test_tool_call_delta_records_span_event(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_run_start("agent", "input")
        await hooks.on_step_start("agent", 1)

        await hooks.on_llm_stream_delta(
            "agent",
            _tool_call_delta("web_search", {"query": "test"}),
        )

        await hooks.on_step_end("agent", 1)

        step_span = hooks._all_spans[0]
        event_names = [e.name for e in step_span.events]
        # stream_tool_call removed — on_tool_call covers tool reporting
        assert "stream_tool_call" not in event_names

        await hooks.on_run_end("agent", "done")

    @pytest.mark.asyncio
    async def test_empty_delta_no_side_effects(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        written: list[str] = []
        monkeypatch.setattr(hooks._console, "write_inline", lambda text: written.append(text))

        await hooks.on_run_start("agent", "input")
        await hooks.on_step_start("agent", 1)

        await hooks.on_llm_stream_delta("agent", _empty_delta())

        # on_step_start writes inline, but empty delta should add nothing extra
        step_writes = [w for w in written if "⟳" in w]
        stream_writes = [w for w in written if "▸" in w]
        assert len(stream_writes) == 0

        await hooks.on_step_end("agent", 1)
        await hooks.on_run_end("agent", "done")

    @pytest.mark.asyncio
    async def test_streaming_indent_depth_in_pipeline(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        written: list[str] = []
        monkeypatch.setattr(hooks._console, "write_inline", lambda text: written.append(text))
        monkeypatch.setattr(
            hooks._console, "print_start",
            lambda *, kind, name, indent=0, attributes=None: None,
        )

        await hooks.on_pipeline_start("pipe")
        await hooks.on_run_start("agent", "input")
        await hooks.on_step_start("agent", 1)

        await hooks.on_llm_stream_delta("agent", _text_delta("hi"))

        # written now includes step marker from on_step_start + stream prefix
        stream_prefix = [w for w in written if "▸" in w]
        assert len(stream_prefix) == 1
        indent_spaces = len(stream_prefix[0]) - len(stream_prefix[0].lstrip())
        assert indent_spaces >= 4

        await hooks.on_step_end("agent", 1)
        await hooks.on_run_end("agent", "done")
        await hooks.on_pipeline_end("pipe")

    @pytest.mark.asyncio
    async def test_concurrent_streaming_independent(self) -> None:
        import asyncio
        from agent_harness.hooks.tracing import _streaming_active

        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        await hooks.on_team_start("team", "supervisor")

        results: dict[str, bool] = {}

        async def worker(name: str) -> None:
            await hooks.on_run_start(name, f"task {name}")
            await hooks.on_step_start(name, 1)
            await hooks.on_llm_stream_delta(name, _text_delta("text"))
            assert _streaming_active.get(False) is True
            await asyncio.sleep(0.01)
            await hooks.on_step_end(name, 1)
            results[name] = not _streaming_active.get(False)
            await hooks.on_run_end(name, f"done {name}")

        await asyncio.gather(worker("a"), worker("b"))
        await hooks.on_team_end("team", "supervisor")

        assert results["a"] is True
        assert results["b"] is True


class TestProgressHooks:
    async def test_tool_call_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        hooks = ProgressHooks(color=False)
        tc = ToolCall(name="web_search", arguments={"query": "5G news"})
        await hooks.on_tool_call("test", tc)
        captured = capsys.readouterr()
        assert "web_search" in captured.out
        assert "5G news" in captured.out
        assert "⚡" in captured.out
        assert "⏺" in captured.out

    async def test_tool_summary_on_step_end(self, capsys: pytest.CaptureFixture[str]) -> None:
        hooks = ProgressHooks(color=False)
        tc = ToolCall(name="web_search", arguments={"query": "5G"})
        result = ToolResult(tool_call_id="tc1", content="ok", is_error=False)
        await hooks.on_tool_call("test", tc)
        await hooks.on_tool_result("test", result)
        await hooks.on_step_end("test", 1)
        captured = capsys.readouterr()
        assert "⎿" in captured.out
        assert "1/1 completed" in captured.out

    async def test_tool_overflow(self, capsys: pytest.CaptureFixture[str]) -> None:
        hooks = ProgressHooks(color=False)
        for i in range(5):
            tc = ToolCall(name=f"tool_{i}", arguments={"x": str(i)})
            await hooks.on_tool_call("test", tc)
            result = ToolResult(tool_call_id=f"tc{i}", content="ok", is_error=False)
            await hooks.on_tool_result("test", result)
        await hooks.on_step_end("test", 1)
        captured = capsys.readouterr()
        assert "tool_0" in captured.out
        assert "tool_2" in captured.out
        assert "tool_3" not in captured.out
        assert "... and 2 more tools" in captured.out
        assert "5/5 completed" in captured.out

    async def test_tool_with_errors(self, capsys: pytest.CaptureFixture[str]) -> None:
        hooks = ProgressHooks(color=False)
        await hooks.on_tool_call("test", ToolCall(name="good", arguments={}))
        await hooks.on_tool_result("test", ToolResult(tool_call_id="1", content="ok", is_error=False))
        await hooks.on_tool_call("test", ToolCall(name="bad", arguments={}))
        await hooks.on_tool_result("test", ToolResult(tool_call_id="2", content="err", is_error=True))
        await hooks.on_step_end("test", 1)
        captured = capsys.readouterr()
        assert "1/2 completed" in captured.out
        assert "1 failed" in captured.out

    async def test_on_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        hooks = ProgressHooks(color=False)
        await hooks.on_error("test", RuntimeError("max steps exceeded"))
        captured = capsys.readouterr()
        assert "❌" in captured.out
        assert "max steps exceeded" in captured.out

    async def test_stream_and_step_end(self, capsys: pytest.CaptureFixture[str]) -> None:
        hooks = ProgressHooks(color=False)
        delta1 = StreamDelta(chunk=MessageChunk(delta_content="Hello "))
        delta2 = StreamDelta(chunk=MessageChunk(delta_content="world"))
        await hooks.on_llm_stream_delta("test", delta1)
        await hooks.on_llm_stream_delta("test", delta2)
        await hooks.on_step_end("test", 1)
        captured = capsys.readouterr()
        assert "Hello world" in captured.out
        assert "⏺" in captured.out
        assert captured.out.endswith("\n")

    async def test_tool_result_silent(self, capsys: pytest.CaptureFixture[str]) -> None:
        hooks = ProgressHooks(color=False)
        result = ToolResult(tool_call_id="tc1", content="data", is_error=False)
        await hooks.on_tool_result("test", result)
        captured = capsys.readouterr()
        assert captured.out == ""


class TestCompositeHooks:
    async def test_calls_all_hooks(self) -> None:
        calls: list[str] = []

        class TrackingHooks(DefaultHooks):
            def __init__(self, name: str) -> None:
                self.name = name

            async def on_run_start(self, agent_name: str, input_text: str) -> None:
                calls.append(self.name)

        composite = CompositeHooks(TrackingHooks("a"), TrackingHooks("b"))
        await composite.on_run_start("test", "hello")
        assert calls == ["a", "b"]


class TestLLMRetryHook:
    async def test_default_hooks_on_llm_retry_is_no_op(self) -> None:
        hooks = DefaultHooks()
        info = LLMRetryInfo(
            kind="stream", attempt=1, max_retries=3, wait=1.0,
            error=LLMConnectionError("boom"),
        )
        await hooks.on_llm_retry("agent", info)

    async def test_composite_fan_out_on_llm_retry_in_order(self) -> None:
        log: list[tuple[str, int]] = []

        class Recorder(DefaultHooks):
            def __init__(self, label: str) -> None:
                self._label = label

            async def on_llm_retry(self, agent_name: str, info: LLMRetryInfo) -> None:
                log.append((self._label, info.attempt))

        composite = CompositeHooks(Recorder("a"), Recorder("b"))
        info = LLMRetryInfo(
            kind="generate", attempt=2, max_retries=3, wait=2.0,
            error=TimeoutError(),
        )
        await composite.on_llm_retry("agent", info)
        assert log == [("a", 2), ("b", 2)]

    async def test_tracing_hooks_on_llm_retry_is_no_op(self) -> None:
        hooks = TracingHooks(trace_dir="/tmp/test_traces")
        info = LLMRetryInfo(
            kind="stream", attempt=1, max_retries=3, wait=1.0,
            error=LLMConnectionError("boom"),
        )
        await hooks.on_llm_retry("agent", info)

    async def test_progress_hooks_on_llm_retry_is_no_op(self) -> None:
        hooks = ProgressHooks()
        info = LLMRetryInfo(
            kind="generate", attempt=1, max_retries=3, wait=1.0,
            error=TimeoutError(),
        )
        await hooks.on_llm_retry("agent", info)


class TestResolveHooksProgress:
    def test_returns_progress_when_tracing_off(self) -> None:
        config = HarnessConfig(tracing=TracingConfig(enabled=False))
        hooks = resolve_hooks(None, config)
        assert isinstance(hooks, ProgressHooks)

    def test_returns_tracing_when_tracing_on(self) -> None:
        config = HarnessConfig(tracing=TracingConfig(enabled=True))
        hooks = resolve_hooks(None, config)
        assert isinstance(hooks, TracingHooks)


class TestTracingExporterConfig:
    def test_exporter_console_only(self) -> None:
        hooks = TracingHooks(exporter="console")
        assert hooks._console is not None
        assert hooks._console_enabled is True
        assert hooks._json_enabled is False

    def test_exporter_json_only(self) -> None:
        hooks = TracingHooks(exporter="json_file")
        assert hooks._console is None
        assert hooks._console_enabled is False
        assert hooks._json_enabled is True

    def test_exporter_both(self) -> None:
        hooks = TracingHooks(exporter="both")
        assert hooks._console is not None
        assert hooks._console_enabled is True
        assert hooks._json_enabled is True

    def test_default_exporter_is_both(self) -> None:
        hooks = TracingHooks()
        assert hooks._console_enabled is True
        assert hooks._json_enabled is True

    async def test_console_only_no_json_export(self, tmp_path: Path) -> None:
        hooks = TracingHooks(trace_dir=str(tmp_path), exporter="console")
        await hooks.on_run_start("agent", "hello")
        await hooks.on_run_end("agent", "done")
        json_files = list(tmp_path.glob("*.jsonl"))
        assert len(json_files) == 0

    async def test_json_only_no_console_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        hooks = TracingHooks(trace_dir=str(tmp_path), exporter="json_file")
        await hooks.on_run_start("agent", "hello")
        await hooks.on_step_start("agent", 1)
        await hooks.on_llm_call("agent", [])
        await hooks.on_step_end("agent", 1)
        await hooks.on_run_end("agent", "done")
        captured = capsys.readouterr()
        assert captured.err == ""
        json_files = list(tmp_path.glob("*.jsonl"))
        assert len(json_files) == 1

    async def test_both_produces_console_and_json(self, tmp_path: Path) -> None:
        hooks = TracingHooks(trace_dir=str(tmp_path), exporter="both")
        await hooks.on_run_start("agent", "hello")
        await hooks.on_step_start("agent", 1)
        await hooks.on_step_end("agent", 1)
        await hooks.on_run_end("agent", "done")
        json_files = list(tmp_path.glob("*.jsonl"))
        assert len(json_files) == 1

    def test_resolve_hooks_passes_exporter_config(self) -> None:
        config = HarnessConfig(
            tracing=TracingConfig(enabled=True, exporter="json_file"),
        )
        hooks = resolve_hooks(None, config)
        assert isinstance(hooks, TracingHooks)
        assert hooks._json_enabled is True
        assert hooks._console_enabled is False
