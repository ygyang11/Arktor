from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_cli.runtime.goal import evaluator
from agent_harness.core.message import Attachment, Message, ToolCall
from agent_harness.llm.types import LLMResponse, ProcessUsageMeter, Usage
from agent_harness.memory.short_term import CallSnapshot, SectionWeights, ShortTermMemory
from agent_harness.utils.token_counter import count_tokens


def _response(content: str, tokens: int = 10) -> LLMResponse:
    return LLMResponse(
        message=Message.assistant(content),
        usage=Usage(prompt_tokens=tokens - 1, completion_tokens=1, total_tokens=tokens),
    )


def _agent(
    messages: list[Message] | None = None,
    responses: list[object] | None = None,
    *,
    max_tokens: int = 10_000,
) -> MagicMock:
    stm = ShortTermMemory(max_tokens=max_tokens, model="test-model")
    stm._messages = list(messages or [])
    llm = MagicMock()
    llm.model_name = "test-model"
    llm.generate_with_events = AsyncMock(side_effect=responses or [])
    agent = MagicMock()
    agent.llm = llm
    agent.context = SimpleNamespace(
        short_term_memory=stm,
        usage_meter=ProcessUsageMeter(),
    )
    return agent


@pytest.mark.parametrize("status", ["continue", "complete", "blocked"])
def test_parse_valid_verdicts(status: str) -> None:
    directive = "next <item>\nand verify" if status == "continue" else ""
    verdict, error = evaluator._parse(
        f"<status>{status.upper()}</status>\n"
        "<reason>coverage is >90%</reason>\n"
        f"<directive>{directive}</directive>"
    )
    assert error is None
    assert verdict is not None
    assert verdict.status == status
    assert verdict.reason == "coverage is >90%"
    assert verdict.directive == directive


@pytest.mark.parametrize("content", [
    "```\n<status>complete</status>\n<reason>x</reason>\n<directive></directive>\n```",
    "extra\n<status>complete</status>\n<reason>x</reason>\n<directive></directive>",
    "<reason>x</reason>\n<status>complete</status>\n<directive></directive>",
    "<status>complete</status>\n<reason>a\nb</reason>\n<directive></directive>",
])
def test_parse_rejects_bad_shape(content: str) -> None:
    assert evaluator._parse(content)[0] is None


def test_parse_field_invariants() -> None:
    assert evaluator._parse(
        "<status>continue</status>\n<reason>x</reason>\n<directive></directive>"
    )[1] == "directive must be non-empty for status continue"
    assert evaluator._parse(
        "<status>complete</status>\n<reason></reason>\n<directive></directive>"
    )[1] == "reason must be non-empty"
    assert evaluator._parse(
        "<status>blocked</status>\n<reason>x</reason>\n<directive>work</directive>"
    )[1] == "directive must be empty for status complete or blocked"


def test_payload_uses_matching_128_bit_nonce() -> None:
    nonce = "a" * 32
    payload = evaluator._payload(
        "objective", "evidence", nonce=nonce, turns=2, elapsed_s=3, tokens=4
    )
    assert payload.count(f"objective-{nonce}") == 2
    assert payload.count(f"transcript-{nonce}") == 2
    assert "turns_completed: 2" in payload
    assert "tokens_used: 4" in payload


async def test_evaluate_corrects_twice_and_records_all_usage() -> None:
    agent = _agent(responses=[
        _response("bad one", 10),
        _response("bad two", 20),
        _response(
            "<status>continue</status>\n"
            "<reason>work remains</reason>\n"
            "<directive>finish it</directive>",
            30,
        ),
    ])
    verdict = await evaluator.evaluate(
        agent, "objective", turns=1, elapsed_s=2, tokens=3
    )
    assert verdict.status == "continue"
    assert agent.llm.generate_with_events.await_count == 3
    third_messages = agent.llm.generate_with_events.await_args_list[2].args[0]
    assert third_messages[2].role.value == "assistant"
    assert third_messages[2].content == "bad one"
    assert third_messages[4].content == "bad two"
    bucket = agent.context.usage_meter.by_source["goal_eval"]
    assert bucket.calls == 3
    assert bucket.usage.total_tokens == 60


async def test_evaluate_rejects_three_invalid_responses() -> None:
    agent = _agent(responses=[_response("bad")] * 3)
    with pytest.raises(evaluator.GoalEvaluationError, match="3 attempts"):
        await evaluator.evaluate(
            agent, "objective", turns=1, elapsed_s=2, tokens=3
        )


async def test_evaluate_wraps_provider_error_but_propagates_cancel() -> None:
    failed = _agent(responses=[RuntimeError("down")])
    with pytest.raises(evaluator.GoalEvaluationError, match="RuntimeError"):
        await evaluator.evaluate(
            failed, "objective", turns=1, elapsed_s=2, tokens=3
        )

    cancelled = _agent(responses=[asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await evaluator.evaluate(
            cancelled, "objective", turns=1, elapsed_s=2, tokens=3
        )


def test_format_transcript_pairs_tools_and_filters_internal_messages() -> None:
    attachment = Attachment(digest="a" * 64, mime="text/plain", filename="a.txt", size=1)
    call = ToolCall(id="call-1", name="terminal_tool", arguments={"command": "pytest -q"})
    messages = [
        Message.system("secret system"),
        Message.system("summary", metadata={"is_compression_summary": True}),
        Message.user("continue body", metadata={"is_goal_continuation": True}),
        Message.user("task", attachments=[attachment]),
        Message.assistant("running", tool_calls=[call]),
        Message.tool("call-1", "ok", attachments=[attachment]),
        Message.assistant(
            tool_calls=[ToolCall(id="missing", name="read", arguments={"path": "x"})]
        ),
    ]
    results = {
        m.tool_result.tool_call_id: m.tool_result
        for m in messages
        if m.tool_result is not None
    }
    text = evaluator._format_transcript(messages, results)
    assert "secret system" not in text
    assert "continue body" not in text
    assert "summary" in text
    assert '[tool terminal_tool]: {"command":"pytest -q"}' in text
    assert "└─ success:\nok" in text
    assert "call-1" not in text
    assert "└─ result missing" in text
    assert text.count("[Attached text/plain: a.txt]") == 2


def test_render_transcript_only_trims_tool_results() -> None:
    body = "prefix " + ("middle " * 5000) + " suffix"
    call = ToolCall(id="c", name="terminal_tool", arguments={})
    messages = [
        Message.user("keep-user"),
        Message.assistant("keep-assistant", tool_calls=[call]),
        Message.tool("c", body),
    ]
    original = messages[-1].tool_result.content
    rendered = evaluator._render_transcript(
        messages,
        budget=1400,
        model="test-model",
    )
    assert "keep-user" in rendered
    assert "keep-assistant" in rendered
    assert evaluator._TOOL_RESULT_OMISSION in rendered
    assert "prefix" in rendered and "suffix" in rendered
    assert messages[-1].tool_result.content == original
    assert count_tokens(rendered, model="test-model") <= 1400


def test_deep_trim_returns_candidate_below_floor(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluator,
        "count_tokens",
        lambda text, model: 1025 if len(text) > 70 else 1023,
    )
    trimmed = evaluator._trim_tool_result(
        "x" * 100,
        target_tokens=1024,
        model="test-model",
    )
    assert trimmed != "x" * 100


async def test_non_transcript_budget_error_suggests_compact() -> None:
    agent = _agent(max_tokens=10)
    with pytest.raises(evaluator.GoalEvaluationError, match="/compact"):
        await evaluator.evaluate(
            agent, "objective", turns=1, elapsed_s=2, tokens=3
        )


async def test_same_model_snapshot_only_corrects_local_undercount(monkeypatch) -> None:
    valid = _response(
        "<status>complete</status>\n<reason>done</reason>\n<directive></directive>"
    )
    agent = _agent(responses=[valid], max_tokens=10_000)
    agent.context.short_term_memory.last_call = CallSnapshot(
        input_tokens=200,
        completion_tokens=0,
        total_tokens=200,
        cache_read=0,
        cache_creation=0,
        model="test-model",
        section_weights=SectionWeights(
            system_prompt=10,
            tools_schema=10,
            dynamic_system=10,
            history=10,
        ),
    )
    seen: list[int] = []
    real_render = evaluator._render_transcript

    def capture(messages, *, budget, model):
        seen.append(budget)
        return real_render(messages, budget=budget, model=model)

    monkeypatch.setattr(evaluator, "_render_transcript", capture)
    await evaluator.evaluate(agent, "objective", turns=1, elapsed_s=2, tokens=3)
    assert seen
    assert seen[0] < int(10_000 * (1 - evaluator._CONTEXT_HEADROOM))


def test_untrimmable_transcript_raises_compact_error() -> None:
    with pytest.raises(evaluator.GoalEvaluationError, match=re.escape("/compact")):
        evaluator._render_transcript(
            [Message.user("x" * 10_000)],
            budget=10,
            model="test-model",
        )
