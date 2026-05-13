"""Tests for agent_harness.memory.short_term — ShortTermMemory buffer and trimming."""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from agent_harness.core.message import Message, Role
from agent_harness.memory.compressor import ContextCompressor
from agent_harness.memory.short_term import (
    CallSnapshot, SectionWeights, ShortTermMemory,
)


class TestShortTermMemoryBasic:
    @pytest.mark.asyncio
    async def test_add_message(self) -> None:
        mem = ShortTermMemory()
        msg = Message.user("hello")
        await mem.add_message(msg)
        assert await mem.size() == 1

    @pytest.mark.asyncio
    async def test_add_text(self) -> None:
        mem = ShortTermMemory()
        await mem.add("some text")
        assert await mem.size() == 1
        msgs = await mem.get_context_messages()
        assert msgs[0].role == Role.USER
        assert msgs[0].content == "some text"

    @pytest.mark.asyncio
    async def test_get_context_messages(self) -> None:
        mem = ShortTermMemory()
        await mem.add_message(Message.system("You are helpful."))
        await mem.add_message(Message.user("Hi"))
        await mem.add_message(Message.assistant("Hello!"))
        msgs = await mem.get_context_messages()
        assert len(msgs) == 3
        assert msgs[0].role == Role.SYSTEM
        assert msgs[1].role == Role.USER
        assert msgs[2].role == Role.ASSISTANT

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        mem = ShortTermMemory()
        await mem.add_message(Message.user("a"))
        await mem.add_message(Message.user("b"))
        assert await mem.size() == 2
        await mem.clear()
        assert await mem.size() == 0
        assert await mem.get_context_messages() == []

    @pytest.mark.asyncio
    async def test_query_returns_relevant(self) -> None:
        mem = ShortTermMemory()
        for i in range(10):
            await mem.add_message(Message.user(f"msg-{i}"))
        items = await mem.query("msg", top_k=3)
        assert len(items) == 3
        assert all("msg-" in item.content for item in items)

    @pytest.mark.asyncio
    async def test_add_message_does_not_trim(self) -> None:
        """add_message only appends — no trimming until get_context_messages."""
        mem = ShortTermMemory(max_tokens=50)
        for i in range(20):
            await mem.add_message(Message.user(f"message-{i} " * 10))
        assert await mem.size() == 20


class TestTokenCount:
    async def test_token_count_empty(self) -> None:
        mem = ShortTermMemory(max_tokens=100000, model="gpt-4o")
        baseline = mem.token_count
        await mem.add_message(Message.user("Hello world"))
        assert mem.token_count > baseline

    async def test_token_count_increases(self) -> None:
        mem = ShortTermMemory(max_tokens=100000, model="gpt-4o")
        await mem.add_message(Message.user("Hello world"))
        count_after_one = mem.token_count
        assert count_after_one > 0

        await mem.add_message(Message.assistant("Hi there"))
        assert mem.token_count > count_after_one

    async def test_token_count_resets_on_clear(self) -> None:
        mem = ShortTermMemory(max_tokens=100000, model="gpt-4o")
        baseline = mem.token_count
        await mem.add_message(Message.user("Hello world"))
        assert mem.token_count > baseline
        await mem.clear()
        assert mem.token_count == baseline


class TestTokenTrim:
    @pytest.mark.asyncio
    async def test_get_context_is_pure_read(self) -> None:
        """get_context_messages is pure read; trim happens in build_llm_messages."""
        mem = ShortTermMemory(max_tokens=50)
        for i in range(20):
            await mem.add_message(Message.user(f"message-{i} " * 10))
        msgs = await mem.get_context_messages()
        assert len(msgs) == 20

    def test_trim_by_tokens_drops_excess(self) -> None:
        msgs = [Message.user(f"message-{i} " * 10) for i in range(20)]
        trimmed = ShortTermMemory._trim_by_tokens(msgs, 50, "gpt-4o")
        assert len(trimmed) < 20

    def test_trim_preserves_system_message(self) -> None:
        msgs = [Message.system("System prompt")]
        msgs.extend(Message.user(f"msg-{i} " * 10) for i in range(20))
        trimmed = ShortTermMemory._trim_by_tokens(msgs, 100, "gpt-4o")
        assert trimmed[0].role == Role.SYSTEM
        assert trimmed[0].content == "System prompt"

    @pytest.mark.asyncio
    async def test_no_trim_when_under_budget(self) -> None:
        mem = ShortTermMemory(max_tokens=100000)
        await mem.add_message(Message.user("one"))
        await mem.add_message(Message.user("two"))
        msgs = await mem.get_context_messages()
        assert len(msgs) == 2

    def test_trim_preserves_multiple_system_messages(self) -> None:
        msgs = [Message.system("sys1"), Message.system("sys2")]
        msgs.extend(Message.user(f"u{i} " * 10) for i in range(20))
        trimmed = ShortTermMemory._trim_by_tokens(msgs, 200, "gpt-4o")
        sys_msgs = [m for m in trimmed if m.role == Role.SYSTEM]
        assert len(sys_msgs) == 2

    def test_trim_keeps_most_recent(self) -> None:
        msgs = [Message.user(f"msg-{i} " * 10) for i in range(20)]
        trimmed = ShortTermMemory._trim_by_tokens(msgs, 100, "gpt-4o")
        contents = [m.content for m in trimmed]
        assert "msg-19 " * 10 in contents[-1]


class TestCompressorAttribute:
    @pytest.mark.asyncio
    async def test_compressor_default_none(self) -> None:
        mem = ShortTermMemory()
        assert mem.compressor is None

    @pytest.mark.asyncio
    async def test_compressor_is_public(self) -> None:
        from unittest.mock import AsyncMock

        from agent_harness.memory.compressor import ContextCompressor

        compressor = ContextCompressor(
            llm=AsyncMock(), threshold=0.75, retain_count=4, model="gpt-4o",
        )
        mem = ShortTermMemory(max_tokens=500, compressor=compressor)
        assert mem.compressor is compressor

    @pytest.mark.asyncio
    async def test_compression_orchestration_moved_to_context(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from agent_harness.context.context import AgentContext
        from agent_harness.hooks.base import DefaultHooks

        compressor = ContextCompressor(
            llm=AsyncMock(),
            threshold=0.0,
            retain_count=4,
            model="gpt-4o",
        )
        ctx = AgentContext()
        ctx.short_term_memory = ShortTermMemory(max_tokens=50, compressor=compressor)
        for i in range(20):
            await ctx.short_term_memory.add_message(
                Message.user(f"message-{i} " * 10)
            )

        target = logging.getLogger("agent_harness.context.context")
        target.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.DEBUG, logger="agent_harness.context.context"):
                with patch.object(
                    compressor,
                    "compress",
                    AsyncMock(side_effect=RuntimeError("boom")),
                ):
                    await ctx.maybe_auto_compress(
                        DefaultHooks(), "test", authoritative_input=10_000,
                    )
        finally:
            target.removeHandler(caplog.handler)

        assert "Compression failed" in caplog.text

    @pytest.mark.asyncio
    async def test_auto_compress_no_op_preserves_last_call(self) -> None:
        """When compressor returns the unchanged list (`take_last_result()` is
        None), `maybe_auto_compress` must not call `replace_messages` — doing
        so would wipe `last_call`, leaving the status bar with `—/Xm` even
        though the messages didn't actually change."""
        from agent_harness.context.context import AgentContext
        from agent_harness.hooks.base import DefaultHooks

        compressor = ContextCompressor(
            llm=AsyncMock(),
            threshold=0.0,
            retain_count=4,
            model="gpt-4o",
        )
        ctx = AgentContext()
        ctx.short_term_memory = ShortTermMemory(
            max_tokens=50, compressor=compressor, model="gpt-4o",
        )
        for i in range(5):
            await ctx.short_term_memory.add_message(
                Message.user(f"message-{i}")
            )

        snapshot = CallSnapshot(
            input_tokens=300, completion_tokens=200, total_tokens=500,
            cache_read=0, cache_creation=0, reasoning_tokens=0,
            model="gpt-4o", message_count=5,
            section_weights=SectionWeights(
                system_prompt=10, tools_schema=20, dynamic_system=5, history=15,
            ),
        )
        ctx.short_term_memory.record_call(snapshot)
        original_messages = list(ctx.short_term_memory._messages)

        with patch.object(
            compressor,
            "compress",
            AsyncMock(return_value=original_messages),
        ):
            await ctx.maybe_auto_compress(
                DefaultHooks(), "test", authoritative_input=10_000,
            )

        # Snapshot survives — `displayed_input_tokens` still has provider truth
        assert ctx.short_term_memory.last_call is snapshot
        assert ctx.short_term_memory.displayed_input_tokens == 500
        # Messages untouched
        assert ctx.short_term_memory._messages == original_messages


class TestDisplayedInputTokens:
    """displayed_input_tokens combines snapshot truth with delta of new messages."""

    @staticmethod
    def _snap(total: int, msg_count: int, *, reasoning: int = 0) -> CallSnapshot:
        return CallSnapshot(
            input_tokens=total - 200,
            completion_tokens=200,
            total_tokens=total,
            cache_read=0,
            cache_creation=0,
            reasoning_tokens=reasoning,
            model="gpt-4o",
            message_count=msg_count,
            section_weights=SectionWeights(
                system_prompt=10, tools_schema=20, dynamic_system=5, history=15,
            ),
        )

    @pytest.mark.asyncio
    async def test_returns_none_when_no_call(self) -> None:
        mem = ShortTermMemory()
        assert mem.displayed_input_tokens is None

    @pytest.mark.asyncio
    async def test_returns_total_when_no_delta(self) -> None:
        mem = ShortTermMemory()
        await mem.add_message(Message.user("hi"))
        await mem.add_message(Message.assistant("hello"))
        mem.last_call = self._snap(total=500, msg_count=2)
        assert mem.displayed_input_tokens == 500

    @pytest.mark.asyncio
    async def test_adds_delta_for_messages_after_snapshot(self) -> None:
        """User pastes large content after last call — displayed must reflect it,
        otherwise compressor.should_compress under-triggers and trim drops history.
        """
        mem = ShortTermMemory()
        await mem.add_message(Message.user("hi"))
        await mem.add_message(Message.assistant("hello"))
        mem.last_call = self._snap(total=500, msg_count=2)
        await mem.add_message(Message.user("a " * 5000))
        displayed = mem.displayed_input_tokens
        assert displayed is not None
        assert displayed > 500

    @pytest.mark.asyncio
    async def test_subtracts_reasoning_tokens(self) -> None:
        """OpenAI reasoning_tokens are billed but not persisted in buffer."""
        mem = ShortTermMemory()
        await mem.add_message(Message.user("q"))
        await mem.add_message(Message.assistant("a"))
        mem.last_call = self._snap(total=500, msg_count=2, reasoning=300)
        assert mem.displayed_input_tokens == 200


class TestForgetImportanceScore:
    """Tests for forget() using importance_score from message metadata."""

    @pytest.mark.asyncio
    async def test_high_importance_retained(self) -> None:
        mem = ShortTermMemory()
        msg = Message.user("important", metadata={"importance_score": 0.9})
        await mem.add_message(msg)
        forgotten = await mem.forget(threshold=0.3)
        assert forgotten == 0
        assert await mem.size() == 1

    @pytest.mark.asyncio
    async def test_low_importance_forgotten(self) -> None:
        mem = ShortTermMemory()
        msg = Message.user("trivial", metadata={"importance_score": 0.1})
        await mem.add_message(msg)
        forgotten = await mem.forget(threshold=1.1)
        assert forgotten == 1
        assert await mem.size() == 0

    @pytest.mark.asyncio
    async def test_default_importance_when_missing(self) -> None:
        mem = ShortTermMemory()
        msg_default = Message.user("default importance", metadata={})
        msg_explicit = Message.user("explicit 0.5", metadata={"importance_score": 0.5})
        await mem.add_message(msg_default)
        await mem.add_message(msg_explicit)
        forgotten = await mem.forget(threshold=0.3)
        assert forgotten == 0
        assert await mem.size() == 2

    @pytest.mark.asyncio
    async def test_importance_affects_weighted_score(self) -> None:
        mem = ShortTermMemory()
        high = Message.user("high", metadata={"importance_score": 0.9})
        low = Message.user("low", metadata={"importance_score": 0.1})
        await mem.add_message(high)
        await mem.add_message(low)
        await mem.forget(threshold=0.9)
        msgs = await mem.get_context_messages()
        assert len(msgs) == 1
        assert msgs[0].content == "high"


class TestDisplayedInputTokensWithSidecar:

    @staticmethod
    def _stm_with_call(
        *,
        total_tokens: int,
        reasoning_tokens: int,
        last_msg: Message,
    ) -> ShortTermMemory:
        stm = ShortTermMemory(model="gpt-4o")
        stm._messages.append(last_msg)
        stm.last_call = CallSnapshot(
            input_tokens=0,
            completion_tokens=0,
            total_tokens=total_tokens,
            cache_read=0,
            cache_creation=0,
            reasoning_tokens=reasoning_tokens,
            model="gpt-4o",
            message_count=1,
            section_weights=SectionWeights(
                system_prompt=0, tools_schema=0, dynamic_system=0, history=0
            ),
        )
        return stm

    def test_subtracts_reasoning_when_sidecar_empty(self) -> None:
        stm = self._stm_with_call(
            total_tokens=180,
            reasoning_tokens=50,
            last_msg=Message(role=Role.ASSISTANT, content="answer"),
        )
        assert stm.displayed_input_tokens == 130

    def test_keeps_reasoning_when_sidecar_has_content(self) -> None:
        stm = self._stm_with_call(
            total_tokens=180,
            reasoning_tokens=50,
            last_msg=Message(
                role=Role.ASSISTANT,
                content="answer",
                provider_metadata={"openai_chat": {"reasoning_content": "the reasoning"}},
            ),
        )
        assert stm.displayed_input_tokens == 180

    def test_subtracts_when_list_holds_empty_dicts(self) -> None:
        stm = self._stm_with_call(
            total_tokens=180,
            reasoning_tokens=50,
            last_msg=Message(
                role=Role.ASSISTANT,
                content="answer",
                provider_metadata={"openai_chat": {"reasoning_details": [{}]}},
            ),
        )
        assert stm.displayed_input_tokens == 130

    def test_keeps_when_list_holds_real_content(self) -> None:
        stm = self._stm_with_call(
            total_tokens=180,
            reasoning_tokens=50,
            last_msg=Message(
                role=Role.ASSISTANT,
                content="answer",
                provider_metadata={"openai_chat": {"reasoning_details": [{"text": "step 1"}]}},
            ),
        )
        assert stm.displayed_input_tokens == 180

    def test_no_reasoning_tokens_means_no_subtraction(self) -> None:
        stm = self._stm_with_call(
            total_tokens=180,
            reasoning_tokens=0,
            last_msg=Message(role=Role.ASSISTANT, content="answer"),
        )
        assert stm.displayed_input_tokens == 180
