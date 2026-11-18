"""Tests for model call request metadata."""
from __future__ import annotations

import json

from agent_harness.llm.call_context import AgentIdentity, ModelCallContext, with_request_context


def test_header_copy_case_insensitive_and_unicode() -> None:
    context = ModelCallContext(AgentIdentity("主.main", None, "main"), 2, "agent", "main")
    original = {"extra_headers": {"x-arktor-context": "old", "Other": "kept"}, "temperature": 0}

    disabled = with_request_context(original, context, enabled=False)
    assert disabled == original and disabled is not original

    request = with_request_context(original, context, enabled=True)
    assert original["extra_headers"]["x-arktor-context"] == "old"
    assert request["extra_headers"]["Other"] == "kept"
    assert "x-arktor-context" not in request["extra_headers"]
    header = request["extra_headers"]["X-Arktor-Context"]
    assert header.isascii() and json.loads(header) == context.to_dict()
