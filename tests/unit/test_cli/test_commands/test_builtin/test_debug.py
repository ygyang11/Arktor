import logging
from unittest.mock import MagicMock

import pytest

from agent_cli import hooks
from agent_cli.commands.builtin.debug import CMD
from agent_harness.utils.logging_config import setup_logging

from ..conftest import render_output


@pytest.fixture(autouse=True)
def reset_debug() -> None:
    hooks._debug_enabled[0] = False
    setup_logging("WARNING")
    yield
    hooks._debug_enabled[0] = False
    setup_logging("WARNING")


async def test_first_call_turns_on() -> None:
    result = await CMD.handler(MagicMock(), "")
    assert hooks.is_debug_enabled() is True
    assert logging.getLogger("agent_harness").level == logging.DEBUG
    assert logging.getLogger("agent_app").level == logging.DEBUG
    assert "Debug mode on" in render_output(result.output)


async def test_second_call_turns_off() -> None:
    await CMD.handler(MagicMock(), "")
    result = await CMD.handler(MagicMock(), "")
    assert hooks.is_debug_enabled() is False
    assert logging.getLogger("agent_harness").level == logging.WARNING
    assert logging.getLogger("agent_app").level == logging.WARNING
    assert "Debug mode off" in render_output(result.output)


async def test_args_ignored() -> None:
    result = await CMD.handler(MagicMock(), "whatever garbage")
    assert hooks.is_debug_enabled() is True
    assert "Debug mode on" in render_output(result.output)


def test_command_metadata() -> None:
    assert CMD.name == "/debug"
    assert "traceback" in CMD.description.lower()
