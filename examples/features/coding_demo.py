"""Coding agent demo — a ReActAgent with filesystem + terminal + web tools.

Demonstrates: filesystem tools (read/write/edit/list/glob/grep),
terminal tool (git, pytest, shell commands), web_search, web_fetch,
tool selection, multi-step coding workflow, and SystemPromptBuilder.

Usage:
    python examples/features/coding_demo.py          # auto mode (stream)
    python examples/features/coding_demo.py --chat    # interactive mode
"""

import argparse
import asyncio
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from agent_app.tools import (
    BACKGROUND_TOOLS, FILESYSTEM_TOOLS, TERMINAL_TOOLS, WEB_TOOLS,
    TODO_TOOLS, SUB_AGENT_TOOLS,
)
from agent_harness import HarnessConfig, ReActAgent

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ALL_TOOLS = [
    *FILESYSTEM_TOOLS, *TERMINAL_TOOLS, *WEB_TOOLS,
    *TODO_TOOLS, *SUB_AGENT_TOOLS, *BACKGROUND_TOOLS,
]

SYSTEM_PROMPT = (
    "You are a skilled Python developer. "
    "You have access to filesystem tools, terminal tool, web_search, and web_fetch. "
    "Always explore the project structure first before making changes. "
    "Read files before editing them."
)

TASK = """\
You are working on a Python calculator project in the current workspace.

There are two known issues:
- Division by zero is not handled in calculator.py — it will crash. \
Research the best practice for handling this if you're unsure.
- `format_result` in formatter.py is not implemented — it should \
format calculation results nicely (e.g. '2 + 3 = 5').

Fix both issues and make sure everything works..
"""


def setup_mini_project(workspace: Path) -> None:
    """Create a mini Python project with intentional bugs."""
    src = workspace / "src"
    src.mkdir()

    (src / "__init__.py").write_text("")

    (src / "calculator.py").write_text(
        '"""Simple calculator module."""\n\n\n'
        "def add(a: float, b: float) -> float:\n"
        "    return a + b\n\n\n"
        "def subtract(a: float, b: float) -> float:\n"
        "    return a - b\n\n\n"
        "def multiply(a: float, b: float) -> float:\n"
        "    return a * b\n\n\n"
        "def divide(a: float, b: float) -> float:\n"
        "    # BUG: no zero-division check\n"
        "    return a / b\n"
    )

    (src / "formatter.py").write_text(
        '"""Result formatting utilities."""\n\n\n'
        "def format_result(operation: str, a: float, b: float, result: float) -> str:\n"
        "    # TODO: implement this function\n"
        "    # Should return something like: '2 + 3 = 5.0'\n"
        "    pass\n"
    )

    tests = workspace / "tests"
    tests.mkdir()

    (tests / "__init__.py").write_text("")

    (tests / "test_calculator.py").write_text(
        '"""Tests for calculator."""\n'
        "import sys\n"
        "from pathlib import Path\n\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))\n\n"
        "from src.calculator import add, subtract, multiply, divide\n\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n\n\n"
        "def test_subtract():\n"
        "    assert subtract(10, 4) == 6\n\n\n"
        "def test_multiply():\n"
        "    assert multiply(3, 4) == 12\n\n\n"
        "def test_divide():\n"
        "    assert divide(10, 2) == 5.0\n\n\n"
        "def test_divide_by_zero():\n"
        "    try:\n"
        "        divide(1, 0)\n"
        "        assert False, 'Should have raised ValueError'\n"
        "    except ValueError:\n"
        "        pass\n"
    )

    (workspace / "README.md").write_text(
        "# Mini Calculator\n\n"
        "A simple calculator project for testing.\n\n"
        "## Known Issues\n"
        "- Division by zero is not handled\n"
        "- `format_result` in formatter.py is not implemented\n"
    )


async def run_auto(config: HarnessConfig, workspace: Path) -> None:
    """Auto mode: agent completes the task autonomously."""
    print("=== Coding Agent Demo (stream) ===\n")

    agent = ReActAgent(
        name="coding-agent",
        tools=list(ALL_TOOLS),
        system_prompt=SYSTEM_PROMPT,
        max_steps=20,
        config=config,
    )

    print(f"Task:\n{TASK}\n")
    result = await agent.run(TASK)

    print(f"\n{'=' * 50}")
    print(f"Agent response:\n{result.output}\n")
    print(f"Steps: {result.step_count}")
    print(f"Tokens: {result.usage.total_tokens}")

    # Tool usage summary
    tool_usage: dict[str, int] = {}
    for step in result.steps:
        if step.action:
            for tc in step.action:
                tool_usage[tc.name] = tool_usage.get(tc.name, 0) + 1
    print(f"Tool usage: {tool_usage}")

    # Verify changes
    print(f"\n{'=' * 50}")
    print("Verification:\n")

    calc = workspace / "src" / "calculator.py"
    if calc.exists():
        content = calc.read_text()
        bug_comment_removed = "# BUG" not in content
        has_zero_guard = (
            "b == 0" in content
            or "b != 0" in content
            or "ZeroDivisionError" in content
            or "ValueError" in content
        )
        if bug_comment_removed and has_zero_guard:
            print("  ✓ Bug fix verified: division by zero is now handled")
        else:
            print("  ✗ Bug fix NOT detected in calculator.py")
    else:
        print("  ✗ calculator.py not found")

    fmt = workspace / "src" / "formatter.py"
    if fmt.exists():
        content = fmt.read_text()
        if content.strip().endswith("pass"):
            print("  ✗ TODO NOT implemented in formatter.py")
        else:
            print("  ✓ TODO implemented: formatter.py has been updated")
    else:
        print("  ✗ formatter.py not found")

    if "terminal_tool" in tool_usage:
        print(f"  ✓ terminal_tool used {tool_usage['terminal_tool']} time(s)")
    else:
        print("  ✗ terminal_tool was NOT used")


async def run_chat(config: HarnessConfig, workspace: Path) -> None:
    """Chat mode: interact with the coding agent."""
    print("=== Coding Agent Demo (interactive) ===")
    print(f"Workspace: {workspace}")
    print("Tools: filesystem + terminal + web + todo + sub_agent + background")
    print("Type 'exit' to quit.\n")

    agent = ReActAgent(
        name="coding-agent",
        tools=list(ALL_TOOLS),
        system_prompt=SYSTEM_PROMPT,
        max_steps=20,
        config=config,
    )

    await agent.chat(prompt="You> ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coding agent demo with filesystem + terminal + web tools",
    )
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode")
    return parser.parse_args()


async def main() -> None:
    config = HarnessConfig.load(PROJECT_ROOT / "arktor.yaml")
    args = parse_args()

    workspace = Path(tempfile.mkdtemp(prefix="coding_demo_")).resolve()
    print(f"Workspace: {workspace}")

    setup_mini_project(workspace)
    print("Mini project created.\n")

    patches = [
        patch(
            "agent_app.tools.filesystem._security.get_workspace_root",
            return_value=workspace,
        ),
        patch(
            "agent_app.tools.filesystem.list_dir.get_workspace_root",
            return_value=workspace,
        ),
        patch(
            "agent_app.tools.filesystem.glob_files.get_workspace_root",
            return_value=workspace,
        ),
        patch(
            "agent_app.tools.filesystem.grep_files.get_workspace_root",
            return_value=workspace,
        ),
        patch(
            "agent_app.tools.terminal.terminal_tool._workspace_root",
            return_value=workspace,
        ),
    ]

    try:
        for p in patches:
            p.start()

        if args.chat:
            await run_chat(config, workspace)
        else:
            await run_auto(config, workspace)
    finally:
        for p in reversed(patches):
            p.stop()
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
