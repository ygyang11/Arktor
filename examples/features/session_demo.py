"""Session demo — multi-turn conversations and persistence.

Usage:
    # Mode 1: Multiple run() calls share context within the same process
    python examples/features/session_demo.py

    # Mode 2: Interactive chat loop (terminal REPL)
    python examples/features/session_demo.py --chat

    # Mode 3: Chat with session persistence (survives process restart)
    python examples/features/session_demo.py --chat --session my-session

    # Mode 4: Resume a previous session
    python examples/features/session_demo.py --resume my-session
"""

import argparse
import asyncio
from pathlib import Path

from agent_harness import HarnessConfig, ReActAgent
from agent_app.tools import WEB_TOOLS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


async def demo_multi_turn(config: HarnessConfig) -> None:
    """Mode 1: Multiple run() calls share context automatically.
    No session needed — state lives in memory between calls."""
    print("=== Mode 1: Multi-turn run() (no session) ===\n")

    agent = ReActAgent(name="assistant", tools=[*WEB_TOOLS], config=config)

    queries = [
        "Search for the latest news about 5G satellite communication simply.",
        "Based on what you just found, which company is leading in this area?",
        "Now search simply for that company's recent partnerships.",
        "Summarize everything we've discussed so far.",
    ]

    for i, query in enumerate(queries, 1):
        print(f"[Turn {i}] User: {query}")
        result = await agent.run(query)
        print(f"[Turn {i}] Agent: {result.output}\n")

    print(f"Total messages in memory: {await agent.context.short_term_memory.size()}")
    # passing session= to run() is only needed for cross-process persistence.
    # Within the same process, run() naturally accumulates conversation history.


async def demo_chat(config: HarnessConfig, session_id: str | None = None) -> None:
    """Mode 2: Interactive chat loop via chat().
    Optionally pass session for persistence."""
    mode = "chat + session" if session_id else "chat (no session)"
    print(f"=== Mode 2: Interactive {mode} ===")
    print("Type 'exit' to quit.\n")

    agent = ReActAgent(name="assistant", tools=[*WEB_TOOLS], config=config)
    await agent.chat(session=session_id, prompt="You> ")


async def demo_resume(config: HarnessConfig, session_id: str) -> None:
    """Mode 3: Resume a previous session and continue chatting."""
    print(f"=== Mode 3: Resuming session '{session_id}' ===")
    print("Type 'exit' to quit.\n")

    agent = ReActAgent(name="assistant", tools=[*WEB_TOOLS], config=config)
    await agent.chat(session=session_id, prompt="You> ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Session demo")
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode")
    parser.add_argument("--session", type=str, default=None, help="Session ID for persistence")
    parser.add_argument("--resume", type=str, default=None, help="Resume a previous session")
    return parser.parse_args()


async def main() -> None:
    config = HarnessConfig.load(PROJECT_ROOT / "arktor.yaml")
    args = parse_args()

    if args.resume:
        await demo_resume(config, args.resume)
    elif args.chat:
        await demo_chat(config, args.session)
    else:
        await demo_multi_turn(config)


if __name__ == "__main__":
    asyncio.run(main())
