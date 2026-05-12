"""/init — generate or update AGENTS.md for the repository."""
from __future__ import annotations

from pathlib import Path

from agent_cli.commands.base import Command, CommandContext, CommandResult

_INIT_NEW = """Generate a file named AGENTS.md that serves as a contributor guide for this repository.
Your goal is to produce a clear, concise, and well-structured document with descriptive headings and actionable explanations for each section.
Follow the outline below, but adapt as needed — add sections if relevant, and omit those that do not apply to this project.

Start by deeply exploring the codebase — read enough source files, manifests, and tests to understand the actual patterns, not just list directory structure. The document is only as good as the understanding behind it.

Document Requirements

- Title the document "Repository Guidelines".
- Use Markdown headings (#, ##, etc.) for structure.
- Keep the document concise. 200–400 words is optimal.
- Keep explanations short, direct, and specific to this repository.
- Provide examples where helpful (commands, directory paths, naming patterns).
- Maintain a professional, instructional tone.

Recommended Sections

Project Structure & Module Organization
- Outline the project structure, including where the source code, tests, and assets are located.

Build, Test, and Development Commands
- List key commands for building, testing, and running locally (e.g., npm test, make build).
- Briefly explain what each command does.

Coding Style & Naming Conventions
- Specify indentation rules, language-specific style preferences, and naming patterns.
- Include any formatting or linting tools used.

Testing Guidelines
- Identify testing frameworks and coverage requirements.
- State test naming conventions and how to run tests.

Commit & Pull Request Guidelines
- Summarize commit message conventions found in the project's Git history.
- Outline pull request requirements (descriptions, linked issues, screenshots, etc.).

(Optional) Add other sections if relevant, such as Security & Configuration Tips, Architecture Overview, or Agent-Specific Instructions.{focus}"""

_INIT_UPDATE = """{target} already exists. Update it in place — do not rewrite from scratch.

1. Read {target} fully.
2. Deeply probe the current repo — read source files, manifests, and tests to ground your understanding in the codebase's current state. Don't rely on a quick scan.
3. Keep lines still accurate, delete what's outdated or contradicted, add what you newly discovered or what the user wants emphasized.

Stay concise (around 200–400 words remains optimal), preserve the existing heading structure (`# Repository Guidelines`, `##` sections), and maintain a professional, repository-specific tone.{focus}"""


async def _handler(ctx: CommandContext, args: str) -> CommandResult:
    target_exists = Path("AGENTS.md").exists()
    focus = f"\n\nFocus: {args.strip()}" if args.strip() else ""
    if target_exists:
        prompt = _INIT_UPDATE.format(target="AGENTS.md", focus=focus)
    else:
        prompt = _INIT_NEW.format(focus=focus)
    return CommandResult(agent_input=prompt)


CMD = Command(
    name="/init",
    description="Generate or update AGENTS.md, accepts an optional focus",
    handler=_handler,
)
