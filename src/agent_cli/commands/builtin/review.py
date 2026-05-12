"""/review — submit a code-review prompt scoped to a target."""
from __future__ import annotations

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.builtin._git import run as _run_git
from agent_cli.commands.ui import err, soft

_REVIEW_PROMPT = """You are reviewing a code change on behalf of the author. Your goal is to find real bugs and risks in this change, and report each one clearly enough that the author can act without further investigation.

Inspect the change indicated by the review focus below — use git as needed — and read enough surrounding code, callers, types, and tests to verify each potential finding end-to-end. Never flag based on what the surrounding code probably does.

Voice: matter-of-fact, brief, direct. No "Great job", thanking, or filler like "I reviewed your changes" or "Hope this helps".

The rest of this prompt is the default approach. If the user prompt or repo conventions provide more specific instructions — different bug categories, output schema, severity scheme, etc. — follow those instead.

## Review guidelines

Apply these when reviewing the change:

- **Materially impacts** correctness, security, performance, or maintainability — not pure style or aesthetic preference.
- **Discrete and actionable** — pinpointed to specific code with a clear fix direction. No general "this module is messy" findings.
- **Tied to this change** — introduced by the diff or meaningfully exacerbated by it. Pure pre-existing bugs that the diff doesn't touch or worsen are out of scope.
- **Verified, not speculative** — you've read the affected code (callers, dependents, related types, tests) and can identify how it breaks. Don't claim impact on code you haven't opened. Uncertainty is permitted only on author intent (e.g., "uncertain whether this is intentional"), never on whether the bug exists.
- **No hidden reasoning steps** — don't depend on unstated assumptions about codebase behavior or author intent.
- **Calibrated to codebase rigor** — don't ask for input validation in a one-off script repo, or extensive tests in a prototype, if the surrounding code doesn't.
- **Not just an intentional change** — if the diff clearly signals deliberate restructuring (renaming, reorganization, refactors), don't flag the new form just because it differs from the old.

## Output format

Tag each finding with **[P0]** (drop everything; blocking release/ops/major usage; reserve for issues independent of unusual inputs), **[P1]** (urgent; fix this cycle), or **[P2]** (normal; fix eventually). Anything below P2 doesn't clear the bar — drop it. The verdict is `needs fixes` if any finding is filed, otherwise `correct`.

File one issue per finding — if two issues sit on the same line with different root causes, file two findings.

Structure your output **exactly** as follows:

```markdown
## Findings

### [P#] Imperative title

**Location**: `path/to/file.ext:start-end` (smallest range that pinpoints the issue; may be an unchanged caller the diff broke)

A short paragraph (as concise as possible) explaining what the bug is, the trigger scenario (specific input, environment, or timing that causes it), and the impact. If a code reference clarifies the issue, include a fenced snippet of ≤3 lines below.

```lang
relevant code
```

### [P#] Next finding...

## Overall

**Verdict**: correct | needs fixes
**Confidence**: high | medium | low
**Rationale**: 1–2 sentences citing the most load-bearing finding(s).
```

## Quantity discipline

- Output **all** qualifying findings — continue scanning until the whole diff is covered. Don't stop at the first finding.
- Output **none** if nothing meets the bar. Do not pad the list to look thorough.

---

Review focus: {target}
"""

_DEFAULT_REVIEW_TARGET = "the uncommitted changes on this branch (staged and unstaged)"


async def _handler(ctx: CommandContext, args: str) -> CommandResult:
    target = args.strip()
    if not target:
        rc, status, _ = await _run_git("status", "--short", timeout=5.0)
        if rc != 0:
            return CommandResult(output=err(
                ("Not a git repo or git failed", ""),
            ))
        if not status:
            return CommandResult(output=soft(
                ("Working tree clean — nothing to review", ""),
            ))
        target = _DEFAULT_REVIEW_TARGET

    prompt = _REVIEW_PROMPT.format(target=target)
    return CommandResult(agent_input=prompt)


CMD = Command(
    name="/review",
    description="Review code changes, defaults to uncommitted changes (accepts focus target)",
    handler=_handler,
)
