"""Command output formatting helpers — line builders + panel renderers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from agent_cli.commands.base import Command
from agent_cli.runtime.status import (
    BucketView, StatusSnapshot, UsageView, WindowView,
)
from agent_cli.theme import APPROVAL, BAR_EMPTY, BAR_FILLED, SEP_DOT, TOOL_DONE
from agent_harness.approval.policy import ApprovalPolicy
from agent_harness.approval.rules import PermissionRule
from agent_harness.memory.short_term import SectionWeights
from agent_harness.session.base import SessionMeta
from agent_harness.utils.token_counter import truncate_text_by_tokens

_FILL_BAR_WIDTH = 60

_SECTION_LABELS: dict[str, str] = {
    "system_prompt": "System prompt",
    "tools_schema": "Tools schema",
    "dynamic_system": "Dynamic Info",
    "history": "Messages",
}

_SECTION_STYLES: dict[str, str] = {
    "system_prompt": "info",
    "tools_schema": "secondary",
    "dynamic_system": "accent",
    "history": "success",
}

_Segment = str | tuple[str, str]


# ── format atoms ──────────────────────────────────────────────────────

def _fmt_tokens(n: int | None) -> str:
    if n is None:
        return "—"
    if n >= 10_000:
        return f"{n / 1000:.1f}k"
    return f"{n:,}"


def _hit_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator * 100 if denominator else 0.0


def _shorten_home(cwd: str) -> str:
    home = str(Path.home())
    return cwd.replace(home, "~", 1) if cwd.startswith(home) else cwd


def _format_cache_value(
    cache_read: int, cache_creation: int, total_input: int,
) -> str:
    if total_input == 0:
        return "—"
    rate = _hit_rate(cache_read, total_input)
    parts = [f"{cache_read:,} / {total_input:,} hit ({rate:.0f}%)"]
    if cache_creation > 0:
        parts.append(f"{cache_creation:,} newly created")
    return " · ".join(parts)


# ── render primitives ────────────────────────────────────────────────

def _section_label(label: str) -> Text:
    return Text(label, style="primary")


def _bar_heading(label: str) -> Text:
    """Section heading with left-bar accent (indented to match data rows)."""
    t = Text("  ")
    t.append("▌", style="primary")
    t.append(" ", style="muted")
    t.append(label, style="primary")
    return t


def _row(key: str, value: str, *, key_width: int = 9) -> Text:
    t = Text("  ")
    t.append(key.ljust(key_width), style="muted")
    t.append(" ")
    t.append(SEP_DOT, style="muted")
    t.append(" ")
    t.append(value)
    return t


def _inline_bar(percent: float, width: int = 20, *, style: str = "primary") -> Text:
    filled = round(percent / 100 * width)
    bar = Text()
    bar.append(BAR_FILLED * filled, style=style)
    bar.append(BAR_EMPTY * (width - filled), style="muted")
    return bar


# ── line builders (status messages) ──────────────────────────────────

def _build(glyph: str, glyph_style: str, parts: tuple[_Segment, ...]) -> Text:
    t = Text()
    t.append(f"{glyph} ", style=glyph_style)
    for p in parts:
        if isinstance(p, tuple):
            t.append(p[0], style=p[1])
        else:
            t.append(p)
    return t


def ok(*parts: _Segment) -> Text:
    """Success line — filled circle in success colour."""
    return _build(TOOL_DONE, "success", parts)


def info(*parts: _Segment) -> Text:
    """Neutral state announcement — filled circle in info colour."""
    return _build(TOOL_DONE, "info", parts)


def err(*parts: _Segment) -> Text:
    """Error line — exclamation in error colour."""
    return _build(APPROVAL, "error", parts)


def soft(*parts: _Segment) -> Text:
    """Soft no-op notice — mid-dot in muted colour."""
    return _build(SEP_DOT, "muted", parts)


# ── mode info + /permissions panel ───────────────────────────────────

@dataclass(frozen=True)
class _ModeInfo:
    label: str
    style: str
    desc: str = ""
    short: str = ""


MODE_INFO: dict[str, _ModeInfo] = {
    "auto":  _ModeInfo("Auto",  "success",
                       "Trusted commands run automatically",
                       "trusted commands auto-allowed"),
    "ask":   _ModeInfo("Ask",   "warning",
                       "Approve every tool call individually",
                       "every call needs approval"),
    "never": _ModeInfo("Yolo",  "primary",
                       "Runs anything without asking",
                       "all calls auto-allowed"),
    "plan":  _ModeInfo("Plan",  "accent"),
}


def _format_rule(r: PermissionRule) -> str:
    return f"{r.tool_name}({r.pattern})" if r.pattern else r.tool_name


def _render_rule_list(rules: list[PermissionRule]) -> list[Text]:
    if not rules:
        return [_row(" ", "—", key_width=1)]
    sorted_rules = sorted(rules, key=lambda r: (r.tool_name, r.pattern or ""))
    return [_row(" ", _format_rule(r), key_width=1) for r in sorted_rules]


def _render_grants(grants: dict[str, list[list[str]] | None]) -> list[Text]:
    if not grants:
        return [_row(" ", "—", key_width=1)]
    out: list[Text] = []
    for tool, items in sorted(grants.items()):
        if items is None:
            out.append(_row(tool, "(any)"))
            continue
        for prefix, kind in items:
            out.append(_row(tool, f"{kind}: {prefix}"))
    return out


def render_permissions_panel(policy: ApprovalPolicy) -> RenderableType:
    cur = policy.mode
    items: list[RenderableType] = [_section_label("Mode")]
    for internal in ("auto", "ask", "never"):
        info = MODE_INFO[internal]
        is_cur = internal == cur
        line = Text("  ")
        line.append("▌" if is_cur else " ", style=info.style)
        line.append(" ")
        line.append(info.label.ljust(6), style="primary" if is_cur else "muted")
        line.append(info.desc, style="muted")
        items.append(line)
    items += [
        Text(""), _section_label("Always allow"),
        *_render_rule_list(policy._allow_rules),
        Text(""), _section_label("Always deny"),
        *_render_rule_list(policy._deny_rules),
        Text(""), _section_label("Session grants"),
        *_render_grants(policy.export_session_grants()),
    ]
    return Panel(
        Group(*items),
        title="Permissions", title_align="left",
        border_style="muted", padding=(0, 1), expand=False,
    )


# ── /resume session list ─────────────────────────────────────────────

_RESUME_LIMIT = 10


def relative_time(dt: datetime) -> str:
    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    delta = now - dt
    secs = delta.total_seconds()
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    if delta.days < 7:
        return f"{delta.days}d ago"
    return dt.strftime("%b %d")


# ── /skill list ─────────────────────────────────────────────

_SKILL_LIST_LIMIT = 15
_SKILL_DESC_TOKEN_LIMIT = 20


def render_skill_list(cmds: list[Command]) -> RenderableType:
    visible = cmds[:_SKILL_LIST_LIMIT]
    truncated = len(cmds) - len(visible)

    rows: list[RenderableType] = [info("Available skills"), Text("")]
    name_w = max((len(c.name) for c in visible), default=0)
    for c in visible:
        line = Text("  ")
        line.append(c.name.ljust(name_w + 2), style="primary")
        desc = truncate_text_by_tokens(
            c.description.strip().replace("\n", " "),
            max_tokens=_SKILL_DESC_TOKEN_LIMIT,
            suffix="…",
        )
        line.append(desc, style="muted")
        rows.append(line)

    if truncated > 0:
        rows.append(Text(""))
        rows.append(soft((f"… and {truncated} more", "")))
    return Group(*rows)


def render_session_list(metas: list[SessionMeta]) -> RenderableType:
    visible = metas[:_RESUME_LIMIT]
    truncated = len(metas) - len(visible)

    rows: list[RenderableType] = [info("Recent sessions"), Text("")]
    for m in visible:
        line = Text("  ")
        line.append(m.session_id, style="primary")
        line.append("  ")
        line.append(relative_time(m.updated_at).rjust(10), style="muted")
        line.append("  ")
        line.append(f"{m.message_count} msg".rjust(7), style="muted")
        line.append("  ")
        preview = m.first_user_preview or "(no user input)"
        line.append(preview)
        rows.append(line)

    if truncated > 0:
        rows.append(Text(""))
        rows.append(soft((f"… and {truncated} more", "")))
    return Group(*rows)


# ── /status panel ────────────────────────────────────────────────────

def render_status_panel(
    snap: StatusSnapshot, session_id: str, branch: str
) -> Panel:
    items: list[RenderableType] = [
        _section_label("Identity"),
        _row("session", session_id),
        _row("model", snap.model),
        _row("cwd", _shorten_home(str(Path.cwd()))),
        _row("branch", branch),
        Text(""),
        _section_label("Config"),
        _row("approval", snap.approval_mode),
        _row("tools", f"{snap.tool_count} registered"),
        _row("skills", str(snap.skill_count)),
        Text(""),
        _section_label("Runtime"),
        _row(
            "messages",
            f"{snap.message_count} "
            f"({_fmt_tokens(snap.input_tokens)}/{_fmt_tokens(snap.max_tokens)} tokens)",
        ),
        _row("todos", str(snap.todo_count)),
        _row("tasks", f"{snap.bg_running} running / {snap.bg_total} total"),
    ]
    return Panel(
        Group(*items),
        title="Status",
        title_align="left",
        border_style="muted",
        padding=(0, 1),
        expand=False,
    )


# ── /context panel ───────────────────────────────────────────────────

def _calibrate(weights: SectionWeights, total: int) -> dict[str, int]:
    items: dict[str, int] = {
        "system_prompt": weights.system_prompt,
        "tools_schema": weights.tools_schema,
        "dynamic_system": weights.dynamic_system,
        "history": weights.history,
    }
    local_sum = sum(items.values())
    if local_sum <= 0:
        return {k: 0 for k in items} | {"history": total}
    out = {k: round(v * total / local_sum) for k, v in items.items()}
    drift = total - sum(out.values())
    if drift:
        biggest = max(out, key=lambda k: out[k])
        out[biggest] += drift
    return out


def _render_fill_bar(input_tokens: int, max_tokens: int) -> Text:
    if max_tokens <= 0:
        return Text("")
    filled = round(input_tokens / max_tokens * _FILL_BAR_WIDTH)
    filled = max(0, min(_FILL_BAR_WIDTH, filled))
    bar = Text("  ")
    bar.append(BAR_FILLED * filled, style="primary")
    bar.append(BAR_EMPTY * (_FILL_BAR_WIDTH - filled), style="muted")
    return bar


def _render_section_rows(sections: dict[str, int], total_input: int) -> Group:
    rows: list[RenderableType] = []
    val_w = max(
        (len(f"{sections.get(k, 0):,}") for k in _SECTION_LABELS),
        default=0,
    )
    for key in _SECTION_LABELS:
        tokens = sections.get(key, 0)
        pct = _hit_rate(tokens, total_input)
        section_style = _SECTION_STYLES[key]
        line = Text("  ")
        line.append(BAR_FILLED + " ", style=section_style)
        line.append(_SECTION_LABELS[key].ljust(16))
        line.append(_inline_bar(pct, style=section_style))
        line.append("  ")
        line.append(f"{tokens:>{val_w},}", style="default")
        line.append(f"  ({pct:>3.0f}%)", style="muted")
        rows.append(line)
    return Group(*rows)


def render_context_panel(view: WindowView) -> RenderableType:
    if view.last_call is None or view.displayed_input_tokens is None:
        return info("Context window — No consume yet")

    last = view.last_call
    displayed = view.displayed_input_tokens
    sections = _calibrate(last.section_weights, last.input_tokens)
    sections["history"] += displayed - last.input_tokens
    pct = _hit_rate(displayed, view.max_tokens)

    summary_rows: list[RenderableType] = [
        _row("Model", last.model),
        _row(
            "Tokens",
            f"{displayed:,} / {view.max_tokens:,} ({pct:.0f}%)",
        ),
        _row("Cache", _format_cache_value(
            last.cache_read, last.cache_creation, last.input_tokens,
        )),
    ]

    return Group(
        info("Context window"),
        Text(""),
        *summary_rows,
        Text(""),
        _render_fill_bar(displayed, view.max_tokens),
        Text(""),
        _render_section_rows(sections, displayed),
    )


# ── /usage panel ─────────────────────────────────────────────────────

def _bucket_col_widths(
    *bucket_groups: dict[str, BucketView],
) -> tuple[int, int, int]:
    """Compute (name_w, in_w, out_w) across all bucket groups so multiple
    sections render with the same column boundaries."""
    all_buckets = [b for group in bucket_groups for b in group.values()]
    all_keys = [k for group in bucket_groups for k in group]
    name_w = max((len(k) for k in all_keys), default=0)
    in_w = max(
        (len(f"{b.usage.prompt_tokens:,}") for b in all_buckets), default=0,
    )
    out_w = max(
        (len(f"{b.usage.completion_tokens:,}") for b in all_buckets), default=0,
    )
    return name_w, in_w, out_w


def _render_buckets(
    buckets: dict[str, BucketView],
    heading: str,
    total_input: int,
    *,
    name_w: int,
    in_w: int,
    out_w: int,
) -> Group:
    rows: list[RenderableType] = [_bar_heading(heading)]
    sorted_items = sorted(
        buckets.items(), key=lambda kv: -kv[1].usage.prompt_tokens,
    )
    for key, bucket in sorted_items:
        pct = _hit_rate(bucket.usage.prompt_tokens, total_input)
        call_word = "call" if bucket.calls == 1 else "calls"
        line = Text("  ")
        line.append(key.ljust(name_w + 2))
        line.append(f"{pct:>3.0f}% ", style="muted")
        line.append(_inline_bar(pct))
        line.append("  ")
        line.append(f"{bucket.usage.prompt_tokens:>{in_w},}", style="default")
        line.append(" in · ", style="muted")
        line.append(f"{bucket.usage.completion_tokens:>{out_w},}", style="default")
        line.append(f" out  ({bucket.calls} {call_word})", style="muted")
        rows.append(line)
    return Group(*rows)


def render_usage_panel(view: UsageView) -> RenderableType:
    if view.call_count == 0:
        return info("Session usage — No consume yet")

    output_value = f"{view.total.completion_tokens:,}"
    if view.total.reasoning_tokens > 0:
        rate = _hit_rate(view.total.reasoning_tokens, view.total.completion_tokens)
        output_value += (
            f" ({view.total.reasoning_tokens:,} reasoning, {rate:.0f}%)"
        )

    summary_rows: list[RenderableType] = [
        _row("Input", f"{view.total.prompt_tokens:,}"),
        _row("Output", output_value),
        _row("Calls", str(view.call_count)),
        _row("Cache", _format_cache_value(
            view.total.cache_read_tokens,
            view.total.cache_creation_tokens,
            view.total.prompt_tokens,
        )),
    ]

    name_w, in_w, out_w = _bucket_col_widths(view.by_model, view.by_source)
    return Group(
        info("Session usage"),
        Text(""),
        *summary_rows,
        Text(""),
        _render_buckets(
            view.by_model, "By model", view.total.prompt_tokens,
            name_w=name_w, in_w=in_w, out_w=out_w,
        ),
        Text(""),
        _render_buckets(
            view.by_source, "By source", view.total.prompt_tokens,
            name_w=name_w, in_w=in_w, out_w=out_w,
        ),
    )
