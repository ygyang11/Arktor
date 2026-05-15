"""CLI visual theme: palette + Rich + prompt_toolkit bundle + persistence."""
from __future__ import annotations

from dataclasses import dataclass

from prompt_toolkit.output.color_depth import ColorDepth
from prompt_toolkit.styles import Style
from rich.theme import Theme as RichTheme

from agent_cli.runtime.prefs import read_prefs, write_prefs

# ── Color depth mapping ───────────────────────────────────────────────
DEPTH_MAP: dict[str, ColorDepth] = {
    "truecolor": ColorDepth.TRUE_COLOR,
    "256":       ColorDepth.DEPTH_8_BIT,
    "standard":  ColorDepth.DEPTH_4_BIT,
    "windows":   ColorDepth.DEPTH_4_BIT,
}


# ── Palette ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Palette:
    bg: str
    section_bg: str
    text: str
    muted: str
    primary: str
    secondary: str
    accent: str
    success: str
    error: str
    info: str
    diff_add_bg: str
    diff_remove_bg: str
    diff_hunk_bg: str
    scrollbar_bg: str
    shell_lane_bg: str


# ── Bundle ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CliTheme:
    """Paired Rich + prompt_toolkit styles — swap one to change the whole UI.

    ``palette`` is kept so render paths that bypass Rich/PT (raw ANSI writes,
    offline-rendered prefixes) can access hex values directly.
    """

    name: str
    rich: RichTheme
    completion: Style
    palette: Palette


def _build_theme(name: str, p: Palette) -> CliTheme:
    return CliTheme(
        name=name,
        palette=p,
        rich=RichTheme({
            "primary":     p.primary,
            "secondary":   p.secondary,
            "accent":      p.accent,
            "text":        p.text,
            "muted":       p.muted,
            "success":     p.success,
            "error":       p.error,
            "info":        p.info,
            "bg":          p.bg,
            "section":     f"on {p.section_bg}",
            "diff_add":    f"{p.text} on {p.diff_add_bg}",
            "diff_remove": f"{p.text} on {p.diff_remove_bg}",
            "diff_hunk":   f"{p.info} on {p.diff_hunk_bg}",
            "diff_meta":   p.muted,
        }),
        completion=Style.from_dict({
            "completion-menu":                         f"bg:{p.section_bg}",
            "completion-menu.completion":              f"bg:{p.section_bg} fg:{p.text}",
            "completion-menu.completion.current":      f"bg:{p.primary} fg:{p.text} bold",
            "completion-menu.meta.completion":         f"bg:{p.section_bg} fg:{p.muted} italic",
            "completion-menu.meta.completion.current": f"bg:{p.secondary} fg:{p.text}",
            "scrollbar.background":                    f"bg:{p.scrollbar_bg}",
            "scrollbar.button":                        f"bg:{p.muted}",
            "bottom-toolbar":                          f"noreverse fg:{p.secondary}",
            "bottom-toolbar.text":                     f"noreverse fg:{p.secondary} bold italic",
            "paste-placeholder":                       f"fg:{p.primary} bold",
            "shell-line":                              f"bg:{p.shell_lane_bg} fg:{p.primary} bold",
            "input-block":                             f"bg:{p.section_bg}",
        }),
    )


# ── Theme definitions ──────────────────────────────────────────────────
FLEXOKI_DARK = _build_theme("flexoki-dark", Palette(
    bg="#100F0F",
    section_bg="#2E2D2B",
    text="#CECDC3",
    muted="#878580",
    primary="#DA702C",
    secondary="#BC5215",
    accent="#D0A215",
    success="#3F9D36",
    error="#AF3029",
    info="#3AA99F",
    diff_add_bg="#002800",
    diff_remove_bg="#3d0000",
    diff_hunk_bg="#1C1B1A",
    scrollbar_bg="#1C1B1A",
    shell_lane_bg="#3D3935",
))


TOKYO_NIGHT = _build_theme("tokyo-night", Palette(
    bg="#1A1B26",
    section_bg="#24283B",
    text="#C0CAF5",
    muted="#565F89",
    primary="#7AA2F7",
    secondary="#706CEB",
    accent="#E0AF68",
    success="#9ECE6A",
    error="#F7768E",
    info="#7DCFFF",
    diff_add_bg="#1F3A2D",
    diff_remove_bg="#3A1F27",
    diff_hunk_bg="#1F2335",
    scrollbar_bg="#1F2335",
    shell_lane_bg="#2F344E",
))


CATPPUCCIN_MOCHA = _build_theme("catppuccin-mocha", Palette(
    bg="#1E1E2E",
    section_bg="#313244",
    text="#CDD6F4",
    muted="#6C7086",
    primary="#CBA6F7",
    secondary="#F5C2E7",
    accent="#FAB387",
    success="#A6E3A1",
    error="#F38BA8",
    info="#89DCEB",
    diff_add_bg="#2D3A2D",
    diff_remove_bg="#3A2D33",
    diff_hunk_bg="#313244",
    scrollbar_bg="#313244",
    shell_lane_bg="#3F4054",
))


ROSE_PINE = _build_theme("rose-pine", Palette(
    bg="#191724",
    section_bg="#26233A",
    text="#E0DEF4",
    muted="#908CAA",
    primary="#C4A7E7",
    secondary="#EA9A97",
    accent="#F6C177",
    success="#9CCFD8",
    error="#EB6F92",
    info="#3E8FB0",
    diff_add_bg="#283531",
    diff_remove_bg="#3A252D",
    diff_hunk_bg="#1F1D2E",
    scrollbar_bg="#1F1D2E",
    shell_lane_bg="#33304B",
))


KANAGAWA = _build_theme("kanagawa", Palette(
    bg="#1F1F28",
    section_bg="#2A2A37",
    text="#DCD7BA",
    muted="#727169",
    primary="#7E9CD8",
    secondary="#957FB8",
    accent="#DCA561",
    success="#76946A",
    error="#C34043",
    info="#7AA89F",
    diff_add_bg="#2B3328",
    diff_remove_bg="#3D2426",
    diff_hunk_bg="#2A2A37",
    scrollbar_bg="#2A2A37",
    shell_lane_bg="#37374A",
))


EVERFOREST_DARK = _build_theme("everforest-dark", Palette(
    bg="#2D353B",
    section_bg="#3D484D",
    text="#D3C6AA",
    muted="#859289",
    primary="#A7C080",
    secondary="#ACCF74",
    accent="#DBBC7F",
    success="#83C092",
    error="#E67E80",
    info="#7FBBB3",
    diff_add_bg="#3C4841",
    diff_remove_bg="#4E3E43",
    diff_hunk_bg="#384247",
    scrollbar_bg="#384247",
    shell_lane_bg="#4D5862",
))


DRACULA = _build_theme("dracula", Palette(
    bg="#282A36",
    section_bg="#44475A",
    text="#F8F8F2",
    muted="#6272A4",
    primary="#BD93F9",
    secondary="#FF79C6",
    accent="#F1FA8C",
    success="#50FA7B",
    error="#FF5555",
    info="#8BE9FD",
    diff_add_bg="#1F3A28",
    diff_remove_bg="#3A1F2C",
    diff_hunk_bg="#343746",
    scrollbar_bg="#343746",
    shell_lane_bg="#535672",
))


# ── Registry + default ─────────────────────────────────────────────────
THEMES: dict[str, CliTheme] = {
    t.name: t for t in (
        FLEXOKI_DARK,
        TOKYO_NIGHT,
        CATPPUCCIN_MOCHA,
        ROSE_PINE,
        KANAGAWA,
        EVERFOREST_DARK,
        DRACULA,
    )
}
DEFAULT_THEME: CliTheme = FLEXOKI_DARK


def available_names() -> list[str]:
    return sorted(THEMES)


# ── Glyphs ─────────────────────────────────────────────────────────────
PROMPT = "❯"
TOOL_DONE = "●"
TODO_IN_PROGRESS = "●"
TODO_PENDING = "○"
TASKS_HEADER = "▶"
SPINNER_STATIC = "⟳"
SPINNER_FRAMES: tuple[str, ...] = (
    "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
)
ELLIPSIS_FRAMES: tuple[str, ...] = (".  ", ".. ", "...")
APPROVAL = "!"
SUBAGENT = "◆"
SUBAGENT_DONE = "◇"
RUN_DONE = "✻"
COMPRESSION = "∵"
DENIED = "⊘"
SEP_DOT = "·"
SEP_ELLIPSIS = "…"
CONTINUATION = "⎿"
LEFT_BAR = "┃"
BAR_FILLED = "█"
BAR_EMPTY = "░"


# ── Persistence ────────────────────────────────────────────────────────
def load_saved_theme() -> CliTheme:
    """Resolve the persisted theme. Any failure falls back to ``DEFAULT_THEME``."""
    name = read_prefs().get("theme")
    if isinstance(name, str) and name in THEMES:
        return THEMES[name]
    return DEFAULT_THEME


def save_theme(name: str) -> None:
    """Persist ``name`` as the active theme. Raises ``KeyError`` if unknown."""
    if name not in THEMES:
        raise KeyError(
            f"Unknown theme: {name!r}. Available: {', '.join(available_names())}"
        )
    prefs = read_prefs()
    prefs["theme"] = name
    write_prefs(prefs)
