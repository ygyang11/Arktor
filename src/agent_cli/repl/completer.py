"""REPL input completer — routes by text-before-cursor context."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import (
    CompleteEvent,
    Completer,
    Completion,
    FuzzyCompleter,
    WordCompleter,
)
from prompt_toolkit.document import Document

from agent_cli.commands.registry import CommandRegistry
from agent_cli.repl.mentions import find_at_token
from agent_cli.runtime.file_index import IGNORE_DIRS, list_project_files


class _RoutedCompleter(Completer):
    def __init__(self, slash: Completer, file: Completer) -> None:
        self._slash = slash
        self._file = file

    def get_completions(
        self, document: Document, complete_event: CompleteEvent,
    ) -> Iterable[Completion]:
        if document.text.startswith("!"):
            return
        text = document.text_before_cursor
        if text.startswith("/") and " " not in text:
            yield from self._slash.get_completions(document, complete_event)
            return
        if find_at_token(text) is not None:
            yield from self._file.get_completions(document, complete_event)

    def invalidate_file_root(self, root: Path) -> None:
        if isinstance(self._file, AtFileCompleter):
            self._file.invalidate(root)


def _build_slash_completer(registry: CommandRegistry) -> Completer:
    pairs = registry.get_completions()
    words = [n for n, _ in pairs]
    meta = {n: d for n, d in pairs}
    base = WordCompleter(words=words, meta_dict=meta, WORD=True)
    return FuzzyCompleter(base, WORD=True)


class AtFileCompleter(Completer):
    """`@`-gated cwd-relative completer.

    Prefer live directory children for the active prefix so nested directory
    completion keeps working even when the startup cache missed that subtree.
    The workspace-wide cache remains as a fallback for non-existent prefixes.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root_resolved = root.resolve()
        self._files: list[str] | None = None

    def invalidate(self, root: Path) -> None:
        self._root = root
        self._root_resolved = root.resolve()
        self._files = None

    def get_completions(
        self, document: Document, complete_event: CompleteEvent,
    ) -> Iterable[Completion]:
        hit = find_at_token(document.text_before_cursor)
        if hit is None:
            return
        _, fragment = hit

        if fragment.startswith(("/", "~")):
            return  # cache is cwd-relative, can't speak to absolute / home

        # Strip explicit cwd prefix so cache lookup (repo-relative) matches.
        while fragment.startswith("./"):
            fragment = fragment[2:]

        dir_part, _, basename_prefix = fragment.rpartition("/")
        listing_prefix = f"{dir_part}/" if dir_part else ""
        show_hidden = basename_prefix.startswith(".")

        source = self._live_entries(listing_prefix)
        if source is None:
            source = self._cached()
        for entry in _rank(source, listing_prefix, basename_prefix, show_hidden):
            shown = entry[len(listing_prefix):]
            yield Completion(
                text=shown,
                start_position=-len(basename_prefix),
                display=shown,
                display_meta="dir" if entry.endswith("/") else "",
            )

    def _cached(self) -> list[str]:
        if self._files is None:
            self._files = list_project_files(self._root)
        return self._files

    def _live_entries(self, listing_prefix: str) -> list[str] | None:
        target = self._root if not listing_prefix else self._root / listing_prefix
        try:
            resolved = target.resolve()
            resolved.relative_to(self._root_resolved)
        except (OSError, ValueError):
            return None
        if not resolved.is_dir():
            return None

        out: list[str] = []
        try:
            for child in sorted(resolved.iterdir(), key=lambda p: (p.name.casefold(), p.name)):
                if child.name in IGNORE_DIRS:
                    continue
                rel = child.relative_to(self._root).as_posix()
                out.append(rel + "/" if child.is_dir() else rel)
        except OSError:
            return None
        return out


def _rank(
    files: list[str],
    listing_prefix: str,
    basename_prefix: str,
    show_hidden: bool,
) -> Iterable[str]:
    """One-level-deep candidates under listing_prefix, prefix > substring."""
    seen: set[str] = set()
    pref_len = len(listing_prefix)
    exact: list[str] = []
    substring: list[str] = []
    for f in files:
        if not f.startswith(listing_prefix):
            continue
        tail = f[pref_len:]
        if not tail:
            continue
        stripped = tail.rstrip("/")
        if "/" in stripped:
            continue
        if stripped in seen:
            continue
        if stripped.startswith(".") and not show_hidden:
            continue
        seen.add(stripped)
        if stripped.startswith(basename_prefix):
            exact.append(f)
        elif basename_prefix and basename_prefix.lower() in stripped.lower():
            substring.append(f)
    yield from exact
    yield from substring


def build_input_completer(registry: CommandRegistry) -> Completer:
    return _RoutedCompleter(
        slash=_build_slash_completer(registry),
        file=AtFileCompleter(Path.cwd()),
    )


def refresh_input_completer(
    pt_session: PromptSession[str], registry: CommandRegistry,
) -> None:
    pt_session.completer = build_input_completer(registry)
