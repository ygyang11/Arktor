"""Skill loader with progressive disclosure.

Loading tiers:
    Tier 1 (startup): YAML frontmatter only (~100 tokens/skill).
    Tier 2 (on-demand): Full SKILL.md body when agent activates a skill.
    Tier 3 (agent-driven): Resource files in subdirectories, read as needed.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, NamedTuple

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class Skill(BaseModel):
    """A loaded skill with metadata and instructions."""

    name: str
    description: str
    body: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    path: Path
    dir: Path

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def list_resources(self) -> dict[str, list[Path]]:
        """Discover resource files in standard subdirectories."""
        resources: dict[str, list[Path]] = {}
        for folder in ("scripts", "references", "examples", "assets"):
            folder_path = self.dir / folder
            if folder_path.exists():
                files = sorted(f for f in folder_path.rglob("*") if f.is_file())
                if files:
                    resources[folder] = files
        return resources


class _SkillMeta(NamedTuple):
    name: str
    description: str
    metadata: dict[str, Any]
    path: Path
    dir: Path


class SkillLoader:
    """Loads skills from multiple directories using progressive disclosure."""

    def __init__(self, dirs: list[str | Path]) -> None:
        self._dirs = self._resolve_dirs(dirs)
        self._metadata: dict[str, _SkillMeta] = {}
        self._skills: dict[str, Skill] = {}
        self._scan()


    @classmethod
    def _resolve_dirs(cls, dirs: list[str | Path]) -> list[Path]:
        return [Path(directory).expanduser().resolve() for directory in dirs]

    @classmethod
    def build_state_key(cls, dirs: list[str | Path]) -> tuple[str, ...]:
        resolved_dirs = cls._resolve_dirs(dirs)
        entries: list[str] = []

        for skills_dir in resolved_dirs:
            if not skills_dir.exists():
                entries.append(f"DIR|{skills_dir}|missing")
                continue

            entries.append(f"DIR|{skills_dir}|ok")
            for entry in sorted(skills_dir.iterdir(), key=lambda p: p.name):
                if not entry.is_dir():
                    continue
                skill_md = entry / "SKILL.md"
                if not skill_md.exists():
                    continue
                try:
                    stat = skill_md.stat()
                except OSError:
                    continue
                entries.append(f"SKILL|{skill_md}|{stat.st_mtime_ns}|{stat.st_size}")

        return tuple(entries)

    @staticmethod
    def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str] | None:
        """Parse YAML frontmatter and body. Returns (meta, body) or None."""
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
        if not match:
            return None
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            return None
        if not isinstance(meta, dict) or "name" not in meta or "description" not in meta:
            return None
        return meta, match.group(2).strip()

    def _scan(self) -> None:
        for skills_dir in self._dirs:
            if not skills_dir.exists():
                continue
            for entry in sorted(skills_dir.iterdir()):
                if not entry.is_dir():
                    continue
                skill_md = entry / "SKILL.md"
                if not skill_md.exists():
                    continue
                try:
                    raw = skill_md.read_text(encoding="utf-8")
                except OSError:
                    logger.warning("Cannot read %s", skill_md)
                    continue
                parsed = self._parse_frontmatter(raw)
                if parsed is None:
                    logger.warning("Invalid SKILL.md: %s", skill_md)
                    continue
                meta, _ = parsed
                name: str = meta["name"]
                if name in self._metadata:
                    logger.warning(
                        "Skill '%s' in %s shadows %s (keeping first occurrence)",
                        name,
                        self._metadata[name].path,
                        skill_md,
                    )
                    continue
                extra = {k: v for k, v in meta.items() if k not in ("name", "description")}
                self._metadata[name] = _SkillMeta(
                    name=name,
                    description=meta["description"],
                    metadata=extra,
                    path=skill_md,
                    dir=entry,
                )

    def get_skill(self, name: str) -> Skill | None:
        """Load a skill by name (Tier 2). Returns None if not found."""
        if name in self._skills:
            return self._skills[name]
        meta = self._metadata.get(name)
        if meta is None:
            return None
        try:
            raw = meta.path.read_text(encoding="utf-8")
        except OSError:
            return None
        parsed = self._parse_frontmatter(raw)
        if parsed is None:
            return None
        fm, body = parsed
        extra = {k: v for k, v in fm.items() if k not in ("name", "description")}
        skill = Skill(
            name=fm["name"],
            description=fm["description"],
            body=body,
            metadata=extra,
            path=meta.path,
            dir=meta.dir,
        )
        self._skills[name] = skill
        return skill

    def get_catalog(self) -> str:
        """Format all skill frontmatter for embedding in tool description."""
        if not self._metadata:
            return "(no skills available)"
        lines: list[str] = []
        for m in self._metadata.values():
            line = f"- {m.name}: {m.description}"
            if m.metadata:
                meta_str = ", ".join(f"{k}: {v}" for k, v in m.metadata.items())
                line += f" ({meta_str})"
            lines.append(line)
        return "\n".join(lines)

    def list_names(self) -> list[str]:
        return list(self._metadata.keys())

    def reload(self) -> None:
        self._metadata.clear()
        self._skills.clear()
        self._scan()
