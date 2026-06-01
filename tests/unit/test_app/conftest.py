from __future__ import annotations

from pathlib import Path

import pytest


def _write_skill(root: Path, name: str, desc: str, body: str, **extra: str) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    fm_lines = [f"name: {name}", f"description: {desc}"]
    for k, v in extra.items():
        fm_lines.append(f"{k}: {v}")
    content = "---\n" + "\n".join(fm_lines) + "\n---\n\n" + body + "\n"
    md = skill_dir / "SKILL.md"
    md.write_text(content, encoding="utf-8")
    return skill_dir


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    root = tmp_path / "skills"
    pdf = _write_skill(root, "pdf", "Process PDF files.", "# PDF\n\nDo PDF things.", version="1.0")
    scripts = pdf / "scripts"
    scripts.mkdir()
    (scripts / "extract.py").write_text("# extract")
    _write_skill(root, "frontend", "Build frontend UIs.", "# Frontend\n\nBuild UIs.")
    return root


@pytest.fixture
def empty_skills_dir(tmp_path: Path) -> Path:
    root = tmp_path / "skills"
    root.mkdir()
    return root
