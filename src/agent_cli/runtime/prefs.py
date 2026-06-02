"""CLI preferences: atomic JSON read/write at ~/.arktor/cli-prefs.json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PREFS_PATH = Path.home() / ".arktor" / "cli-prefs.json"


def read_prefs() -> dict[str, Any]:
    try:
        with PREFS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_prefs(prefs: dict[str, Any]) -> None:
    PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PREFS_PATH.with_suffix(PREFS_PATH.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(prefs, f, indent=2, sort_keys=True)
    tmp.replace(PREFS_PATH)
