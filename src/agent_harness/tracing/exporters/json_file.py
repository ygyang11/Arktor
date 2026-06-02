"""JSON file exporter for trace spans.

Writes spans as JSON Lines (one JSON object per line) for analysis.
Default file path uses a timestamp for unique naming.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from agent_harness.tracing.tracer import Span


def _default_path() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"./traces/{ts}.jsonl")


class JsonFileExporter:
    """Export spans to a JSON Lines file."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else _default_path()

    def export(self, spans: list[Span]) -> None:
        """Append spans to the JSON Lines file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a") as f:
            for span in spans:
                line = span.model_dump_json()
                f.write(line + "\n")

    def read_all(self) -> list[Span]:
        """Read all spans from the file."""
        if not self._path.exists():
            return []
        spans = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if line:
                    spans.append(Span.model_validate_json(line))
        return spans
