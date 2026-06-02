"""File-based session backend with atomic writes."""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from agent_harness.session.base import BaseSession, SessionMeta, SessionState

logger = logging.getLogger(__name__)

_DEFAULT_SESSION_DIR = Path.home() / ".arktor" / "sessions"


class FileSession(BaseSession):

    def __init__(self, session_id: str, path: str | Path | None = None) -> None:
        super().__init__(session_id)
        self._dir = Path(path) if path is not None else _DEFAULT_SESSION_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def _file_path(self) -> Path:
        return self._dir / f"{self.session_id}.json"

    async def load_state(self) -> SessionState | None:
        if not self._file_path.exists():
            return None
        try:
            raw = self._file_path.read_text(encoding="utf-8")
            return SessionState.model_validate_json(raw)
        except Exception:
            logger.warning("Corrupted session file: %s", self._file_path)
            return None

    async def save_state(self, state: SessionState) -> None:
        data = state.model_dump_json(indent=2)
        fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        os.close(fd)
        tmp_path = Path(tmp)
        try:
            tmp_path.write_text(data, encoding="utf-8")
            tmp_path.replace(self._file_path)
        except BaseException:
            logger.warning("Failed to save session %s", self.session_id)
            tmp_path.unlink(missing_ok=True)
            raise

    async def rename(self, new_id: str) -> None:
        old_path = self._file_path
        self.set_session_id(new_id)
        new_path = self._file_path
        if old_path != new_path and old_path.exists():
            old_path.rename(new_path)

    async def clear(self) -> None:
        if self._file_path.exists():
            self._file_path.unlink()

    async def has_session(self, session_id: str) -> bool:
        if not self._is_valid_id(session_id):
            return False
        return (self._dir / f"{session_id}.json").exists()

    async def list_states(self) -> list[SessionMeta]:
        metas: list[SessionMeta] = []
        for fp in self._dir.glob("*.json"):
            try:
                state = SessionState.model_validate_json(fp.read_text(encoding="utf-8"))
                metas.append(SessionMeta.from_state(state))
            except Exception:
                logger.debug("Skip corrupted session file: %s", fp)
        return sorted(metas, key=lambda m: m.updated_at, reverse=True)
