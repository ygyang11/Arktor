import logging
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

from agent_cli.config import ConfigLoadResult, attach_rich_logging, load_config
from agent_harness.utils.logging_config import setup_logging


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    return home


@pytest.fixture
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


def test_project_config_wins(
    isolated_home: Path, isolated_cwd: Path,
) -> None:
    project = isolated_cwd / "arktor.yaml"
    project.write_text("llm:\n  provider: openai\n")
    with patch("agent_harness.core.config.HarnessConfig.load") as mock_load:
        result = load_config()
    assert result == ConfigLoadResult(path=project, bootstrapped=False)
    mock_load.assert_called_once_with(project)


def test_bootstrap_from_repo_template_when_user_config_missing(
    isolated_home: Path, isolated_cwd: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_repo = isolated_cwd.parent / "repo"
    fake_pkg_dir = fake_repo / "src" / "agent_cli"
    fake_pkg_dir.mkdir(parents=True)
    (fake_repo / "arktor_example.yaml").write_text("# example\nllm:\n  provider: anthropic\n")
    fake_init = fake_pkg_dir / "__init__.py"
    fake_init.write_text("")

    import agent_cli as _agent_cli
    monkeypatch.setattr(_agent_cli, "__file__", str(fake_init))

    with patch("agent_harness.core.config.HarnessConfig.load") as mock_load:
        result = load_config()

    user_cfg = isolated_home / ".arktor" / "arktor.yaml"
    assert result == ConfigLoadResult(path=user_cfg, bootstrapped=True)
    assert user_cfg.exists()
    assert "anthropic" in user_cfg.read_text()
    mock_load.assert_called_once_with(user_cfg)


def test_fallback_when_template_missing(
    isolated_home: Path, isolated_cwd: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pkg = isolated_cwd.parent / "pkg" / "agent_cli" / "__init__.py"
    fake_pkg.parent.mkdir(parents=True)
    fake_pkg.write_text("")

    import agent_cli as _agent_cli
    monkeypatch.setattr(_agent_cli, "__file__", str(fake_pkg))

    with patch("agent_harness.core.config.HarnessConfig.load") as mock_load:
        result = load_config()

    assert result == ConfigLoadResult(path=None, bootstrapped=False)
    mock_load.assert_called_once_with(None, env_override=True)


@pytest.fixture
def restore_logging() -> None:
    loggers = [logging.getLogger(n) for n in ("agent_harness", "agent_app")]
    loggers.append(logging.getLogger())  # root — attach_rich_logging touches it
    saved = {
        lg: (list(lg.handlers), lg.level, lg.propagate) for lg in loggers
    }
    yield
    for lg, (handlers, level, propagate) in saved.items():
        lg.handlers[:] = handlers
        lg.setLevel(level)
        lg.propagate = propagate


def test_setup_logging_pins_pypdf_to_warning(
    restore_logging: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pypdf belongs in the third-party WARNING bucket like the other
    libs — its noise is real WARNING-level and rendered (not corrupting)
    once root routes through rich; it must NOT be force-silenced to ERROR."""
    import agent_harness.utils.logging_config as lc

    monkeypatch.setattr(lc, "_configured", False)
    pypdf_logger = logging.getLogger("pypdf")
    saved = pypdf_logger.level
    try:
        pypdf_logger.setLevel(logging.NOTSET)
        lc.setup_logging("DEBUG")
        assert pypdf_logger.level == logging.WARNING
    finally:
        pypdf_logger.setLevel(saved)


def test_attach_rich_logging_routes_root_for_third_party(
    restore_logging: None,
) -> None:
    """Third-party libs (pypdf/httpx/docker/…) propagate to root. Without
    a handler there they hit logging.lastResort and write raw to stderr,
    corrupting the Live region. attach_rich_logging must put the rich
    handler on root so those records render above the Live instead."""
    attach_rich_logging(Console())
    root = logging.getLogger()
    rich_handlers = [h for h in root.handlers if h.get_name() == "cli-rich"]
    assert len(rich_handlers) == 1

    # Idempotent: a second call must not stack duplicate handlers on root.
    attach_rich_logging(Console())
    rich_handlers = [h for h in root.handlers if h.get_name() == "cli-rich"]
    assert len(rich_handlers) == 1


def test_attach_rich_logging_handler_does_not_filter(
    restore_logging: None,
) -> None:
    attach_rich_logging(Console())
    setup_logging("DEBUG")

    lg = logging.getLogger("agent_harness")
    (handler,) = [h for h in lg.handlers if h.get_name() == "cli-rich"]
    assert handler.level == logging.NOTSET
    assert lg.level == logging.DEBUG
    assert lg.isEnabledFor(logging.DEBUG)


def test_attach_rich_logging_preserves_non_stream_handlers(
    restore_logging: None, tmp_path: Path,
) -> None:
    lg = logging.getLogger("agent_harness")
    setup_logging("WARNING")
    file_handler = logging.FileHandler(tmp_path / "x.log")
    lg.addHandler(file_handler)

    attach_rich_logging(Console())
    attach_rich_logging(Console())  # idempotent

    assert file_handler in lg.handlers
    assert not any(type(h) is logging.StreamHandler for h in lg.handlers)
    assert len([h for h in lg.handlers if h.get_name() == "cli-rich"]) == 1
    file_handler.close()
