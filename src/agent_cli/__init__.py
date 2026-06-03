"""Arktor interactive CLI layer."""
from __future__ import annotations

__version__ = "0.5.2"

_MISSING_DEPS_TEMPLATE = (
    "Arktor CLI requires the following missing dependencies: {deps}.\n"
    "Install with: pip install 'arktor[cli]'"
)


def _check_deps() -> None:
    """Verify CLI runtime deps; exit(1) with a precise list if any are missing."""
    missing: list[str] = []
    try:
        import prompt_toolkit  # noqa: F401
    except ImportError:
        missing.append("prompt_toolkit")
    try:
        import rich  # noqa: F401
    except ImportError:
        missing.append("rich")
    if missing:
        import sys
        sys.stderr.write(_MISSING_DEPS_TEMPLATE.format(deps=", ".join(missing)) + "\n")
        raise SystemExit(1)
