"""Logging configuration for agent_harness framework."""
from __future__ import annotations

import logging

_configured = False


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the agent_harness and agent_app namespaces.

    Only touches loggers under 'agent_harness.*' and 'agent_app.*'.
    Does NOT modify the root logger or other libraries' loggers.

    Can be called multiple times to change the level.
    """
    global _configured

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if not _configured:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s:%(lineno)d %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

        for name in ("agent_harness", "agent_app"):
            ns_logger = logging.getLogger(name)
            ns_logger.addHandler(handler)
            ns_logger.setLevel(numeric_level)
            ns_logger.propagate = False

        for name in (
            "openai", "anthropic", "httpx", "urllib3", "httpcore",
            "docker", "trafilatura", "pypdf",
        ):
            logging.getLogger(name).setLevel(logging.WARNING)

        _configured = True
    else:
        for name in ("agent_harness", "agent_app"):
            logging.getLogger(name).setLevel(numeric_level)
