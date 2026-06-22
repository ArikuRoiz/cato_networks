"""Langfuse observability initialisation (SDK v2 compatible with server v2).

Call ``setup_telemetry()`` once at process start. It validates credentials
and wires the Langfuse singleton. When credentials are absent all observations
are silently dropped.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_ENV_PUBLIC_KEY = "LANGFUSE_PUBLIC_KEY"
_ENV_SECRET_KEY = "LANGFUSE_SECRET_KEY"
_ENV_HOST = "LANGFUSE_HOST"
_DEFAULT_HOST = "https://cloud.langfuse.com"


def setup_telemetry(service_name: str = "the-ai-firm") -> None:
    """Initialise the Langfuse singleton if credentials are present."""
    public_key = os.environ.get(_ENV_PUBLIC_KEY, "")
    secret_key = os.environ.get(_ENV_SECRET_KEY, "")
    host = os.environ.get(_ENV_HOST, _DEFAULT_HOST)

    if not public_key or not secret_key:
        logger.debug("Langfuse credentials absent — tracing disabled.")
        return

    try:
        import langfuse  # noqa: F401 (import triggers singleton init via env vars)

        logger.info("Langfuse tracing enabled → %s", host)
    except Exception as exc:
        logger.warning("Langfuse init failed: %s", exc)


def flush_telemetry() -> None:
    """Flush all buffered Langfuse observations before process exit."""
    try:
        from langfuse.decorators import langfuse_context

        langfuse_context.flush()
    except Exception as exc:
        logger.warning("Langfuse flush failed: %s", exc)
